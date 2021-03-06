# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Cortical substrates for exploratory decisions in humans.
                Daw, N. D., O’Doherty, J. P., Dayan, P., Dolan, R. J., & Seymour, B. (2006).
                Nature, 441(7095), 876–9. https://doi.org/10.1038/nature04766
"""
import logging

import numpy as np

from model.modelTemplate import Model


class QLearnK(Model):

    """The q-Learning Kalman algorithm

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Parameters
    ----------
    sigma : float, optional
        Uncertainty scale measure
    sigmaG : float, optional
        Uncertainty measure growth
    drift : float, optional
        The drift rate
    beta : float, optional
        Sensitivity parameter for probabilities. Also known as an exploration-
        exploitation parameter. Defined as :math:`\\beta` in the paper
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    number_cues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    number_critics : integer, optional
        The number of different reaction learning sets.
        Default number_actions*number_cues
    action_codes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((number_actions, number_cues)) / number_critics)``
    expect : array of floats, optional
        The initialisation of the expected reward.
        Default ``ones((number_actions, number_cues)) * 5 / number_cues``
    sigmaA : array of floats, optional
        The initialisation of the uncertainty measure
    alphaA : array of floats, optional
        The initialisation of the learning rates
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.discrete.weightProb
    """

    def __init__(self, beta=4, sigma=1, sigmaG=1, drift=1, sigmaA=None, alphaA=None, invBeta=None, expect=None, **kwargs):

        super(QLearnK, self).__init__(**kwargs)

        if invBeta is not None:
            beta = (1 / invBeta) - 1
        self.beta = beta

        self.sigma = sigma
        self.sigmaG = sigmaG
        self.drift = drift

        if sigmaA is None:
            sigmaA = np.ones(self.number_actions)
        self.sigmaA = sigmaA
        if alphaA is None:
            alphaA = np.ones(self.number_actions)
        self.alphaA = alphaA

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect
        self.expectations0 = self.expectations.copy()


        self.parameters["sigma"] = self.sigma
        self.parameters["sigmaG"] = self.sigmaG
        self.parameters["beta"] = self.beta
        self.parameters["lambda"] = self.drift
        self.parameters["expectation"] = self.expectations.copy()

        # Recorded information
        self.recsigmaA = []
        self.recalphaA = []

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["sigmaA"] = np.array(self.recsigmaA).T
        results["alphaA"] = np.array(self.recalphaA).T

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recsigmaA.append(self.sigmaA.copy())
        self.recalphaA.append(self.alphaA.copy())

    def rewardExpectation(self, observation):
        """Calculate the estimated reward based on the action and stimuli

        This contains parts that are task dependent

        Parameters
        ----------
        observation : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        actionExpectations : array of floats
            The expected rewards for each action
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimulus_shaper.processStimulus(observation)

        actionExpectations = self._actExpectations(self.expectations, stimuli)

        return actionExpectations, stimuli, activeStimuli

    def delta(self, reward, expectation, action, stimuli):
        """
        Calculates the comparison between the reward and the expectation

        Parameters
        ----------
        reward : float
            The reward value
        expectation : float
            The expected reward value
        action : int
            The chosen action
        stimuli : {int | float | tuple | None}
            The stimuli received

        Returns
        -------
        delta
        """

        modReward = self.reward_shaper.processFeedback(reward, action, stimuli)

        delta = modReward - expectation

        return delta

    def updateModel(self, delta, action, stimuli, stimuliFilter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this trialstep

        """

        # Find the new activities
        self._newExpect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        actExpectations = self._actExpectations(self.expectations, stimuli)
        self.probabilities = self.calcProbabilities(actExpectations)

    def _newExpect(self, action, delta, stimuli):

        alphaA = self.sigmaA / (self.sigmaA + self.sigma)
        self.alphaA = alphaA

        newExpectations = self.expectations.copy()
        newExpectations[action] = self.expectations[action] + self.alphaA[action]*delta*stimuli/np.sum(stimuli)
        newExpectations = newExpectations * (newExpectations >= 0)
        self.expectations = self.drift * newExpectations + (1-self.drift) * self.expectations0

        newsigmaA = self.sigmaA.copy()
        newsigmaA[action] = (1-alphaA[action]) * self.sigmaA[action]
        self.sigmaA = (self.drift**2) * newsigmaA + self.sigmaG

    def _actExpectations(self, expectations, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            actionExpectations = self.actStimMerge(expectations, stimuli)
        else:
            actionExpectations = expectations

        return actionExpectations

    def calcProbabilities(self, actionValues):
        # type: (np.ndarray) -> np.ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        actionValues : 1D ndArray of floats

        Returns
        -------
        probArray : 1D ndArray of floats
            The probabilities associated with the actionValues
        """

        numerator = np.exp(self.beta * actionValues)
        denominator = np.sum(numerator)

        probArray = numerator / denominator

#        inftest = isinf(numerator)
#        if inftest.any():
#            possprobs = inftest * 1
#            probs = possprobs / np.sum(possprobs)
#
#            logger = logging.getLogger('QLearn')
#            message = "Overflow in calculating the prob with expectation "
#            message += str(expectation)
#            message += " \n Returning the prob: " + str(probs)
#            logger.warning(message)

        return probArray

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calcProbabilities(self.expectedRewards)

        return probabilities
