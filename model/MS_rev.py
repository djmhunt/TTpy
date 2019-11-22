# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    Moore, S. C., & Sellen, J. L. (2006).
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441

:Notes: In the original paper this model used a modified Luce choice
        algorithm, rather than the logistic algorithm used here.
"""
from __future__ import division, print_function

import logging

import numpy as np

from modelTemplate import Model


class MSRev(Model):

    """An adapted version of the More & Sellen model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    beta : float, optional
        Sensitivity parameter for probabilities
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    eta : float, optional
        Decision threshold parameter
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numCues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numCues
    actionCodes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((numActions, numCues)) / numCritics)``
    activity : array, optional
        The initialisation of the `activity` of the neurons. The values are between ``[0,1]``
        Default ``ones((numActions, numCues)) / numCritics``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    def __init__(self, alpha=0.3, beta=4, invBeta=None, expect=None, **kwargs):

        super(MSRev, self).__init__(**kwargs)

        self.alpha = alpha
        if invBeta is not None:
            beta = (1 / invBeta) - 1
        self.beta = beta

        if expect is None:
            expect = np.ones((self.numActions, self.numCues)) / self.numCritics
        self.expectations = expect

        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["expectations"] = self.expectations

        self.probDifference = 0

        # Recorded information

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be accessed later

        The stored variables are ones that describe the model.
        """

        self.storeStandardResults()

    def rewardExpectation(self, observation):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

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

        activeStimuli, stimuli = self.stimFunc(observation)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numCues > 1:
            actionExpectations = self.actStimMerge(self.expectations, stimuli)
        else:
            actionExpectations = self.expectations

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

        modReward = self.rewFunc(reward, action, stimuli)

        delta = modReward - expectation

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        # Find the new activities
        self._newActivity(delta)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(self.expectations, stimuliFilter)
            self.probabilities = self.calcProbabilities(actExpectations)
        else:
            self.probabilities = self.calcProbabilities(self.expectations)

    def _newActivity(self, delta):

        self.expectations += delta * self.alpha

    def calcProbabilities(self, actionValues):
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

        # The probability of a given jar, using the Luce choice model

        #li = actionValues ** self.beta
        #probArray = li/sum(li)

        numerator = np.exp(self.beta * actionValues)
        denominator = sum(numerator)

        probArray = numerator / denominator

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
