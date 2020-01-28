# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the model QLearn as well as the paper:
                Meta-learning in Reinforcement Learning


"""

from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np

from model.modelTemplate import Model


class QLearnMeta(Model):

    """The q-Learning algorithm with a second-order adaptive beta

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
    tau : float, optional
        Beta rate Sensitivity parameter for probabilities
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
    expect: array of floats, optional
        The initialisation of the the expected reward.
        Default ``ones((number_actions, number_cues)) * 5 / number_cues``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.eta
    """

    def __init__(self, alpha=0.3, tau=0.2, rewardD=None, rewardDD=None, expect=None, **kwargs):

        super(QLearnMeta, self).__init__(**kwargs)

        # A record of the kwarg keys, the variable they create and their default value

        self.tau = tau
        self.alpha = alpha

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect

        if rewardD is None:
            rewardD = 5.5 * np.ones((self.number_actions, self.number_cues))
        self.rewardD = rewardD
        if rewardDD is None:
            rewardDD = 5.5 * np.ones((self.number_actions, self.number_cues))
        self.rewardDD = rewardDD

        self.parameters["alpha"] = self.alpha
        self.parameters["tau"] = self.tau
        self.parameters["expectation"] = self.expectations.copy()

        self.beta = np.exp(self.rewardD - self.rewardDD)

        # Recorded information
        self.recRewardD = []
        self.recRewardDD = []
        self.recBeta = []

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["rewardD"] = np.array(self.recRewardD).T
        results["rewardDD"] = np.array(self.recRewardDD).T
        results["beta"] = np.array(self.recBeta).T

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

        self.recRewardD.append(self.rewardD.flatten())
        self.recRewardDD.append(self.rewardDD.flatten())
        self.recBeta.append(self.beta.flatten())

    def rewardExpectation(self, observation):
        """Calculate the estimated reward based on the action and stimuli

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

        self.updateBeta(reward, action)

        return delta

    def updateBeta(self, reward, action):
        """

        Parameters
        ----------
        reward : float
            The reward value

        """

        #self.rewardD += self.tau * (reward - self.rewardD)
        #self.rewardDD += self.tau * (self.rewardD - self.rewardDD)
        #self.beta = np.exp(self.rewardD - self.rewardDD)

        rewardD = self.rewardD[action]
        rewardDD = self.rewardDD[action]
        rewardD += self.tau * (reward - rewardD)
        rewardDD += self.tau * (rewardD - rewardDD)
        self.beta[action] = np.exp(rewardD - rewardDD)
        self.rewardD[action] = rewardD
        self.rewardDD[action] = rewardDD

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

        newExpectations = self.expectations[action] + self.alpha*delta*stimuli/np.sum(stimuli)

        newExpectations = newExpectations * (newExpectations >= 0)

        self.expectations[action] = newExpectations

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
