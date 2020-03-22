# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on ideas we had.
"""

import logging

import numpy as np

from model.modelTemplate import Model


class ACE(Model):
    """A basic, complete actor-critic model with decision making based on QLearnE

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    alphaE : float, optional
        Learning rate parameter for the update of the expectations. Default ``\alpha``
    alphaA : float, optional
        Learning rate parameter for the update of the actor. Default ``\alpha``
    epsilon : float, optional
        Noise parameter. The larger it is the less likely the model is to choose the highest expected reward
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
        The initialisation of the expected reward.
        Default ``ones((number_actions, number_cues)) * 5 / number_cues``
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

    def __init__(self, alpha=0.3, epsilon=0.1, alphaE=None, alphaA=None, expect=None, actorExpect=None, **kwargs):

        super(ACE, self).__init__(**kwargs)

        # A record of the kwarg keys, the variable they create and their default value
        self.alpha = alpha
        self.epsilon = epsilon

        if alphaE is None:
            alphaE = alpha
        if alphaA is None:
            alphaA = alpha
        self.alphaE = alphaE
        self.alphaA = alphaA

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect
        if actorExpect is None:
            actorExpect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.actorExpectations = actorExpect

        self.parameters["alphaE"] = self.alphaE
        self.parameters["alphaA"] = self.alphaA
        self.parameters["epsilon"] = self.epsilon
        self.parameters["expectation"] = self.expectations.copy()
        self.parameters["actorExpectation"] = self.actorExpectations.copy()

        # Recorded extra information
        self.recActorExpectations = []

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["ActorExpectations"] = np.array(self.recActorExpectations).T

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recActorExpectations.append(self.actorExpectations.flatten())

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
        self.probabilities = self.actorStimulusProbs()

    def _newExpect(self, action, delta, stimuli):

        newExpectations = self.expectations[action] + self.alphaE * delta * stimuli/np.sum(stimuli)
        newExpectations = newExpectations * (newExpectations >= 0)
        self.expectations[action] = newExpectations

        newActorExpectations = self.actorExpectations[action] + self.alphaA * delta * stimuli/np.sum(stimuli)
        newActorExpectations = newActorExpectations * (newActorExpectations >= 0)
        self.actorExpectations[action] = newActorExpectations

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

        cbest = actionValues == np.max(actionValues)
        deltaEpsilon = self.epsilon * (1 / self.number_actions)
        bestEpsilon = (1 - self.epsilon) / np.sum(cbest) + deltaEpsilon
        probArray = bestEpsilon * cbest + deltaEpsilon * (1 - cbest)

        return probArray

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        actExpectations = self._actExpectations(self.actorExpectations, self.stimuli)
        probabilities = self.calcProbabilities(actExpectations)

        return probabilities
