# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on ideas we had.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np
from numpy import ndarray

from model.modelTemplate import Model
from model.decision.discrete import decWeightProb


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
    expect: array of floats, optional
        The initialisation of the the expected reward.
        Default ``ones((numActions, numCues)) * 5 / numCues``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.discrete.decWeightProb
    """


    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        # A record of the kwarg keys, the variable they create and their default value

        self.alpha = kwargRemains.pop('alpha', 0.3)
        self.alphaE = kwargRemains.pop('alphaE', self.alpha)
        self.alphaA = kwargRemains.pop('alphaA', self.alpha)
        self.epsilon = kwargRemains.pop('epsilon', 0.1)
        self.expectations = kwargRemains.pop('expect', np.ones((self.numActions, self.numCues)) / self.numCues)
        self.actorExpectations = kwargRemains.pop('actorExpect', np.ones((self.numActions, self.numCues)) / self.numCues)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decWeightProb(range(self.numActions)))
        self.genEventModifiers(kwargRemains)

        self.genStandardParameterDetails()
        self.parameters["alphaE"] = self.alphaE
        self.parameters["alphaA"] = self.alphaA
        self.parameters["epsilon"] = self.epsilon
        self.parameters["expectation"] = self.expectations.copy()
        self.parameters["actorExpectation"] = self.actorExpectations.copy()

        # Recorded information
        self.genStandardResultsStore()
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

        modReward = self.rewFunc(reward, action, stimuli)

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
        if self.numCues > 1:
            actionExpectations = self.actStimMerge(expectations, stimuli)
        else:
            actionExpectations = expectations

        return actionExpectations

    def calcProbabilities(self, actionValues):
        # type: (ndarray) -> ndarray
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
        deltaEpsilon = self.epsilon * (1 / self.numActions)
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


def blankStim():
    """
    Default stimulus processor. Does nothing.

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return event

    blankStimFunc.Name = "blankStim"
    return blankStimFunc


def blankRew():
    """
    Default reward processor. Does nothing. Returns reward

    Returns
    -------
    blankRewFunc : function
        The function expects to be passed the reward and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankRewFunc(reward):
        return reward

    blankRewFunc.Name = "blankRew"
    return blankRewFunc