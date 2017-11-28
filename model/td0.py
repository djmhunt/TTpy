# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the description on p134-135 of Reinforcement Learning, Sutton & Barto 1998
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from numpy import exp, ones, array, isnan, isinf, sum, sign, zeros

from model.modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class td0(model):

    """The td-Learning algorithm

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
    gamma: float, optional
        future expectation discounting
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numCues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numCues
    probActions : bool, optional
        Defines if the probabilities calculated by the model are for each
        action-stimulus pair or for actions. That is, if the stimuli values for
        each action are combined before the probability calculation.
        Default ``True``
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
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "td0"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        # A record of the kwarg keys, the variable they create and their default value

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.alpha = kwargRemains.pop('alpha', 0.3)
        self.gamma = kwargRemains.pop('gamma', 0.3)
        self.eta = kwargRemains.pop('eta', 0.3)
        self.expectations = kwargRemains.pop('expect', ones((self.numActions, self.numCues)) / self.numCues)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decEta(eta=self.eta))

        self.lastAction = 0
        self.lastStimuliFilter = zeros(self.numCues)

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["gamma"] = self.gamma
        self.parameters["expectation"] = self.expectations.copy()

        # Recorded information
        self.genStandardResultsStore()

    def outputEvolution(self):
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
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

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

        # TD0 updates the model in two parts: once the reward is received and a future discount once the next action
        # has been chosen

        # Find the new activities
        change = self.alpha*delta*stimuliFilter
        self._newExpect(action, change)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(self.expectations, stimuliFilter)
            self.probabilities = self.calcProbabilities(actExpectations)
        else:
            self.probabilities = self.calcProbabilities(self.expectations)

        self.lastStimuliFilter = stimuliFilter
        self.lastAction = action

    def _newExpect(self, action, change):

        newExpectations = self.expectations[action] + change

        newExpectations = newExpectations * (newExpectations >= 0)

        self.expectations[action] = newExpectations

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
        numerator = exp(self.beta * actionValues)
        denominator = sum(numerator)

        probArray = numerator / denominator

#        inftest = isinf(numerator)
#        if inftest.any():
#            possprobs = inftest * 1
#            probs = possprobs / sum(possprobs)
#
#            logger = logging.getLogger('qLearn')
#            message = "Overflow in calculating the prob with expectation "
#            message += str(expectation)
#            message += " \n Returning the prob: " + str(probs)
#            logger.warning(message)

        return probArray

    def choiceReflection(self):
        """
        Allows the model to update its state once an action has been chosen.
        """

        change = self.alpha * self.gamma * self.expectedRewards[self.currAction] * self.lastStimuliFilter
        self._newExpect(self.lastAction, change)

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