# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the Epsilon-greedy method along with a past choice autocorrelation inspired by ``qLearnCorr``
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from numpy import exp, ones, array, isnan, isinf, sum, sign, max, shape

from model.modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.discrete import decWeightProb


class qLearnECorr(model):

    """The q-Learning algorithm

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
    kappa : float, optional
        The autocorrelation parameter for which positive values promote sticking and negative values promote alternation
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

    See Also
    --------
    model.qLearnCorr : This model is heavily based on that one
    """

    Name = "qLearnECorr"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        # A record of the kwarg keys, the variable they create and their default value

        self.kappa = kwargRemains.pop('kappa', 0)
        self.alpha = kwargRemains.pop('alpha', 0.3)
        self.epsilon = kwargRemains.pop('epsilon', 0.1)
        self.expectations = kwargRemains.pop('expect', ones((self.numActions, self.numCues)) / self.numCues)

        self.lastAction = kwargRemains.pop('firstAction', 1)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decWeightProb(range(self.numActions)))
        self.genEventModifiers(kwargRemains)

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["kappa"] = self.kappa
        self.parameters["epsilon"] = self.epsilon
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
            The action chosen by the model in this timestep
        stimuli : list of float
            The weights of the different stimuli in this timestep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this timestep

        """

        # Find the new activities
        self._newExpect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        actExpectations = self._actExpectations(self.expectations, stimuli)
        self.probabilities = self.calcProbabilities(actExpectations)

        self.lastAction = action

    def _newExpect(self, action, delta, stimuli):

        newExpectations = self.expectations[action] + self.alpha*delta*stimuli/sum(stimuli)

        newExpectations = newExpectations * (newExpectations >= 0)

        self.expectations[action] = newExpectations

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

        lastAction = -ones(shape(actionValues))
        lastAction[self.lastAction] = 1

        cbest = actionValues == max(actionValues)
        deltaEpsilon = self.epsilon * (1 / self.numActions)
        bestEpsilon = (1 - self.epsilon) / sum(cbest) + deltaEpsilon
        p = bestEpsilon * cbest + deltaEpsilon * (1 - cbest)

        change = self.kappa * lastAction
        probArray = p + (1 - p) * change * (change > 0) + p * change * (change < 0)
        #probArray = p + p * (1 - p) * change

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