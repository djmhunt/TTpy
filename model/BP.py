# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class BP(model):

    """The Bayesian predictor model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    eta : float, optional
        Decision threshold parameter. Default ``0.3``
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numStimuli : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numStimuli
    probActions : bool, optional
        Defines if the probabilities calculated by the model are for each
        action-stimulus pair or for actions. That is, if the stimuli values for
        each action are combined before the probability calculation.
        Default ``True``
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((numActions, numStimuli)) / numCritics)``
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

    Name = "BP"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        self.beta = kwargRemains.pop('beta', 4)
        self.eta = kwargRemains.pop('eta', 0.3)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decEta(expResponses=tuple(range(1, self.numCritics + 1)), eta=self.eta))

        self.genStandardParameterDetails()
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta

        # Recorded information
        self.genStandardResultsStore()
        self.recPosteriorProb = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["PosteriorProb"] = array(self.recPosteriorProb)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recPosteriorProb.append(self.posteriorProb.copy())

    def rewardExpectation(self, observation, action, response):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ---------
        observation : {int | float | tuple}
            The set of stimuli
        action : int or NoneType
            The chosen action
        response : float or NoneType

        Returns
        -------
        expectedReward : float
            The expected reward
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimFunc(observation, action)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numStimuli > 1:
            actionExpectations = self.actStimMerge(self.posteriorProb, stimuli)
        else:
            actionExpectations = self.posteriorProb

        expectedReward = actionExpectations[action]

        return expectedReward, stimuli, activeStimuli

    def delta(self, reward, expectation, action):
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

        Returns
        -------
        delta
        """

        modReward = self.rewFunc(reward, action)

        delta = modReward * expectation

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        postProb = self._postProb(delta)
        self.posteriorProb = postProb

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actPostProb = self.actStimMerge(postProb, stimuliFilter)
            self.probabilities = self._prob(actPostProb)
        else:
            self.probabilities = self._prob(postProb)


    def _postProb(self, delta):

        newProb = delta/sum(delta)

        return newProb

    def _prob(self, expectation):

        numerator = exp(self.beta * expectation)
        denominator = sum(numerator)

        p = numerator / denominator

        return p
#
#        diff = 2*self.posteriorProb - sum(self.posteriorProb)
#        p = 1.0 / (1.0 + exp(-self.beta*diff))
#
#        self.probabilities = p


def blankStim():
    """
    Default stimulus processor. Does nothing.Returns ([1,0], None)

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return ([1,0], None)

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