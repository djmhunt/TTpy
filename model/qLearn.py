# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Regulatory fit effects in a choice task
                Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007).
                Psychonomic Bulletin & Review, 14(6), 1125–32.
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""

from __future__ import division, print_function

import logging

from numpy import exp, ones, array

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class qLearn(model):

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
    beta : float, optional
        Sensitivity parameter for probabilities
    eta : float, optional
        Decision threshold parameter
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
        Default ``ones(numCritics) / numCritics)``
    expect: array of floats, optional
        The initialisation of the the expected reward.
        Default ``ones((numActions, numStimuli)) * 5 / numStimuli``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "qLearn"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        self.beta = kwargs.pop('beta', 4)
        self.alpha = kwargs.pop('alpha', 0.3)
        self.eta = kwargs.pop('eta', 0.3)
        self.expectation = kwargs.pop('expect', ones((self.numActions, self.numStimuli)) * 5 / self.numStimuli)

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decEta(eta=self.eta))

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
        self.parameters["expectation"] = self.expectation

        # Recorded information
        self.genStandardResultsStore()

        self.recExpectation = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()

        results["Expectation"] = array(self.recExpectation)

        return results

    def _rewardExpectation(self, observation, action, response):
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
        """

        activeStimuli, stimuli = self.stimFunc(observation, action)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.;
        if self.numStimuli > 1:
            actionExpectations = self.actStimMerge(self.expectation, stimuli)
        else:
            actionExpectations = self.expectation

        expectedReward = actionExpectations[action]

        return expectedReward, stimuli, activeStimuli

    def _delta(self, reward, expectation):
        """
        Calculates the comparison between the reward and the expectation

        Parameters
        ----------
        reward : float
            The reward value
        expectation : float

        Returns
        -------
        delta
        """

        delta = reward-expectation

        return delta

    def _updateModel(self, delta, stimuli, action):

        expectation = self.expectation

        # Find the new activities
        self._newAct(delta, stimuli, action)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(expectation, stimuli)
            self.probabilities = self._prob(actExpectations)
        else:
            self.probabilities = self._prob(expectation)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

        self.recExpectation.append(self.expectation.copy())

    def _newAct(self, delta, stimuliFilter, action):

        self.expectation[action, stimuliFilter] += self.alpha*delta

    def _prob(self, expectation):

        """

        Parameters
        ----------
        expectation : tuple of floats
            The expectation values

        Returns
        -------
        probs : list of floats
            The calculated probabilities

        """
        numerator = exp(self.beta*expectation)
        denominator = sum(numerator)

        probs = numerator / denominator

        return probs


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
