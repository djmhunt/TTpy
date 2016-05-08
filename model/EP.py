# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Notes: In the version this model used the Luce choice algorithm,
        rather than the logistic algorithm used here.
"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class EP(model):

    """
    The expectation prediction model

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
    prior : array of floats in ``[0,1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((self.numActions, self.numStimuli)) / self.numCritics)``
    activity : array, optional
        The initialisation of the `activity` of the neurons.
        The values are between ``[0,1]`
        Default ``ones((numActions, numStimuli)) / numCritics```
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

    Name = "EP"

    def __init__(self,**kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        self.alpha = kwargRemains.pop('alpha', 0.3)
        self.eta = kwargRemains.pop('eta', 0.3)
        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.activity = kwargRemains.pop('activity', ones((self.numActions, self.numStimuli)) / self.numCritics)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decEta(expResponses=(1, 2), eta=self.eta))

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
        self.parameters["activity"] = self.activity

        # Recorded information
        self.genStandardResultsStore()
        self.recActivity = []

    def outputEvolution(self):
        """ Returns the relevant data expected from a model as well as the parameters for the current model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["Activity"] = array(self.recActivity)

        return results

    def storeState(self):
        """"
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recActivity.append(self.activity.copy())

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
        expectedReward : array of floats
            The expected rewards
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimFunc(observation, action)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numStimuli > 1:
            actionActivity = self.actStimMerge(self.activity, stimuli)
        else:
            actionActivity = self.activity

        return actionActivity, stimuli, activeStimuli

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

        delta = modReward - expectation

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        # Find the new activities
        self._newAct(delta)

        # Calculate the new probabilities
        if self.probActions:
            actActivity = self.actStimMerge(self.activity, stimuliFilter)
            self.probabilities = self._prob(actActivity)
        else:
            self.probabilities = self._prob(self.activity)

    def _newAct(self, delta):

        self.activity += self.alpha * delta

    def _prob(self, expectation):
        """ Calculate the new probabilities of different actions

        Parameters
        ----------
        expectation : tuple of floats
            The expectation values

        Returns
        -------
        p : list of floats
            The calculated probabilities
        """

        numerat = exp(self.beta*expectation)
        denom = sum(numerat)

        p = numerat / denom

        return p

#        diff = 2*self.activity - sum(self.activity)
#        p = 1.0 / (1.0 + exp(-self.beta*diff))
#
#        self.probabilities = p

def blankStim():
    """
    Default stimulus processor. Does nothing.Returns [1,0]

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return [1, 0]

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

