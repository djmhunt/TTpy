# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Modified version of that found in the paper The role of the
                ventromedial prefrontal cortex in abstract state-based inference
                during decision making in humans.
                Hampton, A. N., Bossaerts, P., & O’Doherty, J. P. (2006).
                The Journal of Neuroscience : The Official Journal of the
                Society for Neuroscience, 26(32), 8360–7.
                doi:10.1523/JNEUROSCI.1010-06.2006

:Notes: In the original paper this model used the Luce choice algorithm,
        rather than the logistic algorithm used here. This generalisation has
        meant that the variable nu is no longer possible to use.
"""

from __future__ import division, print_function

import logging

from numpy import exp, ones, array

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class qLearn2(model):

    """The q-Learning algorithm modified to have different positive and
    negative reward prediction errors

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
        Learning rate parameter. For this model only used when setting alphaPos
        and alphaNeg to the same value. Default 0.3
    alphaPos : float, optional
        The positive learning rate parameter. Used when RPE is positive.
        Default is alpha
    alphaNeg : float, optional
        The negative learning rate parameter. Used when RPE is negative.
        Default is alpha
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
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((numActions, numStimuli)) / numCritics)``
    expect: array of floats, optional
        The initialisation of the the expected reward.
        Default ``ones((numActions, numStimuli)) * 5 / numStimuli``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta

    See Also
    --------
    model.qLearn : This model is heavily based on that one
    """

    Name = "qLearn2"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.alpha = kwargRemains.pop('alpha', 0.3)
        self.alphaPos = kwargRemains.pop('alphaPos', self.alpha)
        self.alphaNeg = kwargRemains.pop('alphaNeg', self.alpha)
        self.eta = kwargRemains.pop('eta', 0.3)
        self.expectations = kwargRemains.pop('expect', ones((self.numActions, self.numStimuli)) / self.numStimuli)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decEta(eta=self.eta))

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["alphaPos"] = self.alphaPos
        self.parameters["alphaNeg"] = self.alphaNeg
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
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

    def rewardExpectation(self, observation, action, response):
        """Calculate the estimated reward based on the action and stimuli

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
            actionExpectations = self.actStimMerge(self.expectations, stimuli)
        else:
            actionExpectations = self.expectations

        expectedReward = actionExpectations[action]

        return expectedReward, stimuli, activeStimuli

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
        self._newAct(delta, action, stimuliFilter)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(self.expectations, stimuliFilter)
            self.probabilities = self._prob(actExpectations)
        else:
            self.probabilities = self._prob(self.expectations)

    def _newAct(self, delta, action, stimuliFilter):

        if delta > 0:
            self.expectations[action] += self.alphaPos*delta*stimuliFilter
        else:
            self.expectations[action] += self.alphaNeg*delta*stimuliFilter

    def _prob(self, expectation):
        """
        Calculate the probabilities

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