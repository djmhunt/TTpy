# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Opponent actor learning (OpAL): Modeling
                interactive effects of striatal dopamine on reinforcement
                learning and choice incentive.
                Collins, A. G. E., & Frank, M. J. (2014).
                Psychological Review, 121(3), 337–66.
                doi:10.1037/a0037015

"""

from __future__ import division, print_function

import logging

from numpy import exp, ones, array

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.discrete import decMaxProb
from utils import callableDetailsString


class OpAL(model):

    """The Opponent actor learning model

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
        Learning rate parameter, used as either the
    alphaGoNogoDiff : float, optional
        The difference between ``alphaGo`` and ``alphaNogo``. Default is ``None``.
        If not ``None`` will overwrite ``alphaNogo``
        :math:`\\alpha_N = \\alpha_G - \\alpha_\\delta`
    alphaCrit : float, optional
        The critic learning rate. Default is ``alpha``
    alphaGo : float, optional
        Learning rate parameter for Go, the positive part of the actor learning
        Default is ``alpha``
    alphaNogo : float, optional
        Learning rate parameter for Nogo, the negative part of the actor learning
        Default is ``alpha``
    alphaGoDiff : float, optional
        The difference between ``alphaCrit`` and ``alphaGo``. The default is ``None``
        If not ``None``  and ``alphaNogoDiff`` is also not ``None``, it will 
        overwrite the ``alphaGo`` parameter
        :math:`\\alpha_G = \\alpha_C + \\alpha_\\deltaG`
    alphaNogoDiff : float, optional
        The difference between ``alphaCrit`` and ``alphaNogo``. The default is ``None``
        If not ``None``  and ``alphaGoDiff`` is also not ``None``, it will 
        overwrite the ``alphaNogo`` parameter
        :math:`\\alpha_N = \\alpha_C + \\alpha_\\deltaN` 
    beta : float, optional
        Sensitivity parameter for probabilities. Also known as an exploration-
        exploitation parameter. Defined as :math:`\\beta` in the paper
    invBeta : float, optional
        Inverse of sensitivity parameter for the probabilities.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    betaDiff : float, optional
        The asymmetry between the actor weights. :math:`\\rho = \\beta_G - \\beta = \\beta_N + \\beta`
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
        Default ``ones((numActions, numStimuli)) / numCritics``
    expectGo : array of floats, optional
        The initialisation of the the expected go and nogo.
        Default ``ones((numActions, numStimuli)) / numCritics``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta

    Notes
    -----
    Actor: The chosen action is updated with

    .. math::

        \\delta_{d,t} = r_t-E_{d,t}

        E_{d,t+1} = E_{d,t} + \\alpha_E \\delta_{d,t}

    Critic: The chosen action is updated with

    .. math::
        G_{d,t+1} = G_{d,t} + \\alpha_G G_{d,t} \\delta_{d,t}

        N_{d,t+1} = N_{d,t} - \\alpha_N N_{d,t} \\delta_{d,t}

    Probabilities: The probabilities for all actions are calculated using

    .. math::
        A_{d,t} = (1+\\rho) G_{d,t}-(1-\\rho) N_{d,t}

        P_{d,t} = \\frac{ e^{\\beta A_{d,t} }}{\\sum_{d \\in D}e^{\\beta A_{d,t}}}
    """

    Name = "OpAL"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.betaDiff = kwargRemains.pop('betaDiff', 0)
        self.betaGo = kwargRemains.pop('betaGo', None)
        self.betaNogo = kwargRemains.pop('betaNogo', None)
        self.alpha = kwargRemains.pop('alpha', 0.1)
        self.alphaGoNogoDiff = kwargRemains.pop('alphaGoNogoDiff', None)
        self.alphaCrit = kwargRemains.pop('alphaCrit', self.alpha)
        self.alphaGo = kwargRemains.pop('alphaGo', self.alpha)
        self.alphaNogo = kwargRemains.pop('alphaNogo', self.alpha)
        self.alphaGoDiff = kwargRemains.pop('alphaGoDiff', None)
        self.alphaNogoDiff = kwargRemains.pop('alphaNogoDiff', None)
        self.expect = kwargRemains.pop('expect', ones((self.numActions, self.numStimuli)) / self.numCritics)
        self.expectGo = kwargRemains.pop('expectGo', ones((self.numActions, self.numStimuli)))

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decMaxProb(range(self.numCritics)))

        if self.alphaGoNogoDiff:
            self.alphaNogo = self.alphaGo - self.alphaGoNogoDiff
            
        if self.alphaGoDiff and self.alphaNogoDiff:
            self.alphaGo = self.alpha + self.alphaGoDiff
            self.alphaNogo = self.alpha + self.alphaNogoDiff

        if self.betaGo and self.betaNogo:
            self.beta = (self.betaGo + self.betaNogo)/2
            self.betaDiff = (self.betaGo - self.betaNogo) / (2 * self.beta)

        self.expectations = array(self.expect)
        self.go = array(self.expectGo)
        self.nogo = array(self.expectGo)
        self.actionValues = ones(self.expectations.shape)

        self.genStandardParameterDetails()
        self.parameters["alphaCrit"] = self.alphaCrit
        self.parameters["alphaGo"] = self.alphaGo
        self.parameters["alphaNogo"] = self.alphaNogo
        self.parameters["beta"] = self.beta
        self.parameters["betaGo"] = self.betaGo
        self.parameters["betaNogo"] = self.betaNogo
        self.parameters["betaDiff"] = self.betaDiff
        self.parameters["expectation"] = self.expect
        self.parameters["expectationGo"] = self.expectGo

        # Recorded information
        self.genStandardResultsStore()
        self.recGo = []
        self.recNogo = []
        self.recActionValues = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["Go"] = array(self.recGo)
        results["Nogo"] = array(self.recNogo)
        results["ActionValues"] = array(self.recActionValues)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recGo.append(self.go.copy())
        self.recNogo.append(self.nogo.copy())
        self.recActionValues.append(self.actionValues.copy())

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
        self._critic(delta, action, stimuliFilter)

        self._actor(delta, action, stimuliFilter)

        self._actionValues(self.go, self.nogo)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(self.actionValues, stimuliFilter)
            self.probabilities = self._prob(actExpectations)
        else:
            self.probabilities = self._prob(self.actionValues)

    def _critic(self, delta, action, stimuliFilter):

        self.expectations[action] += self.alphaCrit * delta * stimuliFilter

    def _actor(self, delta, action, stimuliFilter):

        chosenGo = self.go[action] * stimuliFilter
        chosenNogo = self.nogo[action] * stimuliFilter

        self.go[action] += self.alphaGo * chosenGo * delta
        self.nogo[action] -= self.alphaNogo * chosenNogo * delta

    def _actionValues(self, go, nogo):

        bd = self.betaDiff

        actionValues = (1 + bd) * go - (1 - bd) * nogo

        self.actionValues = actionValues

    def _prob(self, actionValues):

        numerator = exp(self.beta*actionValues)
        denominator = sum(numerator)

        p = numerator / denominator

        return p


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