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

from __future__ import division

import logging

from numpy import exp, ones, array
from random import choice

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
    alphaGoDiff : float, optional
        The difference between ``alphaGo`` and ``alphaNogo``. Default is ``None``.
        If not ``None`` will overwrite any parameters in ``alphaGo`` and ``alphaNogo``
        :math:`\\alpha_G = \\alpha + \\alpha_\\delta` and :math:`\\alpha_N = \\alpha - \\alpha_\\delta`
    alphaC : float, optional
        The critic learning rate. Default is ``alpha``
    alphaGo : float, optional
        Learning rate parameter for Go, the positive part of the actor learning
        Default is ``alpha``
    alphaNogo : float, optional
        Learning rate aprameter for Nogo, the negative part of the actor learning
        Default is ``alpha``
    beta : float, optional
        Sensitivity parameter for probabilities. Also known as an exploration-
        expoitation parameter. Defined as :math:`\\beta` in the paper
    betaDiff : float, optional
        The asymetry beween the actor weights. :math:`\\rho = \\beta_G - \\beta = \\beta_N + \\beta`
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    expect: float, optional
        The initialisation of the the expected reward. Default ``array([5,5])``
    numActions : integer, optional
        The number of different reaction learning sets. Default ``2``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
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

    def __init__(self,**kwargs):

        self.numActions = kwargs.pop('numActions', 4)
        self.beta = kwargs.pop('beta', 1)
        self.betaDiff = kwargs.pop('betaDiff',0)
        self.betaGo = kwargs.pop('betaGo', None)
        self.betaNogo = kwargs.pop('betaNogo', None)
        self.alpha = kwargs.pop('alpha', 0.3)
        self.alphaGoDiff = kwargs.pop('alphaGoDiff', None)
        self.alphaCrit = kwargs.pop('alphaC', self.alpha)
        self.alphaGo = kwargs.pop('alphaGo', self.alpha)
        self.alphaNogo = kwargs.pop('alphaNogo', self.alpha)
        self.prior = kwargs.pop('prior', ones(self.numActions)/self.numActions)
        self.expect = kwargs.pop('expect', ones(self.numActions)/self.numActions)

        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decMaxProb(range(self.numActions)))

        if self.alphaGoDiff:
            self.alphaGo = self.alpha + self.alphaGoDiff
            self.alphaNogo = self.alpha - self.alphaGoDiff

        if self.betaGo and self.alphaNogo:
            self.beta = (self.betaGo + self.betaNogo)/2
            self.betaDiff = (self.betaGo - self.betaNogo)/ (2 * self.beta)

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "betaDiff": self.betaDiff,
                           "betaGo": self.betaGo,
                           "betaNogo": self.betaNogo,
                           "alphaCrit": self.alphaCrit,
                           "alphaGo": self.alphaGo,
                           "alphaNogo": self.alphaNogo,
                           "expectation": self.expect,
                           "prior": self.prior,
                           "numActions": self.numActions,
                           "stimFunc" : callableDetailsString(self.stimFunc),
                           "decFunc" : callableDetailsString(self.decisionFunc)}

        self.currAction = None
        self.expectation = array(self.expect)
        self.go = array(self.expect)
        self.nogo = array(self.expect)
        self.actionValues = array(self.expect)
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.decision = None
        self.validActions = None

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recExpectation = []
        self.recGo = []
        self.recNogo = []
        self.recActionValues = []
        self.recDecision = []

    def action(self):
        """
        Returns
        -------
        action : integer or None
        """

        self.currAction = self.decision

        self.storeState()

        return self.currAction


    def outputEvolution(self):
        """ Returns all the relevent data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.parameters

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["Expectation"] = array(self.recExpectation)
        results["Go"] = array(self.recGo)
        results["Nogo"] = array(self.recNogo)
        results["ActionValues"] = array(self.recActionValues)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            if events != None:
                self._processEvent(events)
            self._processAction()


        elif instance == 'reac':
            if events != None:
                self._processEvent(events)

    def _processEvent(self,events):

        chosen = self.currAction

        event = self.stimFunc(events, chosen)

        self.recEvents.append(event)

        #Find the new activites
        self._critic(event, chosen)

        #Calculate the new probabilities
        self.probabilities = self._prob(self.go, self.nogo)

    def _processAction(self):

        self.decision, self.decProbs = self.decisionFunc(self.probabilities, self.currAction, validResponses = self.validActions)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbs[self.currAction])
        self.recExpectation.append(self.expectation.copy())
        self.recGo.append(self.go.copy())
        self.recNogo.append(self.nogo.copy())
        self.recActionValues.append(self.actionValues.copy())
        self.recDecision.append(self.decision)

    def _critic(self,event, chosen):

        chosenExp = self.expectation[chosen]

        change = event - chosenExp

        self.expectation[chosen] = chosenExp + self.alphaCrit*change

        self._actor(change, chosen)

    def _actor(self, change, chosen):

        chosenGo = self.go[chosen]
        chosenNogo = self.nogo[chosen]

        self.go[chosen] = chosenGo + self.alphaGo * chosenGo * change
        self.nogo[chosen] = chosenNogo - self.alphaNogo * chosenNogo * change

    def _prob(self, go, nogo):

        gd = self.betaDiff

        actionValues = (1+gd)*go - (1-gd)*nogo

        self.actionValues = actionValues

        numerat = exp(self.beta*actionValues)
        denom = sum(numerat)

        p = numerat / denom

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
