# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Notes: In the version this model used the Luce choice algorithm,
        rather than the logistic algorithm used here.
"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones
from random import choice
from types import NoneType

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
    eta : float, optional
        Decision threshold parameter
    activity : array, optional
        The `activity` of the neurons. The values are between ``[0,1]``
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    numStimuli : integer, optional
        The number of different reaction learning sets. Default ``2``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "EP"

    def __init__(self,**kwargs):

        self.alpha = kwargs.pop('alpha',0.3)
        self.eta = kwargs.pop('eta',0.3)
        self.beta = kwargs.pop('beta',4)
        self.numStimuli = kwargs.pop('numStimuli',2)
        self.activity = kwargs.pop('activity',ones(self.numStimuli)*0.5)
        self.prior = kwargs.pop('prior',ones(self.numStimuli)*0.5)

        self.decision = None
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.validActions = None
        self.currAction = None

        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decEta(expResponses = (1,2), eta = self.eta))

        self.parameters = {"Name": self.Name,
                           "alpha": self.alpha,
                           "beta": self.beta,
                           "eta": self.eta,
                           "prior": self.prior,
                           "activity" : self.activity,
                           "numStimuli": self.numStimuli,
                           "stimFunc" : callableDetailsString(self.stimFunc),
                           "decFunc" : callableDetailsString(self.decisionFunc)}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recActivity = []
        self.recDecision = []
        self.recProbabilities = []
        self.recActionProb = []

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

        results = self.parameters.copy()

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["Activity"] = array(self.recActivity)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            if type(events) is not NoneType:
                self._processEvent(events)
            self._processAction()


        elif instance == 'reac':
            if type(events) is not NoneType:
                self._processEvent(events)

    def _processEvent(self,events):

        event = self.stimFunc(events, self.currAction)

        self.recEvents.append(event)

        #Find the new activites
        self._newAct(event)

        #Calculate the new probabilities
        self.probabilities = self._prob(self.activity)

    def _processAction(self):

        self.decision, self.decProbs = self.decisionFunc(self.probabilities, self.currAction, validResponses = self.validActions)

    def storeState(self):
        """"
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbs[self.currAction])

    def _newAct(self,event):

        oldAct = self.activity

        self.activity = oldAct + self.alpha * (event-oldAct)

    def _prob(self, expectation):
        """ Calculate the new probabilities of different actions """

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
        return [1,0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc


