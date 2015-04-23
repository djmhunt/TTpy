# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import logging

from numpy import exp, zeros, array
from random import choice

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decBeta

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
    gamma : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    activity : array, optional
        The `activity` of the neurons. The values are between ``[0,1]``
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one. 
        Default ``array([0.5,0.5])``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decBeta
    """

    Name = "EP"

    def __init__(self,**kwargs):

        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.gamma = kwargs.pop('gamma',4)
        self.activity = kwargs.pop('activity',array([0.5,0.5]))
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        
        self.decision = None
        self.probabilities = zeros(2) + self.prior
        self.lastObs = False
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decBeta(responses = (1,2), beta = self.beta))

        self.parameters = {"Name": self.Name,
                           "alpha": self.alpha,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "prior": self.prior,
                           "activity" : self.activity,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

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

        self.decision = self.decisionFunc(self.probabilities)

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
        results["Activity"] = array(self.recActivity)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            
            self._processEvent(events)

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:
                self._processEvent(events)
                
    def _processEvent(self,events):
        
        event = self.stimFunc(events, self.currAction)
        
        self.recEvents.append(event)

        #Find the new activites
        self._newAct(event)

        #Calculate the new probabilities
        self.probabilities = self._prob(self.activity)

    def storeState(self):
        """" 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.currAction])

    def _newAct(self,event):

        self.activity = self.activity + (event-self.activity)* self.alpha

    def _prob(self, expectation):
        """ Calculate the new probabilities of different actions """
        
        numerat = exp(self.gamma*expectation)
        denom = sum(numerat)

        p = numerat / denom
        
        return p

#        diff = 2*self.activity - sum(self.activity)
#        p = 1.0 / (1.0 + exp(-self.gamma*diff))
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


