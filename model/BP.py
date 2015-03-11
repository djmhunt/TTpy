# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import  beta

class BP(model):

    """The Beysian predictor model
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
    Parameters
    ----------
    gamma : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    oneProb : array, optional
        The prior probability
    prior : array, optional
        The prior probability 
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.beta
    """

    Name = "BP"

    def __init__(self,**kwargs):

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.gamma = kwargs.pop('gamma',4)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.beta = kwargs.pop('beta',0.3)
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',beta(responses = (1,2), beta = self.beta))

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "prior": self.prior,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        self.currAction = 1
        self.posteriorProb = zeros(2) + self.prior
        self.probabilities = zeros(2) + self.prior
        self.decision = None
        self.lastObs = False

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recPosteriorProb = []
        self.recDecision = []

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
        results["PosteriorProb"] = array(self.recPosteriorProb)
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
        
        event = self.stimFunc(events)
        
        self.recEvents.append(event)

        #Calculate the new probabilities
        self._prob(event)

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)

    def _prob(self, event):


        li = self.posteriorProb * event
        self.posteriorProb = li/sum(li)

#        self.probabilities = 1.0/(1.0 +exp(-self.gamma*(self.posteriorProb-0.5)))

        diff = 2*self.posteriorProb - sum(self.posteriorProb)
        p = 1.0 / (1.0 + exp(-self.gamma*diff))

        self.probabilities = p
        
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
