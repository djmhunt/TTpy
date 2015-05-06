# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Notes: In the version this model used the Luce choice algorithm,
        rather than the logistic algorithm used here.
"""
from __future__ import division

import logging

from numpy import exp, array, ones

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decBeta

class BP(model):

    """The Beysian predictor model
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
    Parameters
    ----------
    gamma : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    beta : float, optional
        Decision threshold parameter. Default ``0.3``
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one. 
        Default ``array([0.5,0.5])``
    numActions : integer, optional
        The number of different reaction learning sets. Default ``2``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decBeta
    """

    Name = "BP"

    def __init__(self,**kwargs):

        self.numActions = kwargs.pop('numActions', 2)
        self.gamma = kwargs.pop('gamma', 4)
        self.prior = kwargs.pop('prior', ones(self.numActions)*0.5)
        self.beta = kwargs.pop('beta', 0.3)
        
        
        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decBeta(responses = tuple(range(1,self.numActions+1)), beta = self.beta))

        self.parameters = {"Name": self.Name,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "prior": self.prior,
                           "numActions": self.numActions,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        self.currAction = 1
#        if len(prior) != self.numActions:
#            raise warning.
        self.posteriorProb = array(self.prior)
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.decision = None
        self.lastObs = False

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recPosteriorProb = []
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
        results["PosteriorProb"] = array(self.recPosteriorProb)
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
        
        postProb = self._postProb(event, self.posteriorProb)
        self.posteriorProb = postProb

        #Calculate the new probabilities
        self.probabilities = self._prob(postProb)
        
        self.decision, self.decProbs = self.decisionFunc(self.probabilities)

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbs[self.currAction])
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)
        
    def _postProb(self, event, postProb):
        
        li = postProb * event
        newProb = li/sum(li)
        
        return newProb

    def _prob(self, expectation):

        numerat = exp(self.gamma*expectation)
        denom = sum(numerat)

        p = numerat / denom
        
        return p
#
#        diff = 2*self.posteriorProb - sum(self.posteriorProb)
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
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting
        
    Attributes
    ----------
    Name : string
        The identifier of the function
        
    """
    
    def blankStimFunc(event):
        return [1,0]
        
    blankStimFunc.Name = "blankStim"
    return blankStimFunc
