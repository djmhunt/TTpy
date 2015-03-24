# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Regulatory fit effects in a choice task
                Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007).
                Psychonomic Bulletin & Review, 14(6), 1125–32. 
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""

from __future__ import division

import logging

from numpy import exp, zeros, array
from random import choice

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decBeta

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
    gamma : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one. 
        Default ``array([0.5,0.5])`` 
    expect: float, optional
        The initialisation of the the expected reward
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decBeta
    """

    Name = "qLearn"

    def __init__(self,**kwargs):

        self.gamma = kwargs.pop('gamma',4)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.expect = kwargs.pop('expect',5)
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decBeta(beta = self.beta))

        self.parameters = {"Name": self.Name,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "alpha": self.alpha,
                           "expectation": self.expect,
                           "prior": self.prior,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        self.currAction = None
        self.expectation = zeros(2) + self.expect
        self.probabilities = zeros(2) + self.prior
        self.decision = None
        self.lastObs = False

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recExpectation = []
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
        results["ActionProb"] = array(self.recActionProb)
        results["Expectation"] = array(self.recExpectation)
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
        
        chosen = self.currAction
        
        event = self.stimFunc(events, chosen)
        
        self.recEvents.append(event)

        #Calculate jar information
        self.expectation[chosen] += self.alpha*(event - self.expectation[chosen])

        #Calculate the new probabilities
        self._prob()

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.currAction])
        self.recExpectation.append(self.expectation.copy())
        self.recDecision.append(self.decision)

    def _prob(self):

        numerat = exp(self.gamma*self.expectation)
        denom = sum(numerat)

        self.probabilities= numerat / denom
        
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
