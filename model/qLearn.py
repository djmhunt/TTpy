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

from numpy import exp, ones, array
from random import choice

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decEta
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
    """

    Name = "qLearn"

    def __init__(self,**kwargs):

        self.numActions = kwargs.pop('numActions', 2)
        self.beta = kwargs.pop('beta', 4)
        self.prior = kwargs.pop('prior', ones(self.numActions)*0.5)
        self.alpha = kwargs.pop('alpha', 0.3)
        self.eta = kwargs.pop('eta', 0.3)
        self.expect = kwargs.pop('expect', ones(self.numActions)*5)
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decEta(eta = self.eta))

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "eta": self.eta,
                           "alpha": self.alpha,
                           "expectation": self.expect,
                           "prior": self.prior,
                           "numActions": self.numActions,
                           "stimFunc" : callableDetailsString(self.stimFunc),
                           "decFunc" : callableDetailsString(self.decisionFunc)}

        self.currAction = None
        self.expectation = array(self.expect)
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.decision = None
        self.lastObs = False
        self.validActions = None

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

        #Find the new activites
        self._newAct(event, chosen)

        #Calculate the new probabilities
        self.probabilities = self._prob(self.expectation)
        
        self.decision, self.decProbs = self.decisionFunc(self.probabilities, validResponses = self.validActions)


    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbs[self.currAction])
        self.recExpectation.append(self.expectation.copy())
        self.recDecision.append(self.decision)
        
    def _newAct(self,event, chosen):
        
        chosenExp = self.expectation[chosen]
        
        self.expectation[chosen] = chosenExp + self.alpha*(event - chosenExp)

    def _prob(self, expectation):

        numerat = exp(self.beta*expectation)
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
