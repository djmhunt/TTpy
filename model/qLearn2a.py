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
"""

from __future__ import division

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decBeta

class qLearn2a(model):

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
    gamma : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    nu : float in range [0,0.5], optional
        The idecision point between the two choices. Default is 0
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
        
    See Also
    --------
    model.qLearn2 : This model is heavily based on that one
    """

    Name = "qLearn2a"

    def __init__(self,**kwargs):

        self.gamma = kwargs.pop('gamma',4)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.alpha = kwargs.pop('alpha',0.3)
        self.alphaPos = kwargs.pop('alphaPos', self.alpha)
        self.alphaNeg = kwargs.pop('alphaNeg', self.alpha)
        self.beta = kwargs.pop('beta',0.3)
        self.nu = kwargs.pop('nu', 0)
        self.expect = kwargs.pop('expect',5)
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decBeta(beta = self.beta))

        self.parameters = {"Name": self.Name,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "alpha": self.alpha,
                           "alphaPos" : self.alphaPos,
                           "alphaNeg" : self.alphaNeg,
                           "nu" : self.nu,
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
        self._expectUpdate(event, chosen)

        #Calculate the new probabilities
        self.probabilities = self._prob(self.expectation)
    
    def _expectUpdate(self, event, chosen):
        
        diff = event - self.expectation[chosen]
        
        if diff > 0:
            self.expectation[chosen] += self.alphaPos*diff
        else:
            self.expectation[chosen] += self.alphaNeg*diff
        

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

    def _prob(self, expectation):
        
        numerat = exp(self.gamma*expectation)
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
