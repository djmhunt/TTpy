# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    Moore, S. C., & Sellen, J. L. (2006). 
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9. 
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
from __future__ import division

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from decision.binary import decBeta

class MS_rev(model):

    """An adapted version of the Morre & Sellen model
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
    oneProb : float in ``[0,1]``, optional
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar.
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one. 
        Default ``array([0.5,0.5])``
    activity : array, optional
        The `activity` of the neurons. The values are between ``[0,1]``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decBeta
    """

    Name = "MS_rev"

    def __init__(self,**kwargs):

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.gamma = kwargs.pop('gamma',4)
        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.activity = kwargs.pop('activity',array([0.5,0.5]))
        # The alpha is an activation rate paramenter. The M&S paper uses a value of 1.
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decBeta(responses = (1,2), beta = self.beta))
        
        self.currAction = 1
        self.probabilities = zeros(2) + self.prior
        self.probDifference = 0
        self.activity = zeros(2) + self.activity
        self.decision = None
        self.lastObs = False

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "gamma": self.gamma,
                           "beta": self.beta,
                           "alpha": self.alpha,
                           "prior": self.prior,
                           "activity" : self.activity,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recProbDifference = []
        self.recActivity = []
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
        results["ProbDifference"] = array(self.recProbDifference)
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
        self._newActivity(event)

        #Calculate the new probabilities
        self._prob()

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be accessed later 
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.currAction])
        self.recProbDifference.append(self.probDifference)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)

    def _newActivity(self, event):
        self.activity = self.activity + (event - self.activity)  * self.alpha

    def _prob(self):
        # The probability of a given jar, using the Luce choice model

#        li = self.activity ** self.gamma
#        p = li/sum(li)

        numerat = exp(self.gamma*self.activity)
        denom = sum(numerat)

        p = numerat / denom

        self.probabilities = p
        self.probDifference = p[0] - p[1]
        
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