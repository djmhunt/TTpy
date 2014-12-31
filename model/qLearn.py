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

class qLearn(model):

    """The q-Learning algorithm
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    theta : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    prior : array, optional
        The prior probability 
    expect: float, optional
        The initialisation of the the expected reward
    """

    Name = "qLearn"

    def __init__(self,**kwargs):

        self.theta = kwargs.pop('theta',4)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.expect = kwargs.pop('expect',5)

        self.parameters = {"Name": self.Name,
                           "theta": self.theta,
                           "beta": self.beta,
                           "alpha": self.alpha}#,
#                           "expectation": self.expect}

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

        self._decision()

        self.currAction = self.decision

        self._storeState()

        return self.currAction
        
    

    def outputEvolution(self):
        """ Returns all the relevent data for this model 
        
        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name, 
            Probabilities, Actions and Events.
        """

        results = {"Name": self.Name,
                   "theta": self.theta,
                   "beta": self.beta,
                   "alpha": self.alpha,
                   "prior": self.prior,
                   "Probabilities": array(self.recProbabilities),
                   "ActionProb": array(self.recActionProb),
                   "Expectation": array(self.recExpectation),
                   "Actions":array(self.recAction),
                   "Decsions": array(self.recDecision),
                   "Events":array(self.recEvents)}

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""
        
        event = events

        if instance == 'obs':

            self.recEvents.append(event)

            chosen = self.currAction

            #Calculate jar information
            self.expectation[chosen] += self.alpha*(event - self.expectation[chosen])

            #Calculate the new probabilities
            self._prob()

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(event)

                chosen = self.currAction

                #Calculate jar information
                self.expectation[chosen] += self.alpha*(event - self.expectation[chosen])

                #Calculate the new probabilities
                self._prob()

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.currAction])
        self.recExpectation.append(self.expectation.copy())
        self.recDecision.append(self.decision)

    def _prob(self):

        numerat = exp(self.theta*self.expectation)
        denom = sum(numerat)

        self.probabilities= numerat / denom

    def _decision(self):

        prob = self.probabilities[0]

        if abs(prob-0.5) >= self.beta:
            if prob>0.5:
                self.decision = 0
            elif prob == 0.5:
                self.decision = choice([0,1])
            else:
                self.decision = 1
        else:
            self.decision = None
