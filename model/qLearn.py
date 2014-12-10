# -*- coding: utf-8 -*-
"""
@author: Dominic

Based on the paper Regulatory fit effects in a choice task
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

    """The documentation for the class"""

    Name = "qLearn"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

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
        """ Returns the action of the model"""

        self._decision()

        self.currAction = self.decision

        self._storeState()

        return self.currAction
        
    

    def outputEvolution(self):
        """ Returns all the relavent data for this model """

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

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

#            self.recEvents.append(event)
#
#            #Calculate jar information
#            info = self.oneProb*event + (1-self.oneProb)*(1-event)
#            self.information = array([info,1-info])
#
#            #Calculate the new probabilities
#            self._prob()

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

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        """Abstract class for the creation of plots relevant to a set of models"""
