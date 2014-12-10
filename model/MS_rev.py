# -*- coding: utf-8 -*-
"""
@author: Dominic

Based on the paper Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
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

class MS_rev(model):

    """The documentation for the class"""

    Name = "MS_rev"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.currAction = 1
        self.information = zeros(2)
        self.probabilities = zeros(2)
        self.probDifference = 0
        self.activity = zeros(2)
        self.decision = None
        self.firstDecision = 0
        self.lastObs = False

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.theta = kwargs.pop('theta',4)
        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        # The alpha is an activation rate paramenter. The M&S paper uses a value of 1.

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "theta": self.theta,
                           "beta": self.beta,
                           "alpha": self.alpha}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recInformation = []
        self.recProbabilities = []
        self.recProbDifference = []
        self.recActivity = []
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
                   "oneProb": self.oneProb,
                   "theta": self.theta,
                   "alpha": self.alpha,
                   "Information": array(self.recInformation),
                   "Probabilities": array(self.recProbabilities),
                   "ProbDifference": array(self.recProbDifference),
                   "Activity": array(self.recActivity),
                   "Actions":array(self.recAction),
                   "Decsions": array(self.recDecision),
                   "Events":array(self.recEvents)}

        return results

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            self.recEvents.append(event)

            #Calculate jar information
            info = self.oneProb*event + (1-self.oneProb)*(1-event)
            self.information = array([info,1-info])

            #Find the new activites
            self._newActivity()

            #Calculate the new probabilities
            self._prob()

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(event)

                #Calculate jar information
                info = self.oneProb*event + (1-self.oneProb)*(1-event)
                self.information = array([info,1-info])

                #Find the new activites
                self._newActivity()

                #Calculate the new probabilities
                self._prob()

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)
        self.recInformation.append(self.information.copy())
        self.recProbabilities.append(self.probabilities.copy())
        self.recProbDifference.append(self.probDifference)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)

    def _prob(self):
        # The probability of a given jar, using the Luce choice model

#        li = self.activity ** self.theta
#        p = li/sum(li)

        diff = 2*self.activity - sum(self.activity)
        p = 1.0 / (1.0 + exp(-self.theta*diff))

        self.probabilities = p
        self.probDifference = p[0] - p[1]

    def _newActivity(self):
        self.activity = self.activity + (self.information - self.activity)  * self.alpha

    def _decision(self):

        prob = self.probabilities[0]

        if abs(prob-0.5)>self.beta:
            if prob>0.5:
                self.decision = 1
            else:
                self.decision = 2
        else:
            self.decision = None

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        """Abstract class for the creation of plots relevant to a set of models"""