# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from numpy import exp, zeros, array
from random import choice

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot

class EP(model):

    """The documentation for the class"""

    Name = "EP"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.theta = kwargs.pop('theta',4)
        self.activity = kwargs.pop('activity',array([0.5,0.5]))
        self.decision = None
        self.firstDecision = 0
        self.probabilities = zeros(2) + 0.5
        self.lastObs = False

        self.parameters = {"Name": self.Name,
                           "alpha": self.alpha,
                           "theta": self.theta,
                           "beta": self.beta}

        # Recorded information

        self.recInformation = []
        self.recAction = []
        self.recEvents = []
        self.recActivity = []
        self.recDecision = []
        self.recProbabilities = []

    def action(self):
        """ Returns the action of the model"""

        self._decision()

        self.currAction = self.decision

        self._storeState()

        return self.currAction

    def outputEvolution(self):
        """ Returns all the relavent data for this model """

        results = { "Name": self.Name,
                    "Actions":array(self.recAction),
                    "Events":array(self.recEvents),
                    "Information": array(self.recInformation),
                    "Activity": array(self.recActivity),
                    "Decsions": array(self.recDecision),
                    "Probabilities": array(self.recProbabilities),
                    "alpha": self.alpha,
                    "theta": self.theta,
                    "beta": self.beta}

        return results

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            self.recEvents.append(event)

            #Calculate jar information
            self.information = array([event,1-event])

            #Find the new activites
            self._newAct()

            #Calculate the new probabilities
            self._prob()

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(event)

                #Calculate jar information
                self.information = array([event,1-event])

                #Find the new activites
                self._newAct()

                #Calculate the new probabilities
                self._prob()

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recInformation.append(self.information.copy())
        self.recAction.append(self.currAction)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)
        self.recProbabilities.append(self.probabilities.copy())

    def _prob(self):

        diff = 2*self.activity - sum(self.activity)
        p = 1.0 / (1.0 + exp(-self.theta*diff))

        self.probabilities = p

    def _newAct(self):
        """ Calculate the new probabilities of different actions """

        self.activity = self.activity + (self.information-self.activity)* self.alpha

    def _decision(self):

        prob = self.probabilities[0]

        if abs(prob-0.5)>self.beta:
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


