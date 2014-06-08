# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from numpy import exp, zeros, array
from model import model

class EP(model):

    """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.Name = "EP"
        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.theta = kwargs.pop('theta',4)
        self.activity = zeros(2) + 0.5
        self.decision = None
        self.firstDecision = 0
        self.probabilities = zeros(2)

        # Recorded information

        self.recInformation = []
        self.recAction = []
        self.recEvents = []
        self.recActivity = []
        self.recDecOneProb = []
        self.recDecision = []
        self.recProbabilities = []

    def action(self):
        """ Returns the action of the model"""

        self._decision()

        self.recDecOneProb.append(self.probabilities[0])

        self.currAction = self.decision

        self._storeState()

        return self.currAction

    def observe(self, event):
        """ Recieves the latest observation"""

        self.recEvents.append(event)

        #Calculate jar information
        self.information = array([event,1-event])

        #Find the new activites
        self._newAct()

        #Calculate the new probabilities
        self._prob()

    def feedback(self,response):
        """ Recieves the reaction to the action """

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this model """

        results = { "Name": self.Name,
                    "Actions":array(self.recAction),
                    "Events":array(self.recEvents),
                    "Information": array(self.recInformation),
                    "Activity": array(self.recActivity),
                    "Decsions": array(self.recDecision),
                    "Probabilities": array(self.recProbabilities),
                    "DecOneProb": array(self.recDecOneProb),
                    "firstDecision": self.firstDecision,
                    "alpha": self.alpha}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recInformation.append(self.information)
        self.recAction.append(self.currAction)
        self.recActivity.append(self.activity)
        self.recDecision.append(self.decision)
        self.recProbabilities.append(self.probabilities)

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
                self.decision = 1
            else:
                self.decision = 2
            if not self.firstDecision:
                self.firstDecision = len(self.recDecision) + 1
        else:
            self.decision = None


