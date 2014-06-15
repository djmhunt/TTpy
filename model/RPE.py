# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from numpy import exp, zeros, array
from model import model

class RPE(model):

    """The documentation for the class"""

    Name = "RPE"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.alpha = kwargs.pop('alpha',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.activity = zeros(2) + 0.05
        self.decision = None
        self.firstDecision = 0

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "alpha": self.alpha}

        # Recorded information

        self.recInformation = []
        self.recAction = []
        self.recEvents = []
        self.recActivity = []
        self.recDecision = []

    def action(self):
        """ Returns the action of the model"""

        self._decision()

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
                    "alpha": self.alpha}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recInformation.append(self.information)
        self.recAction.append(self.currAction)
        self.recActivity.append(self.activity)
        self.recDecision.append(self.decision)

    def _newAct(self):
        """ Calculate the new probabilities of different actions """

        self.activity = self.activity + (self.information-self.activity)* self.alpha

    def _decision(self):

        prob = self.activity[0]

        if abs(prob-0.5)>self.beta:
            if prob>0.5:
                self.decision = 1
            else:
                self.decision = 2
        else:
            self.decision = None

    def plot(self):
        """Returns a plotting class relavent for this model"""

