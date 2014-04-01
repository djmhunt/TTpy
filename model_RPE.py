# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from numpy import exp, zeros, array
from model import model

class model_RPE(model):

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.Name = "model_RPE"
        self.rateConst = kwargs.pop('rateConst',0.3)
        self.beta = kwargs.pop('beta',0.3)
        self.activity = zeros(2) + 0.05
        self.decision = None
        self.firstDecision = 0

        # Recorded information

        self.recInformation = []
        self.recAction = []
        self.recEvents = []
        self.recActivity = []
        self.recDecOneProb = []
        self.recDecision = []

    def action(self):
        """ Returns the action of the model"""

        self._newAct()

        self.currAction = self.decision

        self._decision()

        self.recDecOneProb.append(self.activity[0])

        self._storeState()

        return self.currAction

    def observe(self, event):
        """ Recieves the latest observation"""

        self.recEvents.append(event)

        #Calculate jar information
        self.information = array([event,1-event])

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
                    "DecOneProb": array(self.recDecOneProb),
                    "firstDecision": self.firstDecision,
                    "rateConst": self.rateConst}

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

        self.activity = self.activity + (self.information-self.activity)* self.rateConst

    def _decision(self):

        prob = self.activity[0]

        if abs(prob-0.5)>self.beta:
            if prob>0.5:
                self.decision = 1
            else:
                self.decision = 2
            if not self.firstDecision:
                self.firstDecision = len(self.recDecision) + 1
        else:
            self.decision = None


