# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from numpy import exp, zeros, array

from model import model

class model_MS(model):

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.Name = "model_M&S"

        self.currAction = 1
        self.information = zeros(2)
        self.probabilities = zeros(2)
        self.probDifference = 0
        self.activity = zeros(2)
        self.decision = None
        self.firstDecision = 0

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.theta = kwargs.pop('theta',1)
        self.actParam = kwargs.pop('actParam',0.2)
        self.beta = kwargs.pop('beta',0.5)
        # The actParam is an activation rate paramenter. The paper uses a value of 1.

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

        self.currAction = self.probDifference

        self._decision()

        self._storeState()

        return self.currAction

    def observe(self,event):
        """ Recieves the latest observation"""

        self.recEvents.append(event)

        #Calculate jar information
        info = self.oneProb*event + (1-self.oneProb)*(1-event)
        self.information = array([info,1-info])

        #Find the new activites
        self._newActivity()

        #Calculate the new probabilities
        self._prob()

    def feedback(self,response):
        """ Recieves the reaction to the action """

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this model """

        results = {"Name": self.Name,
                   "oneProb": self.oneProb,
                   "theta": self.theta,
                   "actParam": self.actParam,
                   "Information": array(self.recInformation),
                   "Probabilities": array(self.recProbabilities),
                   "ProbDifference": array(self.recProbDifference),
                   "Activity": array(self.recActivity),
                   "Actions":array(self.recAction),
                   "Decsions": array(self.recDecision),
                   "firstDecision": self.firstDecision,
                   "Events":array(self.recEvents)}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)
        self.recInformation.append(self.information)
        self.recProbabilities.append(self.probabilities)
        self.recProbDifference.append(self.probDifference)
        self.recActivity.append(self.activity)
        self.recDecision.append(self.decision)

    def _prob(self):
        p = 1.0 / (1.0 + exp(-self.theta*self.activity))

        self.probabilities = p
        self.probDifference = p[0] - p[1]

    def _newActivity(self):
        self.activity = self.activity + (1-self.activity) * self.information * self.actParam

    def _decision(self):

        prob = self.probDifference

        if abs(prob)>self.beta:
            if prob>0:
                self.decision = 1
            else:
                self.decision = 2
            if not self.firstDecision:
                self.firstDecision = len(self.recDecision) + 1
        else:
            self.decision = None

