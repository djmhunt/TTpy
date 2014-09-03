# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot

class BP(model):

    """The documentation for the class"""

    Name = "BP"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.theta = kwargs.pop('theta',4)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.beta = kwargs.pop('beta',0.3)

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "theta": self.theta,
                           "beta": self.beta,
                           "prior": self.prior}

        self.currAction = 1
        self.information = zeros(2)
        self.posteriorProb = zeros(2) + self.prior
        self.probabilities = zeros(2) + self.prior
        self.decision = None
        self.firstDecision = 0
        self.lastObs = False

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recInformation = []
        self.recProbabilities = []
        self.recPosteriorProb = []
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
                   "beta": self.beta,
                   "prior": self.prior,
                   "Information": array(self.recInformation),
                   "Probabilities": array(self.recProbabilities),
                   "PosteriorProb": array(self.recPosteriorProb),
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

                #Calculate the new probabilities
                self._prob()

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)
        self.recInformation.append(self.information)
        self.recProbabilities.append(self.probabilities)
        self.recPosteriorProb.append(self.posteriorProb)
        self.recDecision.append(self.decision)

    def _prob(self):


        li = self.posteriorProb * self.information
        self.posteriorProb = li/sum(li)

#        self.probabilities = 1.0/(1.0 +exp(-self.theta*(self.posteriorProb-0.5)))

        diff = 2*self.posteriorProb - sum(self.posteriorProb)
        p = 1.0 / (1.0 + exp(-self.theta*diff))

        self.probabilities = p

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
