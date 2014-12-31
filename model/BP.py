# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot

class BP(model):

    """The Beysian predictor model
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
    Parameters
    ----------
    theta : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    oneProb : array, optional
        The prior probability
    prior : array, optional
        The prior probability 
    """

    Name = "BP"

    def __init__(self,**kwargs):

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

    def _update(self,events,instance):
        """Processes updates to new actions"""
        
        event = events

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
        self.recInformation.append(self.information.copy())
        self.recProbabilities.append(self.probabilities.copy())
        self.recPosteriorProb.append(self.posteriorProb.copy())
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
