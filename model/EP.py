# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import logging

from numpy import exp, zeros, array
from random import choice

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot

class EP(model):

    """
    The expectation prediction model

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
    activity : array, optional
        The `activity` of the neurons. The values are between [0,1]
    """

    Name = "EP"

    def __init__(self,**kwargs):

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
        """ Calculate the new probabilities of different actions """

        diff = 2*self.activity - sum(self.activity)
        p = 1.0 / (1.0 + exp(-self.theta*diff))

        self.probabilities = p

    def _newAct(self):

        self.activity = self.activity + (self.information-self.activity)* self.alpha

    def _decision(self):

        prob = self.probabilities[0]

        if abs(prob-0.5)>self.beta:
            if prob>0.5:
                self.decision = 1
            elif prob == 0.5:
                self.decision = choice([1,2])
            else:
                self.decision = 2
        else:
            self.decision = None


