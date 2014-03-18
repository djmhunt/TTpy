# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from numpy import array

from model import model

class model_MS(model):

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a gneral template for a model"""

        self.Name = "model_M&S"

        self.action = 1

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.theta = kwargs.pop('theta',1)
        self.actparam = kwargs.pop('actparam',0.2)

        # Recorded information

        self.recAction = []
        self.recEvents = []

    def action(self):
        """ Returns the action of the model"""

        self._storeState()

        return self.action

    def observe(self,event):
        """ Recieves the latest observation"""

        self.recEvents.append(event)

    def feedback(self,response):
        """ Recieves the reaction to the action """

    def outputEvolution(self,folderName):
        """ Plots and saves files containing all the relavent data for this model """

        results = {"Name": self.Name,
                   "Actions":array(self.recAction),
                   "Events":array(self.recEvents)}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.action)

