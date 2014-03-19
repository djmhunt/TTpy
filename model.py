# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from numpy import array

class model:

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.Name = "model_Empty"

        self.currAction = 1

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

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this model """

        results = {"Name": self.Name,
                   "Actions":array(self.recAction),
                   "Events":array(self.recEvents)}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)

