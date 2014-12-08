# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
from __future__ import division

from numpy import array
from modelSetPlot import modelSetPlot
from modelPlot import modelPlot

class model(object):

    """The documentation for the class"""

    Name = "model"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.currAction = 1
        self.lastObs = False

        self.parameters = {"Name" : self.Name}

        # Recorded information

        self.recAction = []
        self.recEvents = []

    def __eq__(self, other):

        if self.Name == other.Name:
            return True
        else:
            return False

    def __ne__(self, other):

        if self.Name != other.Name:
            return True
        else:
            return False

    def __hash__(self):

        return hash(self.Name)


    def action(self):
        """ Returns the action of the model"""

        self._storeState()

        return self.currAction

    def observe(self,event):
        """ Recieves the latest observation"""

        if event != None:
            self._update(event,'obs')

    def feedback(self,response):
        """ Recieves the reaction to the action """

        if response != None:
            self._update(response,'reac')

    def outputEvolution(self):
        """ Returns all the relavent data for this model """

        results = {"Name": self.Name,
                   "Actions":array(self.recAction),
                   "Events":array(self.recEvents)}

        return results

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            self.recEvents.append(event)

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(event)

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)

    def params(self):
        """ Returns the parameters of the model as a dictionary"""

        return self.parameters

    def plot(self):
        """Returns a plotting class relavent for this model"""

        return self.modelPlot

    def plotSet(self):
        """Returns a plotting class relavent for a parameter analysis for this model"""

        return self.modelSetPlot

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        """Abstract class for the creation of plots relevant to a set of models"""






