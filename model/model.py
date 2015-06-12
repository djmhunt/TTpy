# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from numpy import array
from modelSetPlot import modelSetPlot
from modelPlot import modelPlot

class model(object):

    """
    The model class is a general template for a model. It also contains 
    universal methods used by all models.
        
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting
    """

    Name = "model"

    def __init__(self,**kwargs):
        """"""

        self.currAction = 1
        self.lastObs = False
        self.validActions = None

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
        """
        Returns the action of the model
        
        Returns
        -------
        action : integer or None
        """

        self._storeState()

        return self.currAction

    def observe(self,state):
        """
        Receives the latest observation
        
        Parameters
        ----------
        state : float or None
            The stimulus from the experiment. Returns without doing anything if
            the value of event is `None`.
        
        """
        event, self.validActions = state
        
        if event != None:
            self._update(event,'obs')

    def feedback(self,response):
        """
        Receives the reaction to the action
        
        Parameters
        ----------
        response : float
            The stimulus from the experiment. Returns without doing anything if
            the value of response is `None`.
        """

        if response != None:
            self._update(response,'reac')

    def outputEvolution(self):
        """
        Returns all the relevent data for this model

        Returns
        -------
        results : dictionary
        """

        results = {"Name": self.Name,
                   "Actions":array(self.recAction),
                   "Events":array(self.recEvents)}

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            self.recEvents.append(events)

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(events)

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)

    def params(self):
        """
        Returns the parameters of the model
        
        Returns
        -------
        parameters : dictionary
        """

        return self.parameters

    def plot(self):
        """
        Returns a plotting class relavent for this model
        
        Returns
        -------
        modelPlot : model.modelPlot
        """

        return self.modelPlot

    def plotSet(self):
        """
        Returns a plotting class relavent analysis of sets of results from this 
        model
        
        Returns
        -------
        modelSetPlot : model.modelSetPlot
        """

        return self.modelSetPlot

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        """Abstract class for the creation of plots relevant to a set of models"""






