# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
from __future__ import division

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt

from experimentPlot import experimentPlot

class experiment(object):
    """The abstract experiment class from which all others inherit
    
    Many general methods for experiments are found only here
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    
    """

    Name = "Empty"

    def __init__(self,**kwargs):

        self.kwargs = kwargs

        self.reset()

    def reset(self):
        """ 
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        self.plotArgs = kwargs.pop('plotArgs',{})

        self.recAction = []

        self.parameters = {"Name": self.Name}


    def __iter__(self):
        """ Returns the iterator for the experiment"""

        return self

    def next(self):
        """
        Produces the next bead for the iterator
        
        Returns
        -------
        stimulus : None
        
        Raises
        ------
        StopIteration
        """

        # Since there is nothing to iterate over, just return the final state

        raise StopIteration
#        if self.index == self.maxIndex:
#            raise StopIteration
#
#        self.index += 1
#
#        self._storeState()
#
#        return self.data[self.index]

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

    def receiveAction(self,action):
        """ Receives the next action from the participant"""

        self.recAction.append(action)

    def procede(self):
        """
        Updates the experiment before the next timestep
        """

        pass

    def feedback(self):
        """ 
        Responds to the action from the participant
        
        For this experiment there is no possible response
        
        Returns
        -------
        feedback : None
        """
        return None

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = {"Name": self.Name,
                   "Actions": self.recAction}

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        pass

    def params(self):
        """ Returns the parameters of the experiment as a dictionary"""

        return self.parameters

    def plot(self):
        """ Returns a plotting class relavent for this experiment"""

        return self.experimentPlot, self.plotArgs

    class experimentPlot(experimentPlot):

        """Abstract class for the creation of plots relevant to a experiment"""