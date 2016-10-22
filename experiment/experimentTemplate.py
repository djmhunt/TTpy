# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt

from experimentPlot import experimentPlot
from experimentSetPlot import experimentSetPlot

class experiment(object):
    """The abstract experiment class from which all others inherit

    Many general methods for experiments are found only here

    Parameters
    ----------
    plotArgs : dict
        The arguments for plotting functions

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
        """
        Returns the iterator for the experiment
        """

        return self

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints
            The list of valid actions that the model can respond with. Set to
            ``None``, as they never vary.

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
#        return self.data[self.index], None

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
        """
        Receives the next action from the participant
        """

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
        """
        Returns all the relevent data for this experiment run

        Returns
        -------
        results : dictionary
            The dictionary contains a series of keys including Name,
            Observables and Actions.
        """

        results = {"Name": self.Name,
                   "Actions": self.recAction}

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        pass

    def params(self):
        """
        Returns the parameters of the experiment as a dictionary

        Returns
        -------
        parameters : dict
            The parameters of the experiment
        """

        return self.parameters

    def plot(self):
        """
        Returns a plotting class relevant for this experiment

        Returns
        -------
        experimentPlot : experiment.experimentPlot.experimentPlot
            The plots created for the experiment
        plotArgs : dict
            Plot arguments that may be used within the experimentPlot instance
        """

        return self.experimentPlot, self.plotArgs

    def plotSet(self):
        """
        Returns a plotting class relavent for this experiment set

        Returns
        -------
        experimentSetPlot : experiment.experimentSetPlot.experimentSetPlot
            The plots created for the experiment set
        plotArgs : dict
            Plot arguments that may be used within the experimentSetPlot instance
        """

        return self.experimentSetPlot, self.plotArgs

    class experimentPlot(experimentPlot):

        """Abstract class for the creation of plots relevant to a experiment"""

    class experimentSetPlot(experimentSetPlot):

        """Abstract class for the creation of plots relevant to a set of experiments"""