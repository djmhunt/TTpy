# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt


class experiment(object):

    """The documentation for the class"""

    Name = "Empty"

    def __init__(self,**kwargs):
        """ Creates a new experiment instance"""

        self.parameters = {"Name": self.Name}

    def __iter__(self):
        """ Returns the iterator for the experiment"""

        return self

    def next(self):
        """ Produces the next item for the iterator"""

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

    def procede(self):
        """Updates the experiment"""

    def feedback(self):
        """ Responds to the action from the participant"""

    def outputEvolution():
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = {"Name": self.Name}

        return results

    def plots(self, ivText, **models):

        return

    def _storeState():
        """ Stores the state of all the important variables so that they can be
        output later """

    def params(self):
        """ Returns the parameters of the experiment as a dictionary"""

        return self.parameters

    def plot(self):
        """ Returns a plotting class relavent for this experiment"""