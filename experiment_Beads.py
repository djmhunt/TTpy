# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt

from numpy import array
from experiment import experiment

defaultBeads = [1,1,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0]

#Define the different types of lines that will be plotted and their properties.
dots = ['.', 'o', '^', 'x', 'd', '2', 'H', ',', 'v', '+', 'D', 'p', '<', 's', '1', 'h', '4', '>', '3']
scatterdots = ['o', '^', 'x', 'd', 'v', '+', 'p', '<', 's', 'h', '>', '8']
lines = ['-', '--', ':', '-.','-']
lines_width = [1,1,2,2,2]
large = ['^', 'x', 'D', '4', 'd', 'p', '>', '2', ',', '3', 'H']
large_line_width = [2]*len(large)

lpl = lines + large
lpl_linewidth = large_line_width + lines_width
colours = ['g','r','b','m','0.85'] + ['k']*len(large)

class experiment_Beads(experiment):

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,**kwargs):
        """ Creates a new experiment instance

        N: Number of beads that could potentially be shown
        beadSequence: The sequence of beads"""

        self.Name = "experiment_Beads"

        N = kwargs.pop('N',None)
        beadSequence = kwargs.pop("beadSequence",defaultBeads)

        self.beads = beadSequence
        if N:
            self.T = N
        else:
            self.T = len(beadSequence)

        # Set timestep count
        self.t = -1

        # Recording variables

        self.recBeads = [-1]*self.T

    def __iter__(self):
        """ Returns the iterator for the experiment"""

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        self._storeState()

        return self.beads[self.t]

    def receiveAction(self,action):
        """ Receives the next action from the participant"""

    def procede(self):
        """Updates the experiment"""

    def feedback(self):
        """ Responds to the action from the participant"""

    def outputEvolution(self,folderName):
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = { "Name": self.Name,
                    "Observables":array(self.recBeads)}

        return results

    def plots(self, ivText, **models):

        figSets = []

        fig = self.plotActions(ivText, **models)

        figSets.append(('Actions',fig))

        return figSets

    def plotActions(self, ivText, **models):

        fig = plt.figure(1)

        plt.plot(self.recBeads, 'o', label = "Beads drawn", color = 'k', linewidth=2,markersize = 3)

        for i, (label,v) in enumerate(models.iteritems()):

            plt.plot(v["Actions"], lpl[i], label = ivText + ": " + label, color = colours[i], linewidth=lpl_linewidth[i],markersize = 3)#, axes=axs[0])

        plt.xlabel("Bead")
        plt.ylabel("Action")
        plt.title("Opinion of next bead being white")
        leg = plt.legend(loc = 'best', fancybox=True)

        return fig

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recBeads[self.t] = self.beads[self.t]
