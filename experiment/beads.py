# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

#matplotlib.interactive(True)
import logging

from numpy import array, zeros
from numpy.random import rand
from experiment import experiment
from plotting import dataVsEvents

defaultBeads = [1,1,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0]

class beads(experiment):

    """The documentation for the class"""

    def __init__(self,**kwargs):
        """ Creates a new experiment instance

        N: Number of beads that could potentially be shown
        beadSequence: The sequence of beads"""

        self.Name = "beads"

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

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = { "Name": self.Name,
                    "Observables":array(self.recBeads)}

        return results

    def plots(self, ivText, **models):

        figSets = []

        fig = self.plotProbJar1(ivText, **models)

        figSets.append(('Actions',fig))

        return figSets

    def plotProbJar1(self, ivText, **models):
        """
        Plots a set of lines for the probability of jar 1.

        plotProbJar1(self, ivText, **models)
        """

        data = []
        labels = []

        for label,v in models.iteritems():
            data.append(v["DecOneProb"])
            labels.append(ivText + ": " + label)

        events = self.recBeads

        axisLabels = {"title":"Opinion of next bead being white"}
        axisLabels["xLabel"] = "Time"
        axisLabels["yLabel"] = "Probability of Jar 1"
        axisLabels["y2Label"] = "Bead presented"
        axisLabels["yMax"] = 1
        axisLabels["yMin"] = 0
        eventLabel = "Beads drawn"

        fig = dataVsEvents(data,events,labels,eventLabel,axisLabels)

        return fig

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recBeads[self.t] = self.beads[self.t]

def generateSequence(numBeads, oneProb, switchProb):

    sequence = zeros(numBeads)

    probs = rand(numBeads,2)
    bead = 1

    for i in range(numBeads):
        if probs[i,1]< switchProb:
            bead = 1-bead

        if probs[i,0]< oneProb:
            sequence[i] = bead
        else:
            sequence[i] = 1-bead

    return sequence

