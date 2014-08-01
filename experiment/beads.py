# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

#matplotlib.interactive(True)
import logging

import pandas as pd

from numpy import array, zeros
from numpy.random import rand
from experiment import experiment
from plotting import dataVsEvents, varDynamics
from experimentPlot import experimentPlot
from utils import varyingParams


# Bead Sequences:
beadSequences = {"MooreSellen": [1,1,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0]}
defaultBeads = beadSequences["MooreSellen"]

class beads(experiment):

    """Based on the Moore&Sellen Beads task"""

    Name = "beads"

    def __init__(self,**kwargs):
        """ Creates a new experiment instance

        N: Number of beads that could potentially be shown
        beadSequence: The sequence of beads"""

        N = kwargs.pop('N',None)
        beadSequence = kwargs.pop("beadSequence",defaultBeads)

        self.plotArgs = kwargs.pop('plotArgs',{})

        if isinstance(beadSequence, str):
            if beadSequence in beadSequences:
                self.beads = beadSequences[beadSequence]
            else:
                raise "Unknown bead sequence"
        else:
            self.beads = beadSequence

        if N:
            self.T = N
        else:
            self.T = len(self.beads)

        self.parameters = {"Name": self.Name,
                           "N": self.T,
                           "beadSequence": self.beads}

        # Set timestep count
        self.t = -1

        # Recording variables

        self.recBeads = [-1]*self.T
        self.recAction = [-1]*self.T
        self.firstDecision = 0

    def next(self):
        """ Produces the next item for the iterator"""

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        self._storeState()

        return self.beads[self.t]

    def receiveAction(self,action):
        """ Receives the next action from the participant"""

        self.recAction[self.t] = action

        if action and not self.firstDecision:
            self.firstDecision = self.t + 1

    def procede(self):
        """Updates the experiment"""

    def feedback(self):
        """ Responds to the action from the participant"""

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = { "Name": self.Name,
                    "Observables":array(self.recBeads),
                    "Actions": self.recAction,
                    "FirstDecision" : self.firstDecision}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recBeads[self.t] = self.beads[self.t]

    class experimentPlot(experimentPlot):

        """Abstract class for the creation of plots relevant to a experiment"""

    #    def __init__(self, expSet, expParams, expLabel, modelSet, modelParams, modelLables):
    #
    #        self.expStore = expSet
    #        self.expParams = expParams
    #        self.expLabel = expLabel
    #        self.modelStore = modelSet
    #        self.modelParams = modelParams
    #        self.modelLabels = modelLabels
    #
    #        self._figSets()

        def _figSets(self):

            # Create all the plots and place them in in a list to be iterated

            self.figSets = []

            fig = self.plotProbJar1()
            self.figSets.append(('Actions',fig))

            fig = self.varCategoryDynamics()
            self.figSets.append(('decisionCoM',fig))

            fig = self.varDynamicPlot()
            self.figSets.append(("firstDecision",fig))

        def varDynamicPlot(self):

            params = self.modelParams[0].keys()

            paramSet = varyingParams(self.modelStore,params)
            decisionTimes = array([exp["FirstDecision"] for exp in self.expStore])

            fig = varDynamics(paramSet, decisionTimes, **self.plotArgs)

            return fig

        def varCategoryDynamics(self):

            params = self.modelParams[0].keys()
            #We assume that the parameters are the same for all the data to be analised,
            # otherwise this data is meaningless

            dataSet = varyingParams(self.modelStore,params)
            dataSet["decisionTimes"] = [exp["FirstDecision"] for exp in self.expStore]

            initData = pd.DataFrame(dataSet)


            maxDecTime = max(dataSet["decisionTimes"])
            if maxDecTime == 0:
                logger = logging.getLogger('categoryDynamics')
                message = "No decisions taken, so no useful data"
                logger.info(message)
                return

            dataSets = {d:initData[initData['decisionTimes'] == d] for d in range(1,maxDecTime+1)}

            CoM = pd.DataFrame([dS.mean() for dS in dataSets.itervalues()])

            CoM = CoM.set_index('decisionTimes')

            return CoM

        def plotProbJar1(self):
            """
            Plots a set of lines for the probability of jar 1.

            self.plotProbJar1(modelLables, modelSet)
            """

            data = [model["Probabilities"][:,0] for model in self.modelStore]

            events = self.expStore[0]["Observables"]

            axisLabels = {"title":"Opinion of next bead being white"}
            axisLabels["xLabel"] = "Time"
            axisLabels["yLabel"] = "Probability of Jar 1"
            axisLabels["y2Label"] = "Bead presented"
            axisLabels["yMax"] = 1
            axisLabels["yMin"] = 0
            eventLabel = "Beads drawn"

            fig = dataVsEvents(data,events,self.modelLabels,eventLabel,axisLabels)

            return fig



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

