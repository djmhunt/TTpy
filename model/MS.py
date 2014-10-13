# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from plotting import dataVsEvents, lineplot

class MS(model):

    """The documentation for the class"""

    Name = "M&S"

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.currAction = 1
        self.information = zeros(2)
        self.probabilities = zeros(2)
        self.probDifference = 0
        self.activity = zeros(2)
        self.decision = None
        self.firstDecision = 0
        self.lastObs = False

        self.oneProb = kwargs.pop('oneProb',0.85)
        self.theta = kwargs.pop('theta',4)
        self.alpha = kwargs.pop('alpha',1)
        self.beta = kwargs.pop('beta',0.5)
        # The alpha is an activation rate paramenter. The paper uses a value of 1.

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "theta": self.theta,
                           "beta": self.beta,
                           "alpha": self.alpha}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recInformation = []
        self.recProbabilities = []
        self.recProbDifference = []
        self.recActivity = []
        self.recDecision = []

    def action(self):
        """ Returns the action of the model"""

        self._decision()

        self.currAction = self.decision

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
                   "oneProb": self.oneProb,
                   "theta": self.theta,
                   "beta": self.beta,
                   "alpha": self.alpha,
                   "Information": array(self.recInformation),
                   "Probabilities": array(self.recProbabilities),
                   "ProbDifference": array(self.recProbDifference),
                   "Activity": array(self.recActivity),
                   "Actions":array(self.recAction),
                   "Decsions": array(self.recDecision),
                   "Events":array(self.recEvents)}

        return results

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            self.recEvents.append(event)

            #Calculate jar information
            info = self.oneProb*event + (1-self.oneProb)*(1-event)
            self.information = array([info,1-info])

            #Find the new activites
            self._newActivity()

            #Calculate the new probabilities
            self._prob()

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:

                self.recEvents.append(event)

                #Calculate jar information
                info = self.oneProb*event + (1-self.oneProb)*(1-event)
                self.information = array([info,1-info])

                #Find the new activites
                self._newActivity()

                #Calculate the new probabilities
                self._prob()


    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)
        self.recInformation.append(self.information.copy())
        self.recProbabilities.append(self.probabilities.copy())
        self.recProbDifference.append(self.probDifference)
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)

    def _prob(self):
        p = 1.0 / (1.0 + exp(-self.theta*self.activity))

        self.probabilities = p
        self.probDifference = p[0] - p[1]

    def _newActivity(self):

        self.activity = self.activity + (1-self.activity) * self.information * self.alpha

    def _decision(self):

        prob = self.probDifference

        if abs(prob)>self.beta:
            if prob>0:
                self.decision = 1
            else:
                self.decision = 2
        else:
            self.decision = None

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        def _figSets(self):
            """ Contains all the figures """

            self.figSets = []

            # Create all the plots and place them in in a list to be iterated

            fig = self.dPChanges()
            self.figSets.append(('dPChanges',fig))

            fig = self.trial3_4Diff()
            self.figSets.append(('trial3_4Diff',fig))

        def dPChanges(self):

            gainLables = array(["Gain " + str(m["theta"]) for m in self.modelStore])

            dP = array([m["ProbDifference"] for m in self.modelStore])
            events = array(self.modelStore[0]["Events"])

            axisLabels = {"title":"Confidence by Learning Trial for Different Gain Parameters"}
            axisLabels["xLabel"] = "Trial number"
            axisLabels["yLabel"] = r"$\Delta P$"
            axisLabels["y2Label"] = "Bead presented"
            axisLabels["yMax"] = 1
            axisLabels["yMin"] = 0
            eventLabel = "Beads drawn"

            fig = dataVsEvents(dP,events,gainLables,eventLabel,axisLabels)

            return fig


        def trial3_4Diff(self):

            dPDiff = array([m["ProbDifference"][3]-m["ProbDifference"][2] for m in self.modelStore])

            gain = array([m["theta"] for m in self.modelStore])

            axisLabels = {"title":"Change in Confidence in Light of Disconfirmatory Evidence"}
            axisLabels["xLabel"] = "Trial number"
            axisLabels["yLabel"] = r"$\Delta P\left(4\right) - \Delta P\left(3\right)$"
#            axisLabels["yMax"] = 0
#            axisLabels["yMin"] = -0.5

            fig = lineplot(gain,dPDiff,[],axisLabels)

            return fig
