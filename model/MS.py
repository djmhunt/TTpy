# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    `Moore, S. C., & Sellen, J. L. (2006)`. 
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9. 
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
from __future__ import division

import logging

from numpy import exp, zeros, array

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from plotting import dataVsEvents, lineplot

class MS(model):

    """The Moore & Sellen model
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    theta : float, optional
        Sensitivity parameter for probabilities
    beta : float, optional
        Decision threshold parameter
    oneProb : array, optional
        The prior probability
    """

    Name = "M&S"

    def __init__(self,**kwargs):

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
        """
        Returns
        -------
        action : integer or None
        """

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
        """ Returns all the relevent data for this model 
        
        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name, 
            Probabilities, Actions and Events.
        """

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

    def _update(self,events,instance):
        """Processes updates to new actions"""
        
        event = events

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


    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

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

    class modelSetPlot(modelSetPlot):
        
        """Class for the creation of plots relevant to the model set"""

        def _figSets(self):
            """ Contains all the figures """

            self.figSets = []

            # Create all the plots and place them in in a list to be iterated

            fig = self.dPChanges()
            self.figSets.append(('dPChanges',fig))

            fig = self.trial3_4Diff()
            self.figSets.append(('trial3_4Diff',fig))

        def dPChanges(self):
            """
            A graph reproducing figures 3 & 4 from the paper
            """

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
            """
            A graph reproducing figures 5 from the paper
            """

            dPDiff = array([m["ProbDifference"][3]-m["ProbDifference"][2] for m in self.modelStore])

            gain = array([m["theta"] for m in self.modelStore])

            axisLabels = {"title":"Change in Confidence in Light of Disconfirmatory Evidence"}
            axisLabels["xLabel"] = "Trial number"
            axisLabels["yLabel"] = r"$\Delta P\left(4\right) - \Delta P\left(3\right)$"
#            axisLabels["yMax"] = 0
#            axisLabels["yMin"] = -0.5

            fig = lineplot(gain,dPDiff,[],axisLabels)

            return fig
