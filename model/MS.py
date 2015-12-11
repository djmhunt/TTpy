# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    `Moore, S. C., & Sellen, J. L. (2006)`.
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
from __future__ import division, print_function

import logging

from numpy import exp, zeros, array
from types import NoneType

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from plotting import dataVsEvents, lineplot

class MS(model):

    """The Moore & Sellen model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    beta : float, optional
        Sensitivity parameter for probabilities
    eta : float, optional
        Decision threshold parameter
    oneProb : float in ``[0,1]``, optional
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar.
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    activity : array, optional
        The `activity` of the neurons. The values are between ``[0,1]``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "M&S"

    def __init__(self,**kwargs):
        self.oneProb = kwargs.pop('oneProb',0.85)
        self.beta = kwargs.pop('beta',4)
        self.alpha = kwargs.pop('alpha',1)
        self.eta = kwargs.pop('eta',0.5)
        self.prior = kwargs.pop('prior',array([0.5,0.5]))
        self.activity = kwargs.pop('activity',array([0.5,0.5]))
        # The alpha is an activation rate paramenter. The paper uses a value of 1.

        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',decEta(expResponses = (1,2), eta = self.eta))

        self.currAction = 1
        self.probabilities = zeros(2) + self.prior
        self.probDifference = 0
        self.activity = zeros(2) + self.activity
        self.decision = None
        self.firstDecision = 0
        self.lastObs = False
        self.validActions = None

        self.parameters = {"Name": self.Name,
                           "oneProb": self.oneProb,
                           "beta": self.beta,
                           "eta": self.eta,
                           "alpha": self.alpha,
                           "prior": self.prior,
                           "activity" : self.activity,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recActivity = []
        self.recDecision = []

    def action(self):
        """
        Returns
        -------
        action : integer or None
        """

        self.decision = self.decisionFunc(self.probabilities, validResponses = self.validActions)


        self.currAction = self.decision

        self.storeState()

        return self.currAction

    def outputEvolution(self):
        """ Returns all the relevent data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.parameters.copy()

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["Activity"] = array(self.recActivity)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            if type(events) is not NoneType:
                self._processEvent(events)

        elif instance == 'reac':
            if type(events) is not NoneType:
                self._processEvent(events)

    def _processEvent(self,events):

        event = self.stimFunc(events, self.currAction)

        self.recEvents.append(event)

        #Find the new activites
        self._newActivity(event)

        #Calculate the new probabilities
        self._prob()


    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.currAction])
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)

    def _newActivity(self, event):

        self.activity = self.activity + (1-self.activity) * event * self.alpha

    def _prob(self):
        p = 1.0 / (1.0 + exp(-self.beta*self.activity))

        self.probabilities = p
        self.probDifference = p[0] - p[1]

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

            gainLables = array(["Gain " + str(m["beta"]) for m in self.modelStore])

            dP = array([m["Probabilities"][:,0] - m["Probabilities"][:,1] for m in self.modelStore])
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

            gain = array([m["beta"] for m in self.modelStore])

            axisLabels = {"title":"Change in Confidence in Light of Disconfirmatory Evidence"}
            axisLabels["xLabel"] = "Trial number"
            axisLabels["yLabel"] = r"$\Delta P\left(4\right) - \Delta P\left(3\right)$"
#            axisLabels["yMax"] = 0
#            axisLabels["yMin"] = -0.5

            fig = lineplot(gain,dPDiff,[],axisLabels)

            return fig

def blankStim():
    """
    Default stimulus processor. Does nothing.Returns [1,0]

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return [1,0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc
