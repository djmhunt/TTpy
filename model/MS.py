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

from numpy import exp, zeros, array, ones

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from plotting import dataVsEvents, lineplot
from utils import callableDetailsString


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
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    eta : float, optional
        Decision threshold parameter
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numStimuli : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numStimuli
    probActions : bool, optional
        Defines if the probabilities calculated by the model are for each
        action-stimulus pair or for actions. That is, if the stimuli values for
        each action are combined before the probability calculation.
        Default ``True``
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((numActions, numStimuli)) / numCritics)``
    activity : array, optional
        The initialisation of the `activity` of the neurons. The values are between ``[0,1]``
        Default ``ones((numActions, numStimuli)) / numCritics``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "M&S"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.alpha = kwargRemains.pop('alpha', 1)
        self.eta = kwargRemains.pop('eta', 0.5)

        self.activity = kwargRemains.pop('activity', ones((self.numActions, self.numStimuli)) / self.numCritics)
        # The alpha is an activation rate parameter. The paper uses a value of 1.

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decEta(expResponses=(1, 2), eta=self.eta))

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
        self.parameters["activity"] = self.activity

        self.probDifference = 0
        self.firstDecision = 0

        # Recorded information
        self.genStandardResultsStore()
        self.recActivity = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["Activity"] = array(self.recActivity)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recActivity.append(self.activity.copy())

    def rewardExpectation(self, observation, action, response):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ---------
        observation : {int | float | tuple}
            The set of stimuli
        action : int or NoneType
            The chosen action
        response : float or NoneType

        Returns
        -------
        expectedReward : float
            The expected reward
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimFunc(observation, action)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numStimuli > 1:
            actionExpectations = self.actStimMerge(self.activity, stimuli)
        else:
            actionExpectations = self.activity

        expectedReward = actionExpectations

        return expectedReward, stimuli, activeStimuli

    def delta(self, reward, expectation, action):
        """
        Calculates the comparison between the reward and the expectation

        Parameters
        ----------
        reward : float
            The reward value
        expectation : float
            The expected reward value
        action : int
            The chosen action

        Returns
        -------
        delta
        """

        modReward = self.rewFunc(reward, action)

        delta = modReward * (1-expectation)

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        # Find the new activities
        self._newActivity(delta)

        # Calculate the new probabilities
        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actExpectations = self.actStimMerge(self.activity, stimuliFilter)
            self.probabilities = self._prob(actExpectations)
        else:
            self.probabilities = self._prob(self.activity)

    def _newActivity(self, delta):

        self.activity += delta * self.alpha

    def _prob(self, expectation):
        p = 1.0 / (1.0 + exp(-self.beta*expectation))

        self.probabilities = p
        self.probDifference = p[0] - p[1]

    class modelSetPlot(modelSetPlot):

        """Class for the creation of plots relevant to the model set"""

        def _figSets(self):
            """ Contains all the figures """

            self.figSets = []

            # Create all the plots and place them in in a list to be iterated

            fig = self.dPChanges()
            self.figSets.append(('dPChanges', fig))

            fig = self.trial3_4Diff()
            self.figSets.append(('trial3_4Diff', fig))

        def dPChanges(self):
            """
            A graph reproducing figures 3 & 4 from the paper
            """

            gainLables = array(["Gain " + str(m["beta"]) for m in self.modelStore])

            dP = array([m["Probabilities"][:, 0] - m["Probabilities"][:, 1] for m in self.modelStore])
            events = array(self.modelStore[0]["Events"])

            axisLabels = {"title":"Confidence by Learning Trial for Different Gain Parameters"}
            axisLabels["xLabel"] = "Trial number"
            axisLabels["yLabel"] = r"$\Delta P$"
            axisLabels["y2Label"] = "Bead presented"
            axisLabels["yMax"] = 1
            axisLabels["yMin"] = 0
            eventLabel = "Beads drawn"

            fig = dataVsEvents(dP, events, gainLables, eventLabel, axisLabels)

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

            fig = lineplot(gain, dPDiff, [], axisLabels)

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
        return [1, 0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc


def blankRew():
    """
    Default reward processor. Does nothing. Returns reward

    Returns
    -------
    blankRewFunc : function
        The function expects to be passed the reward and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankRewFunc(reward):
        return reward

    blankRewFunc.Name = "blankRew"
    return blankRewFunc
