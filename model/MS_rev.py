# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    Moore, S. C., & Sellen, J. L. (2006).
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441

:Notes: In the original paper this model used a modified Luce choice
        algorithm, rather than the logistic algorithm used here.
"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class MS_rev(model):

    """An adapted version of the Morre & Sellen model

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
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    numCritics : integer, optional
        The number of different reaction learning sets. Default ``2``
    activity : array, optional
        The `activity` of the neurons. The values are between ``[0,1]``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "MS_rev"

    def __init__(self, **kwargs):

        self.numCritics = kwargs.pop('numCritics', 2)
        self.prior = kwargs.pop('prior', ones(self.numCritics) * 0.5)

        self.beta = kwargs.pop('beta', 4)
        self.alpha = kwargs.pop('alpha', 0.3)
        self.eta = kwargs.pop('eta', 0.3)
        self.activity = kwargs.pop('activity', ones(self.numCritics) * 0.5)
        # The alpha is an activation rate parameter. The M&S paper uses a value of 1.

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decEta(expResponses=(1, 2), eta=self.eta))

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "eta": self.eta,
                           "alpha": self.alpha,
                           "prior": self.prior,
                           "activity": self.activity,
                           "numCritics": self.numCritics,
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

        self.currAction = 1
        self.probabilities = array(self.prior)
        self.decProbabilities = array(self.prior)
        self.probDifference = 0
        self.activity = array(self.activity)
        self.decision = None
        self.validActions = None
        self.lastObservation = None

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recActivity = []
        self.recDecision = []

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
        results["Decisions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _updateModel(self, event):

        # Find the new activities
        self._newActivity(event)

        # Calculate the new probabilities
        self.probabilities = self._prob(self.activity)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be accessed later

        The stored variables are ones that describe the model.
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbabilities[self.currAction])
        self.recActivity.append(self.activity.copy())
        self.recDecision.append(self.decision)

    def _newActivity(self, event):
        self.activity += (event - self.activity) * self.alpha

    def _prob(self, expectation):
        # The probability of a given jar, using the Luce choice model

#        li = self.activity ** self.beta
#        p = li/sum(li)

        numerator = exp(self.beta*expectation)
        denominator = sum(numerator)

        p = numerator / denominator

        return p


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