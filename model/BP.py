# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones
from types import NoneType

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decEta
from utils import callableDetailsString


class BP(model):

    """The Bayesian predictor model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    eta : float, optional
        Decision threshold parameter. Default ``0.3``
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    numStimuli : integer, optional
        The number of different reaction learning sets. Default ``2``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decEta
    """

    Name = "BP"

    def __init__(self, **kwargs):

        self.numStimuli = kwargs.pop('numStimuli', 2)
        self.beta = kwargs.pop('beta', 4)
        self.prior = kwargs.pop('prior', ones(self.numStimuli)*0.5)
        self.eta = kwargs.pop('eta', 0.3)

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decEta(expResponses=tuple(range(1, self.numStimuli+1)), eta=self.eta))

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "eta": self.eta,
                           "prior": self.prior,
                           "numStimuli": self.numStimuli,
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

        self.currAction = 1
#        if len(prior) != self.numStimuli:
#            raise warning.
        self.posteriorProb = array(self.prior)
        self.probabilities = array(self.prior)
        self.decProbabilities = array(self.prior)
        self.decision = None
        self.validActions = None

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recPosteriorProb = []
        self.recDecision = []

    def action(self):
        """
        Returns
        -------
        action : integer or None
        """

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
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["Actions"] = array(self.recAction)
        results["Decisions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _updateObservation(self, events):
        """Processes updates to new actions"""
        if type(events) is not NoneType:
            self._processEvent(events)
        self._processAction()

    def _updateReaction(self, events):
        """Processes updates to new actions"""
        if type(events) is not NoneType:
            self._processEvent(events)

    def _processEvent(self, events):

        event = self.stimFunc(events, self.currAction)

        self.recEvents.append(event)

        postProb = self._postProb(event, self.posteriorProb)
        self.posteriorProb = postProb

        #  Calculate the new probabilities
        self.probabilities = self._prob(postProb)

    def _processAction(self):

        self.decision, self.decProbabilities = self.decisionFunc(self.probabilities, self.currAction, validResponses=self.validActions)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.decProbabilities[self.currAction])
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)

    def _postProb(self, event, postProb):

        li = postProb * event
        newProb = li/sum(li)

        return newProb

    def _prob(self, expectation):

        numerator = exp(self.beta * expectation)
        denominator = sum(numerator)

        p = numerator / denominator

        return p
#
#        diff = 2*self.posteriorProb - sum(self.posteriorProb)
#        p = 1.0 / (1.0 + exp(-self.beta*diff))
#
#        self.probabilities = p


def blankStim():
    """
    Default stimulus processor. Does nothing.Returns [1,0]

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return [1, 0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc
