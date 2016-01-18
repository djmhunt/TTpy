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
from model.decision.binary import decSingle
from utils import callableDetailsString


class BPMS(model):

    """The Bayesian Predictor with Markovian Switching model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    eta : float, optional
        Decision threshold parameter. Default ``0``
    delta : float in range ``[0,1]``, optional
        The switch probability parameter. Default ``0``
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decSingle

    Notes
    -----
    The Markovian switcher is the same as that used in BHMM and the rest is
    taken from BP. It currently does not allow more than two actions, equally
    you can only have two stimuli
    """

    Name = "BPMS"

    def __init__(self, **kwargs):

        self.numCritics = kwargs.pop('numCritics', 2)
        self.prior = kwargs.pop('prior', ones(self.numCritics) * 0.5)

        self.beta = kwargs.pop('beta', 4)
        self.eta = kwargs.pop('eta', 0)
        delta = kwargs.pop('delta', 0)

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decSingle(expResponses=tuple(range(0, self.numCritics))))

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "eta": self.eta,
                           "delta": delta,
                           "prior": self.prior,
                           "numCritics": self.numCritics,
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

        self.currAction = 0
        # This way for the first run you always consider that you are switching
        self.previousAction = None
#        if len(prior) != self.numCritics:
#            raise warning.
        self.posteriorProb = array(self.prior)
        self.probabilities = array(self.prior)
        self.decProbabilities = array(self.prior)
        self.decision = None
        self.validActions = None
        self.lastObservation = None
        self.switchProb = 0
        self.stayMatrix = array([[1-delta, delta], [delta, 1-delta]])
        self.switchMatrix = array([[delta, 1-delta], [1-delta, delta]])
        self.actionLoc = {k: k for k in range(0, self.numCritics)}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recSwitchProb = []
        self.recPosteriorProb = []
        self.recDecision = []
        self.recActionLoc = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.parameters.copy()

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["SwitchProb"] = array(self.recSwitchProb)
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["ActionLocation"] = array(self.recActionLoc)
        results["Actions"] = array(self.recAction)
        results["Decisions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _updateModel(self, event):

        currAction = self.currAction

        postProb = self._postProb(event, self.posteriorProb, currAction)
        self.posteriorProb = postProb

        # Calculate the new probabilities
        priorProb = self._prob(postProb, currAction)
        self.probabilities = priorProb

        self.switchProb = self._switch(priorProb)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.actionLoc[self.currAction]])
        self.recSwitchProb.append(self.switchProb)
        self.recActionLoc.append(self.actionLoc.values())
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)

    def _postProb(self, event, postProb, action):

        loc = self.actionLoc

        p = postProb * event

        li = array([p[loc[action]], p[loc[1-action]]])

        newProb = li/sum(li)

        loc[action] = 0
        loc[1-action] = 1
        self.actionLoc = loc

        return newProb

    def _prob(self, postProb, action):
        """Return the new prior probability that each state is the correct one
        """

        # The probability of the current state being correct, given if the previous state was correct.
        if self.previousAction == action:
            # When the subject has stayed
            pr = self.stayMatrix.dot(postProb)
        else:
            # When the subject has switched
            pr = self.switchMatrix.dot(postProb)

        self.previousAction = action

        return pr

    def _switch(self, prob):
        """Calculate the probability that the participant switches choice

        Parameters
        ----------
        prob : array of floats
            The probabilities for the two options
        """

        pI = prob[1]
        ps = 1.0 / (1.0 - exp(-self.beta * (pI - self.eta)))

        return ps


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
