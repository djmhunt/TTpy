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

    This model currently only copes with two actions and single stimuli

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    eta : float, optional
        Decision threshold parameter. Default ``0``
    delta : float in range ``[0,1]``, optional
        The switch probability parameter. Default ``0``
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
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
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

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.eta = kwargRemains.pop('eta', 0)
        delta = kwargRemains.pop('delta', 0)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.decisionFunc = kwargRemains.pop('decFunc', decSingle(expResponses=tuple(range(0, self.numCritics))))

        self.genStandardParameterDetails()
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
        self.parameters["delta"] = delta

        # This way for the first run you always consider that you are switching
        self.previousAction = None
#        if len(prior) != self.numCritics:
#            raise warning.

        self.posteriorProb = ones(self.numActions) / self.numActions
        self.switchProb = 0
        self.stayMatrix = array([[1-delta, delta], [delta, 1-delta]])
        self.switchMatrix = array([[delta, 1-delta], [1-delta, delta]])
        self.actionLoc = {k: k for k in range(0, self.numActions)}

        # Recorded information
        self.genStandardResultsStore()
        self.recSwitchProb = []
        self.recPosteriorProb = []
        self.recActionLoc = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["SwitchProb"] = array(self.recSwitchProb)
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["ActionLocation"] = array(self.recActionLoc)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recSwitchProb.append(self.switchProb)
        self.recActionLoc.append(self.actionLoc.values())
        self.recPosteriorProb.append(self.posteriorProb.copy())

    def rewardExpectation(self, observation):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ---------
        observation : {int | float | tuple}
            The set of stimuli

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

        # # If there are multiple possible stimuli, filter by active stimuli and calculate
        # # calculate the expectations associated with each action.
        # if self.numStimuli > 1:
        #     actionExpectations = self.actStimMerge(self.posteriorProb, stimuli)
        # else:
        #     actionExpectations = self.posteriorProb

        actionExpectations = self.posteriorProb

        expectedReward = actionExpectations[action]

        return expectedReward, stimuli, activeStimuli

    def delta(self, reward, expectation, action, stimuli):
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
        stimuli : {int | float | tuple | None}
            The stimuli received

        Returns
        -------
        delta
        """

        modReward = self.rewFunc(reward, action, stimuli)

        delta = modReward * expectation

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        currAction = self.currAction

        # Find the new posterior probabilities
        postProb = self._postProb(delta, currAction)
        self.posteriorProb = postProb

        if self.probActions:
            # Then we need to combine the expectations before calculating the probabilities
            actPostProb = self.actStimMerge(postProb, stimuliFilter)
            priorProb = self._prob(actPostProb, currAction)
        else:
            priorProb = self._prob(postProb, currAction)

        self.probabilities = priorProb

        self.switchProb = self._switch(priorProb)

    def _postProb(self, delta, action):

        loc = self.actionLoc

        p = delta

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
