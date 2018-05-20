# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from numpy import exp, array, ones, expand_dims, repeat, apply_along_axis, fromiter, ndarray
from scipy.stats import dirichlet #, beta
from collections import OrderedDict
from itertools import izip

from model.modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decRandom
from utils import callableDetailsString


class BP(model):

    """The Bayesian predictor model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numCues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numCues
    validRewards : list, ndarray, optional
        The different reward values that can occur in the task. Default ``array([0, 1])``
    actionCodes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    betaInit : float, optional
        The initial values for the alpha and beta values of the beta distribution.
        Normally 0, 1/2 or 1. Default 1
    prior : array of floats in ``[0, 1]``, optional
        Ignored in this case
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.binary.decRandom
    """

    Name = "BP"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.alpha = kwargRemains.pop('alpha', 0.3)
        betaInit = kwargRemains.pop('betaInit', 1)
        self.validRew = kwargRemains.pop('validRewards', array([0, 1]))
        self.rewLoc = OrderedDict((k, v for k, v in izip(self.validRew, range(len(self.validRew)))))

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decRandom())

        self.dirichletVals = ones((self.numActions, self.numCues, len(self.validRew))) * betaInit
        self.expectations = self.updateExpectations(self.dirichletVals)

        self.genStandardParameterDetails()
        self.parameters["beta"] = self.beta
        self.parameters["alpha"] = self.alpha
        self.parameters["betaInit"] = betaInit

        # Recorded information
        self.genStandardResultsStore()
        self.recDirichletVals = []

    def outputEvolution(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["dirichletVals"] = array(self.recDirichletVals)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recDirichletVals.append(self.dirichletVals.copy())

    def rewardExpectation(self, observation):
        """Calculate the estimated reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ----------
        observation : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        actionExpectations : array of floats
            The expected rewards for each action
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimFunc(observation)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numCues > 1:
            actionExpectations = self.calcActExpectations(self.actStimMerge(self.dirichletVals, stimuli))
        else:
            actionExpectations = self.calcActExpectations(self.dirichletVals)

        return actionExpectations, stimuli, activeStimuli

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

        return modReward

    def updateModel(self, delta, action, stimuli, stimuliFilter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this timestep
        stimuli : list of float
            The weights of the different stimuli in this timestep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this timestep

        """

        # Find the new activities
        self._newExpect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        if self.numCues > 1:
            actionExpectations = self.calcActExpectations(self.actStimMerge(self.dirichletVals, stimuli))
        else:
            actionExpectations = self.calcActExpectations(self.dirichletVals)

        self.probabilities = self.calcProbabilities(actionExpectations)

    def _newExpect(self, action, delta, stimuli):

        self.dirichletVals[action, :, self.rewLoc[delta]] += self.alpha * stimuli/sum(stimuli)

        self.expectations = self.updateExpectations(self.dirichletVals)

    def calcProbabilities(self, actionValues):
        # type: (ndarray) -> ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        actionValues : 1D ndArray of floats

        Returns
        -------
        probArray : 1D ndArray of floats
            The probabilities associated with the actionValues
        """
        numerator = exp(self.beta * actionValues)
        denominator = sum(numerator)

        probArray = numerator / denominator

        return probArray

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calcProbabilities(self.expectedRewards)

        return probabilities

    def actStimMerge(self, dirichletVals, stimuli):

        dirVals = dirichletVals * expand_dims(repeat([stimuli], self.numActions, axis=0), 2)

        actDirVals = sum(dirVals, 1)

        return actDirVals

    def calcActExpectations(self, dirichletVals):

        actExpect = fromiter((sum(dirichlet(d).mean() * self.validRew) for d in dirichletVals), float, count=self.numActions)

        return actExpect

    def updateExpectations(self, dirichletVals):

        def sumFunc(p, r=[]):
            return sum(dirichlet(p).mean() * r)

        expectations = apply_along_axis(sumFunc, 2, dirichletVals, r=self.validRew)

        return expectations

def blankStim():
    """
    Default stimulus processor. Does nothing.Returns ([1,0], None)

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
        return ([1,0], None)

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