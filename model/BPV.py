# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from numpy import exp, array, ones, expand_dims, repeat, apply_along_axis, fromiter, ndarray, sum
from scipy.stats import dirichlet #, beta
from collections import OrderedDict
from itertools import izip

from model.modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.discrete import decWeightProb


class BPV(model):

    """The Bayesian predictor model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
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
    dirichletInit : float, optional
        The initial values for values of the dirichlet distribution.
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
        in to a decision. Default is model.decision.discrete.decWeightProb
    """

    Name = "BPV"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        self.alpha = kwargRemains.pop('alpha', 0.3)
        dirichletInit = kwargRemains.pop('dirichletInit', 1)
        self.validRew = kwargRemains.pop('validRewards', array([0, 1]))
        self.rewLoc = OrderedDict(((k, v) for k, v in izip(self.validRew, range(len(self.validRew)))))

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decWeightProb(range(self.numActions)))
        self.genEventModifiers(kwargRemains)

        self.dirichletVals = ones((self.numActions, self.numCues, len(self.validRew))) * dirichletInit
        self.initDirichletVals = self.dirichletVals.copy()
        self.expectations = self.updateExpectations(self.dirichletVals)
        self.beta = ones(self.numActions)

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["dirichletInit"] = dirichletInit

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

        actionExpectations = self._actExpectations(self.dirichletVals, stimuli)

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
        actionExpectations = self._actExpectations(self.dirichletVals, stimuli)
        self.probabilities = self.calcProbabilities(actionExpectations)

    def _newExpect(self, action, delta, stimuli):

        self.dirichletVals[action, :, self.rewLoc[delta]] += self.alpha * stimuli/sum(stimuli)

        self.expectations = self.updateExpectations(self.dirichletVals)

    def _actExpectations(self, dirichletVals, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numCues > 1:
            actionExpectations, actionUncertainty = self.calcActExpectations(self.actStimMerge(dirichletVals, stimuli))
            baselineExpectations, baselineUncertainty = self.calcActExpectations(self.actStimMerge(self.initDirichletVals, stimuli))
        else:
            actionExpectations, actionUncertainty = self.calcActExpectations(dirichletVals[:, 0, :])
            baselineExpectations, baselineUncertainty = self.calcActExpectations(self.initDirichletVals[:, 0, :])

        self.beta = baselineUncertainty / actionUncertainty - 1

        return actionExpectations

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

        actExpect = fromiter((sum(dirichlet(d).mean() * self.validRew) for d in dirichletVals), float)

        actUncertainty = fromiter((sum(dirichlet(d).var()) for d in dirichletVals), float)

        return actExpect, actUncertainty

    def updateExpectations(self, dirichletVals):

        expectations = apply_along_axis(self._meanFunc, 2, dirichletVals, r=self.validRew)

        return expectations

    def _meanFunc(self, w, r=[]):
        e = sum(dirichlet(w).mean() * r)
        return e

def blankStim():
    """
    Default stimulus processor. Does nothing.

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return event

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