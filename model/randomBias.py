# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""

from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np

import re
import itertools

from model.modelTemplate import Model


class RandomBias(Model):

    """A model replicating a participant who chooses randomly, but with a bias towards certain actions

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Parameters
    ----------
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    actionCodes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.discrete.weightProb
    """


    def __init__(self, expect=None, **kwargs):

        super(RandomBias, self).__init__(**kwargs)

        pattern = '^prob\d+$'
        actProbLab = sorted([k for k in kwargs if re.match(pattern, k)])
        actionProbs = []
        if len(actProbLab) != self.numActions:
            raise IndexError("Wrong number of action weights. Received {} instead of {}".format(len(actProbLab), self.numActions))
        else:
            for p in actProbLab:
                actionProbs.append(kwargs.pop(p))
        self.actionProbs = np.array(actionProbs) / np.sum(actionProbs)

        if expect is None:
            expect = np.ones((self.numActions, self.numCues)) / self.numCues
        self.expectations = expect

        for k, v in itertools.izip(actProbLab, self.actionProbs):
            self.parameters[k] = v
        self.parameters["expectation"] = self.expectations.copy()

        # Recorded information

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

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

        actionExpectations = self.actionProbs

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

        delta = 0

        return delta

    def updateModel(self, delta, action, stimuli, stimuliFilter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this trialstep

        """

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        self.probabilities = self.calcProbabilities()

    def calcProbabilities(self):
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

        probArray = self.actionProbs

        return probArray

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calcProbabilities()

        return probabilities
