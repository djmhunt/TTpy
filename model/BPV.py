# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
import logging

import numpy as np
import scipy as sp

import collections
import itertools

from model.modelTemplate import Model


class BPV(Model):

    """The Bayesian predictor model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    number_cues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    number_critics : integer, optional
        The number of different reaction learning sets.
        Default number_actions*number_cues
    validRewards : list, np.ndarray, optional
        The different reward values that can occur in the task. Default ``array([0, 1])``
    action_codes : dict with string or int as keys and int values, optional
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
        in to a decision. Default is model.decision.discrete.weightProb
    """

    def __init__(self, alpha=0.3, dirichletInit=1, validRewards=np.array([0, 1]), **kwargs):

        super(BPV, self).__init__(**kwargs)

        self.alpha = alpha

        self.validRew = validRewards

        self.rewLoc = collections.OrderedDict(((k, v) for k, v in zip(self.validRew, range(len(self.validRew)))))

        self.dirichletVals = np.ones((self.number_actions, self.number_cues, len(self.validRew))) * dirichletInit
        self.initDirichletVals = self.dirichletVals.copy()
        self.expectations = self.updateExpectations(self.dirichletVals)
        self.beta = np.ones(self.number_actions)

        self.parameters["alpha"] = self.alpha
        self.parameters["dirichletInit"] = dirichletInit

        # Recorded information
        self.recDirichletVals = []

    def return_task_state(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standard_results_output()
        results["dirichletVals"] = np.array(self.recDirichletVals)

        return results

    def store_state(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.store_standard_results()
        self.recDirichletVals.append(self.dirichletVals.copy())

    def reward_expectation(self, observation):
        """Calculate the estimated reward based on the action and stimuli

        This contains parts that are task dependent

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

        activeStimuli, stimuli = self.stimulus_shaper.process_stimulus(observation)

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

        modReward = self.reward_shaper.process_feedback(reward, action, stimuli)

        return modReward

    def update_model(self, delta, action, stimuli, stimuli_filter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuli_filter : list of bool
            A list describing if a stimulus cue is present in this trialstep

        """

        # Find the new activities
        self._newExpect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        actionExpectations = self._actExpectations(self.dirichletVals, stimuli)
        self.probabilities = self.calculate_probabilities(actionExpectations)

    def _newExpect(self, action, delta, stimuli):

        self.dirichletVals[action, :, self.rewLoc[delta]] += self.alpha * stimuli/np.sum(stimuli)

        self.expectations = self.updateExpectations(self.dirichletVals)

    def _actExpectations(self, dirichletVals, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            actionExpectations, actionUncertainty = self.calcActExpectations(self.action_cue_merge(dirichletVals, stimuli))
            baselineExpectations, baselineUncertainty = self.calcActExpectations(self.action_cue_merge(self.initDirichletVals, stimuli))
        else:
            actionExpectations, actionUncertainty = self.calcActExpectations(dirichletVals[:, 0, :])
            baselineExpectations, baselineUncertainty = self.calcActExpectations(self.initDirichletVals[:, 0, :])

        self.beta = baselineUncertainty / actionUncertainty - 1

        return actionExpectations

    def calculate_probabilities(self, action_values):
        # type: (np.ndarray) -> np.ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        action_values : 1D np.ndarray of floats

        Returns
        -------
        probArray : 1D np.ndarray of floats
            The probabilities associated with the actionValues
        """
        numerator = np.exp(self.beta * action_values)
        denominator = np.sum(numerator)

        probArray = numerator / denominator

        return probArray

    def actor_stimulus_probs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D np.ndarray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calculate_probabilities(self.expected_rewards)

        return probabilities

    def action_cue_merge(self, dirichletVals, stimuli):

        dirVals = dirichletVals * np.expand_dims(np.repeat([stimuli], self.number_actions, axis=0), 2)

        actDirVals = np.sum(dirVals, 1)

        return actDirVals

    def calcActExpectations(self, dirichletVals):

        actExpect = np.fromiter((np.sum(sp.stats.dirichlet(d).mean() * self.validRew) for d in dirichletVals), float)

        actUncertainty = np.fromiter((np.sum(sp.stats.dirichlet(d).var()) for d in dirichletVals), float)

        return actExpect, actUncertainty

    def updateExpectations(self, dirichletVals):

        expectations = np.apply_along_axis(self._meanFunc, 2, dirichletVals, r=self.validRew)

        return expectations

    def _meanFunc(self, w, r=[]):
        e = np.sum(sp.stats.dirichlet(w).mean() * r)
        return e
