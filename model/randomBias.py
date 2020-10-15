# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
import logging

import numpy as np

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
    prob* : float, optional
        The probabilities for each action. Can be un-normalised. The parameter names are ``prob`` followed by a number
        e.g. ``prob1``, ``prob2``. It is expected that there will be same number as ``number_actions``.
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    action_codes : dict with string or int as keys and int values, optional
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

        if 'prob0' not in kwargs:
            kwargs['prob0'] = 0.5
        if 'prob1' not in kwargs:
            kwargs['prob1'] = 0.5
        pattern_parameters = self.add_pattern_parameters(kwargs, patterns=['^prob\\d+$'])

        number_pattern_parameters = len(pattern_parameters)

        if number_pattern_parameters != self.number_actions:
            raise IndexError(
                "Wrong number of action weights. Received {} instead of {}".format(number_pattern_parameters,
                                                                                   self.number_actions))

        action_probabilities = list(pattern_parameters.values())
        self.actionProbs = np.array(action_probabilities) / np.sum(action_probabilities)

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect

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

        modReward = self.reward_shaper.process_feedback(reward, action, stimuli)

        delta = 0

        return delta

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

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        self.probabilities = self.calculate_probabilities()

    def calculate_probabilities(self):
        # type: (np.ndarray) -> np.ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        action_values : 1D ndArray of floats

        Returns
        -------
        probArray : 1D ndArray of floats
            The probabilities associated with the action_values
        """

        probArray = self.actionProbs

        return probArray

    def actor_stimulus_probs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calculate_probabilities()

        return probabilities
