# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Regulatory fit effects in a choice task
                Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007).
                Psychonomic Bulletin & Review, 14(6), 1125–32.
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""

import numpy as np

from typing import Union, Tuple, List, Optional

from model.modelTemplate import Model


class QLearn(Model):
    """The q-Learning algorithm

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
        Sensitivity parameter for probabilities. Also known as an exploration-
        exploitation parameter. Defined as :math:`\\beta` in the paper
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    number_cues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    number_critics : integer, optional
        The number of different reaction learning sets.
        Default number_actions*number_cues
    action_codes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((number_actions, number_cues)) / number_critics)``
    expect: array of floats, optional
        The initialisation of the expected reward.
        Default ``ones((number_actions, number_cues)) * 5 / number_cues``
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

    def __init__(self, *,
                 alpha: Optional[float] = 0.3,
                 beta: Optional[float] = 4,
                 invBeta: Optional[float] = None,
                 expect=None,
                 **kwargs):

        self.alpha = alpha
        if invBeta is not None:
            beta = (1 / invBeta) - 1
        self.beta = beta

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_critics
        self.expectations = np.array(expect)

    def reward_expectation(self, observation: Tuple[List[bool], List[float]]):
        """Calculate the estimated reward based on the action and stimuli

        This contains parts that are task dependent

        Parameters
        ----------
        observation : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        action_expectations : array of floats
            The expected rewards for each action
        stimuli : list of floats
            The processed observations
        active_stimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        active_stimuli, stimuli = self.stimulus_shaper.process_stimulus(observation)

        action_expectations = self._action_expectations(self.expectations, stimuli)

        return action_expectations, stimuli, active_stimuli

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

        mod_reward = self.reward_shaper.process_feedback(reward, action, stimuli)

        delta = mod_reward - expectation

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

        # Find the new activities
        self._new_expect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        act_expectations = self._action_expectations(self.expectations, stimuli)
        self.probabilities = self.calculate_probabilities(act_expectations)

    def _new_expect(self, action, delta, stimuli):

        new_expectations = self.expectations[action] + self.alpha*delta*stimuli/np.sum(stimuli)
        new_expectations = new_expectations * (new_expectations >= 0)
        self.expectations[action] = new_expectations

    def _action_expectations(self, expectations, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            actionExpectations = self.action_cue_merge(expectations, stimuli)
        else:
            actionExpectations = expectations

        return actionExpectations

    def calculate_probabilities(self, action_values):
        # type: (np.ndarray) -> np.ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        action_values : 1D ndArray of floats

        Returns
        -------
        prob_array : 1D ndArray of floats
            The probabilities associated with the actionValues
        """

        numerator = np.exp(self.beta * action_values)
        denominator = np.sum(numerator)

        prob_array = numerator / denominator

#        inftest = isinf(numerator)
#        if inftest.any():
#            possprobs = inftest * 1
#            probs = possprobs / np.sum(possprobs)
#
#            logger = logging.getLogger('QLearn')
#            message = "Overflow in calculating the prob with expectation "
#            message += str(expectation)
#            message += " \n Returning the prob: " + str(probs)
#            logger.warning(message)

        return prob_array

    def actor_stimulus_probs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calculate_probabilities(self._expected_rewards)

        return probabilities

