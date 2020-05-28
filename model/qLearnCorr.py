# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the QLearn model and the choice autocorrelation equation in the paper
                Trial-by-trial data analysis using computational models.
                Daw, N. D. (2011).
                Decision Making, Affect, and Learning: Attention and Performance XXIII (pp. 3–38).
                http://doi.org/10.1093/acprof:oso/9780199600434.003.0001
"""
import logging

import numpy as np

from model.modelTemplate import Model


class QLearnCorr(Model):

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
        Sensitivity parameter for probabilities
    kappa : float, optional
        The autocorelation parameter for which positive values promote sticking and negative values promote alternation
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
        The initialisation of the the expected reward.
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

    See Also
    --------
    model.QLearn : This model is heavily based on that one
    """

    def __init__(self, alpha=0.3, beta=4, kappa=0.1, invBeta=None, expect=None, **kwargs):

        super(QLearnCorr, self).__init__(**kwargs)

        self.alpha = alpha
        if invBeta is not None:
            beta = (1 / invBeta) - 1
        self.beta = beta
        self.kappa = kappa

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect

        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["kappa"] = self.kappa
        self.parameters["expectation"] = self.expectations.copy()

        # Recorded information

    def return_task_state(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standard_results_output()

        return results

    def store_state(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.store_standard_results()

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

        actionExpectations = self._actExpectations(self.expectations, stimuli)

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

        delta = modReward - expectation

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
        self._newExpect(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        actExpectations = self._actExpectations(self.expectations, stimuli)
        self.probabilities = self.calculate_probabilities(actExpectations)

        self.lastAction = action

    def _newExpect(self, action, delta, stimuli):

        newExpectations = self.expectations[action] + self.alpha*delta*stimuli/np.sum(stimuli)

        newExpectations = newExpectations * (newExpectations >= 0)

        self.expectations[action] = newExpectations

    def _actExpectations(self, expectations, stimuli):

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
        probArray : 1D ndArray of floats
            The probabilities associated with the actionValues
        """
        lastAction = np.zeros(np.shape(action_values))
        lastAction[self.lastAction] = 1

        numerator = np.exp(self.beta * (action_values + self.kappa * lastAction))
        denominator = np.sum(numerator)

        probArray = numerator / denominator

        return probArray

    def actor_stimulus_probs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        probabilities = self.calculate_probabilities(self.expected_rewards)

        return probabilities
