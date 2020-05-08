# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the model QLearn as well as the paper:
                Meta-learning in Reinforcement Learning


"""
import numpy as np

from typing import Union, Tuple, List

from model.modelTemplate import Model


class QLearnMeta(Model):

    """The q-Learning algorithm with a second-order adaptive beta

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
    tau : float, optional
        Beta rate Sensitivity parameter for probabilities
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
        in to a decision. Default is model.decision.binary.eta
    """

    def __init__(self, alpha=0.3, tau=0.2, reward_d=None, reward_dd=None, expect=None, **kwargs):

        super(QLearnMeta, self).__init__(**kwargs)

        # A record of the kwarg keys, the variable they create and their default value

        self.tau = tau
        self.alpha = alpha

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect

        if reward_d is None:
            reward_d = 5.5 * np.ones((self.number_actions, self.number_cues))
        self.reward_d = reward_d
        if reward_dd is None:
            reward_dd = 5.5 * np.ones((self.number_actions, self.number_cues))
        self.reward_dd = reward_dd

        self.parameters["alpha"] = self.alpha
        self.parameters["tau"] = self.tau
        self.parameters["expectation"] = self.expectations.copy()

        self.beta = np.exp(self.reward_d - self.reward_dd)

        # Recorded information
        self.rec_reward_d = []
        self.rec_reward_dd = []
        self.rec_beta = []

    def returnTaskState(self) -> dict:
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["reward_d"] = np.array(self.rec_reward_d).T
        results["reward_dd"] = np.array(self.rec_reward_dd).T
        results["beta"] = np.array(self.rec_beta).T

        return results

    def storeState(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

        self.rec_reward_d.append(self.reward_d.flatten())
        self.rec_reward_dd.append(self.reward_dd.flatten())
        self.rec_beta.append(self.beta.flatten())

    def rewardExpectation(self, observation: Union[int, float, Tuple]) -> Tuple[np.ndarray, List[float], List[bool]]:
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

        active_stimuli, stimuli = self.stimulus_shaper.processStimulus(observation)

        action_expectations = self._action_expectations(self.expectations, stimuli)

        return action_expectations, stimuli, active_stimuli

    def delta(self, reward: float, expectation: float, action: int, stimuli: Union[int, float, Tuple]) -> float:
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

        mod_reward = self.reward_shaper.processFeedback(reward, action, stimuli)

        delta = mod_reward - expectation

        self.update_beta(reward, action)

        return delta

    def update_beta(self, reward: float, action: int) -> None:
        """
        Update the estimate of beta

        Parameters
        ----------
        reward : float
            The reward value
        action : int
            The chosen action

        """

        #self.reward_d += self.tau * (reward - self.reward_d)
        #self.reward_dd += self.tau * (self.reward_d - self.reward_dd)
        #self.beta = np.exp(self.reward_d - self.reward_dd)

        reward_d = self.reward_d[action]
        reward_dd = self.reward_dd[action]
        reward_d += self.tau * (reward - reward_d)
        reward_dd += self.tau * (reward_d - reward_dd)
        self.beta[action] = np.exp(reward_d - reward_dd)
        self.reward_d[action] = reward_d
        self.reward_dd[action] = reward_dd

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

        # Find the new activities
        self._new_expectations(action, delta, stimuli)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        act_expectations = self._action_expectations(self.expectations, stimuli)
        self.probabilities = self.calcProbabilities(act_expectations)

    def _new_expectations(self, action, delta, stimuli):

        new_expectations = self.expectations[action] + self.alpha*delta*stimuli/np.sum(stimuli)

        new_expectations = new_expectations * (new_expectations >= 0)

        self.expectations[action] = new_expectations

    def _action_expectations(self, expectations, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            action_expectations = self.actStimMerge(expectations, stimuli)
        else:
            action_expectations = expectations

        return action_expectations

    def calcProbabilities(self, action_values: np.ndarray) -> np.ndarray:
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

        return prob_array

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
