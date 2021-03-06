# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the description on p134-135 of Reinforcement Learning, Sutton & Barto 1998
"""
import logging

import numpy as np

from model.modelTemplate import Model


class TDE(Model):
    """The td-Learning algorithm

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
    epsilon : float, optional
        Sensitivity parameter for probabilities
    gamma: float, optional
        future expectation discounting
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

    See Also
    --------
    model.TD0 : This model is heavily based on that one
    """

    def __init__(self, alpha=0.3, epsilon=0.1, gamma=0.3, expect=None, **kwargs):

        super(TDE, self).__init__(**kwargs)

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_cues
        self.expectations = expect

        self.lastAction = 0
        self.lastStimuli = np.ones(self.number_cues)

        self.parameters["alpha"] = self.alpha
        self.parameters["epsilon"] = self.epsilon
        self.parameters["gamma"] = self.gamma
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

        activeStimuli, stimuli = self.stimulus_shaper.processStimulus(observation)

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

        modReward = self.reward_shaper.processFeedback(reward, action, stimuli)

        delta = modReward - expectation

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

        # TDE updates the model in two parts: once the reward is received and a future discount once the next action
        # has been chosen

        # Find the new activities
        change = self.alpha*delta*stimuli/np.sum(stimuli)
        self._newExpect(action, change)

        # Calculate the new probabilities
        # We need to combine the expectations before calculating the probabilities
        actExpectations = self._actExpectations(self.expectations, stimuli)
        self.probabilities = self.calcProbabilities(actExpectations)

        self.lastStimuli = stimuli
        self.lastAction = action

    def _newExpect(self, action, change):

        newExpectations = self.expectations[action] + change

        newExpectations = newExpectations * (newExpectations >= 0)

        self.expectations[action] = newExpectations

    def _actExpectations(self, expectations, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            actionExpectations = self.actStimMerge(expectations, stimuli)
        else:
            actionExpectations = expectations

        return actionExpectations

    def calcProbabilities(self, actionValues):
        # type: (np.ndarray) -> np.ndarray
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

        cbest = actionValues == max(actionValues)
        deltaEpsilon = self.epsilon * (1 / self.number_actions)
        bestEpsilon = (1 - self.epsilon) / np.sum(cbest) + deltaEpsilon
        probArray = bestEpsilon * cbest + deltaEpsilon * (1 - cbest)

        return probArray

    def lastChoiceReinforcement(self):
        """
        Allows the model to update its expectations once the action has been chosen.
        """

        lastStimuli = self.lastStimuli

        change = self.alpha * self.gamma * self.expectedRewards[self.currAction] * lastStimuli/np.sum(lastStimuli)
        self._newExpect(self.lastAction, change)

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

