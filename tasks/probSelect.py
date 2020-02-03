# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
            Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
            Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
            doi:10.1073/pnas.0706111104

"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

import itertools

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards


class ProbSelect(Task):
    """
    Probabilistic selection task based on Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
                                        Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
                                        Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
                                        doi:10.1073/pnas.0706111104

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    reward_probability : float in range [0,1], optional
        The probability that a reward is given for choosing action A. Default
        is 0.7
    action_reward_probabilities : dictionary, optional
        A dictionary of the potential actions that can be taken and the
        probability of a reward.
        Default {0:rewardProb, 1:1-rewardProb, 2:0.5, 3:0.5}
    learnActPairs : list of tuples, optional
        The pairs of actions shown together in the learning phase.
    learning_length : int, optional
        The number of trials in the learning phase. Default is 240
    test_length : int, optional
        The number of trials in the test phase. Default is 60
    reward_size : float, optional
        The size of reward given if successful. Default 1
    number_actions : int, optional
        The number of actions that can be chosen at any given time, chosen at
        random from actRewardProb. Default 4

    Notes
    -----
    The task is broken up into two sections: a learning phase and a
    transfer phase. Participants choose between pairs of four actions: A, B, M1
    and M2. Each provides a reward with a different probability: A:P>0.5,
    B:1-P<0.5, M1=M2=0.5. The transfer phase has all the action pairs but no
    feedback. This class only covers the learning phase, but models are
    expected to be implemented as if there is a transfer phase.

    """

    def __init__(self,
                 reward_probability=0.7,
                 learning_action_pairs=None,
                 action_reward_probabilities=None,
                 learning_length=240,
                 test_length=60,
                 number_actions=None,
                 reward_size=1):

        if learning_action_pairs is None:
            learning_action_pairs = [(0, 1), (2, 3)]

        if not action_reward_probabilities:
            action_reward_probabilities = {0: reward_probability,
                                           1: 1 - reward_probability,
                                           2: 0.5,
                                           3: 0.5}

        if not number_actions:
            number_actions = len(action_reward_probabilities)

        super(ProbSelect, self).__init__()

        self.parameters["reward_probability"] = reward_probability
        self.parameters["action_reward_probabilities"] = action_reward_probabilities
        self.parameters["learning_action_pairs"] = learning_action_pairs
        self.parameters["learning_length"] = learning_length
        self.parameters["test_length"] = test_length
        self.parameters["number_actions"] = number_actions
        self.parameters["reward_size"] = reward_size

        self.t = -1
        self.reward_probability = reward_probability
        self.action_reward_probabilities = action_reward_probabilities
        self.learning_action_pairs = learning_action_pairs
        self.learning_length = learning_length
        self.reward_size = reward_size
        self.task_length = learning_length + test_length
        self.action = None
        self.reward_value = -1
        self.number_actions = number_actions
        self.choices = action_reward_probabilities.keys()

        self.action_sequence = self.__generate_action_sequence(action_reward_probabilities,
                                                               learning_action_pairs,
                                                               learning_length,
                                                               test_length)

        # Recording variables
        self.record_reward_values = [-1] * self.task_length
        self.record_actions = [-1] * self.task_length

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        next_valid_actions : Tuple of length 2 of ints
            The list of valid actions that the model can respond with.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.task_length:
            raise StopIteration

        next_stimulus = None
        next_valid_actions = self.action_sequence[self.t]

        return next_stimulus, next_valid_actions

    def receiveAction(self, action):
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model
        """

        self.action = action

    def feedback(self):
        """
        Responds to the action from the participant
        """
        # The probability of success varies depending on if it is choice

        if self.t < self.learning_length:
            action_reward_probabilities = self.action_reward_probabilities[self.action]

            if action_reward_probabilities >= np.random.rand(1):
                reward = self.reward_size
            else:
                reward = 0
        else:
            reward = float('Nan')

        self.reward_value = reward

        self.storeState()

        return reward

    def proceed(self):
        """
        Updates the task after feedback
        """

        pass

    def returnTaskState(self):
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standardResultOutput()

        results["rewVals"] = np.array(self.record_reward_values)
        results["Actions"] = np.array(self.record_actions)
        results["validAct"] = np.array(self.action_sequence)

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.record_actions[self.t] = self.action
        self.record_reward_values[self.t] = self.reward_value

    @staticmethod
    def __generate_action_sequence(action_reward_probability,
                                   learning_action_pairs,
                                   learning_length,
                                   test_length):

        pair_nums = range(len(learning_action_pairs))
        action_pairs = np.array(learning_action_pairs)

        pairs = np.random.choice(pair_nums, size=learning_length, replace=True)
        action_sequence = list(action_pairs[pairs])

        for t in xrange(test_length):
            pairs = np.random.choice(pair_nums, size=2, replace=False)
            elements = np.random.choice([0, 1], size=2, replace=True)

            pair = [action_pairs[p, e] for p, e in itertools.izip(pairs, elements)]
            action_sequence.append(pair)

        return action_sequence


class StimulusProbSelectDirect(Stimulus):
    """
    Processes the selection stimuli for models expecting just the event

    Examples
    --------
    >>> stim = StimulusProbSelectDirect()
    >>> stim.processStimulus(1)
    (1, 1)
    >>> stim.processStimulus(0)
    (1, 1)
    """

    def processStimulus(self, observation):
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
        stimuliActivity : float or list of float

        """
        return 1, 1


class RewardProbSelectDirect(Rewards):
    """
    Processes the probabilistic selection reward for models expecting just the reward

    """

    def processFeedback(self, reward, action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return reward
