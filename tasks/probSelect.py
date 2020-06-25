# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
            Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, trial_duration., & Hutchison, K. E. (2007).
            Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
            doi:10.1073/pnas.0706111104

"""
import numpy as np

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])


class ProbSelect(Task):
    """
    Probabilistic selection task based on Genetic triple dissociation reveals multiple roles for dopamine in
    reinforcement learning. Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
    Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
    doi:10.1073/pnas.0706111104

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Parameters
    ----------
    reward_probability : float in range [0,1], optional
        The probability that a reward is given for choosing action A. Default
        is 0.7
    action_reward_probabilities : dictionary, optional
        A dictionary of the potential actions that can be taken and the
        probability of a reward.
        Default {0:rewardProb, 1:1-rewardProb, 2:0.5, 3:0.5}
    learning_action_pairs : list of tuples, optional
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

    number_cues = 1
    valid_actions = [0, 1, 2, 3]

    def __init__(self,
                 reward_probability: Optional[float] = 0.7,
                 learning_action_pairs: Optional[List[Tuple[Action, Action]]] = None,
                 action_reward_probabilities: Optional[Dict[Action, float]] = None,
                 learning_length: Optional[int] = 240,
                 test_length: Optional[int] = 60,
                 number_actions: Optional[int] = None,
                 reward_size: Optional[float] = 1):

        if learning_action_pairs is None:
            learning_action_pairs = [(0, 1), (2, 3)]

        if not action_reward_probabilities:
            action_reward_probabilities = {0: reward_probability,
                                           1: 1 - reward_probability,
                                           2: 0.5,
                                           3: 0.5}

        if not number_actions:
            number_actions = len(action_reward_probabilities)

        self.reward_probability = reward_probability
        self.action_reward_probabilities = action_reward_probabilities
        self.learning_action_pairs = learning_action_pairs
        self.learning_length = learning_length
        self.reward_size = reward_size
        self.task_length = learning_length + test_length
        self.number_actions = number_actions
        self.valid_actions = list(action_reward_probabilities.keys())
        self.action_sequence = self.__generate_action_sequence(learning_action_pairs,
                                                               learning_length,
                                                               test_length)

        self._action = None
        self._trial_step = -1
        self._reward_value = -1

    def next_trialstep(self) -> Tuple[None, List[Action]]:
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

        self._trial_step += 1

        if self._trial_step == self.task_length:
            raise StopIteration

        next_stimulus = None
        next_valid_actions = self.action_sequence[self._trial_step]

        return next_stimulus, next_valid_actions

    def action_feedback(self, action: Action) -> float:
        """
        Receives the next action from the participant and responds to the action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : None, int or float
        """
        # The probability of success varies depending on if it is chosen

        if self._trial_step < self.learning_length:
            action_reward_probabilities = self.action_reward_probabilities[action]

            if action_reward_probabilities >= np.random.rand(1):
                reward = self.reward_size
            else:
                reward = 0
        else:
            reward = np.nan

        self._action = action
        self._reward_value = reward

        return reward

    @staticmethod
    def __generate_action_sequence(learning_action_pairs: List[Tuple[Action, Action]],
                                   learning_length: int,
                                   test_length: int
                                   ) -> List[List[Action]]:

        pair_ids = list(range(len(learning_action_pairs)))
        action_pairs = np.array(learning_action_pairs)

        pairs = np.random.choice(pair_ids, size=learning_length, replace=True)
        action_sequence = list(action_pairs[pairs])

        for t in range(test_length):
            pairs = np.random.choice(pair_ids, size=2, replace=False)
            elements = np.random.choice([0, 1], size=2, replace=True)

            pair = [action_pairs[p, e] for p, e in zip(pairs, elements)]
            action_sequence.append(pair)

        return action_sequence


class StimulusProbSelectDirect(Stimulus):
    """
    Processes the selection stimuli for models expecting just the event

    Examples
    --------
    >>> stim = StimulusProbSelectDirect()
    >>> stim.process_stimulus(1)
    (1, 1)
    >>> stim.process_stimulus(0)
    (1, 1)
    """

    def process_stimulus(self, observation: None) -> Tuple[int, int]:
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

    def process_feedback(self, reward: float, action: Union[str, int], stimuli: Any) -> float:
        """

        Returns
        -------
        modelFeedback:
        """
        return reward
