# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Regulatory fit effects in a choice task
                `Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007)`.
                Psychonomic Bulletin & Review, 14(6), 1125–32.
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""
import numpy as np

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task

from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])

deck_sets = {"WorthyMaddox": [[2, 2, 1, 1, 2, 1, 1, 3, 2, 6, 2, 8, 1, 6, 2, 1, 1,
                               5, 8, 5, 10, 10, 8, 3, 10, 7, 10, 8, 3, 4, 9, 10, 3, 6,
                               3, 5, 10, 10, 10, 7, 3, 8, 5, 8, 6, 9, 4, 4, 4, 10, 6,
                               4, 10, 3, 10, 5, 10, 3, 10, 10, 5, 4, 6, 10, 7, 7, 10, 10,
                               10, 3, 1, 4, 1, 3, 1, 7, 1, 3, 1, 8],
                              [7, 10,  5, 10,  6,  6, 10, 10, 10,  8,  4,  8, 10,  4,  9, 10,  8,
                               6, 10, 10, 10,  4,  7, 10,  5, 10,  4, 10, 10,  9,  2,  9,  8, 10,
                               7,  7,  1, 10,  2,  6,  4,  7,  2,  1,  1,  1,  7, 10,  1,  4,  2,
                               1,  1,  1,  4,  1,  4,  1,  1,  1,  1,  3,  1,  4,  1,  1,  1,  5,
                               1,  1,  1,  7,  2,  1,  2,  1,  4,  1,  4,  1]]}
default_decks = deck_sets["WorthyMaddox"]


class Decks(Task):
    """
    Based on the Worthy & Maddox 2007 paper "Regulatory fit effects in a choice task.

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Parameters
    ----------
    draws: int, optional
        Number of cards drawn by the participant
    decks: list of floats, optional
        The decks of cards
    discard: bool
        Defines if you discard the card not chosen or if you keep it.
    """

    number_cues = 1
    valid_actions = [0, 1]

    def __init__(self, draws: Optional[int] = None,
                 decks: Optional[Union[str, List[List[float]]]] = None,
                 discard: Optional[bool] = False):

        self.discard = discard

        if isinstance(decks, str):
            if decks in deck_sets:
                self.decks = deck_sets[decks]
            else:
                raise Exception("Unknown deck sets")
        elif isinstance(decks, list):
            self.decks = decks
        else:
            self.decks = default_decks

        if draws is not None:
            self.task_length = draws
        else:
            self.task_length = len(self.decks[0])

        # Set draw count
        self._trial_step: int = -1
        self._card_value: float = np.nan
        self._deck_drawn: Optional[Action] = None
        if self.discard:
            self._drawn: int = -1
        else:
            self._drawn: List[int] = [-1, -1]

    def next_trialstep(self) -> Tuple[List[Union[int, float]], List[Action]]:
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        next_valid_actions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self._trial_step += 1

        if self._trial_step == self.task_length:
            raise StopIteration

        next_stim = [1]
        next_valid_actions = [0, 1]

        return next_stim, next_valid_actions

    def action_feedback(self, action: Action) -> float:
        """
        Receives the next action from the participant and responds to the action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : float
        """

        deck_drawn: int = action

        if self.discard:
            card_drawn = self._drawn + 1
            self._drawn = card_drawn
        else:
            card_drawn = self._drawn[deck_drawn] + 1
            self._drawn[deck_drawn] = card_drawn

        self._card_value = self.decks[deck_drawn][card_drawn]

        self._deck_drawn = action

        return self._card_value


class StimulusDecksLinear(Stimulus):

    def process_stimulus(self, observation: int) -> Tuple[int, float]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
            The elements present of the stimulus
        stimuliActivity : float or list of float
            The activity of each of the elements

        """
        return 1, 1


class RewardDecksLinear(Rewards):
    """
    Processes the decks reward for models expecting just the reward

    """

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> float:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback


class RewardDecksNormalised(Rewards):
    """
    Processes the decks reward for models expecting just the reward, but in range [0,1]

    Parameters
    ----------
    max_reward : int, optional
        The highest value a reward can have. Default ``10``

    See Also
    --------
    model.OpAL
    """
    max_reward: float = 10

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> float:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback / self.max_reward


class RewardDecksPhi(Rewards):
    """
    Processes the decks reward for models expecting just the reward, but in range [0, 1]

    Parameters
    ----------
    phi : float
        The scaling value of the reward
    """

    phi: float = 1

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> float:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback * self.phi


class RewardDecksAllInfo(Rewards):
    """
    Processes the decks reward for models expecting the reward information
    from all possible actions

    Parameters
    ----------
    max_reward_val : int
        The highest value a reward can have
    min_reward_val : int
        The lowest value a reward can have
    number_actions : int
        The number of actions the participant can perform. Assumes the lowest
        valued action is 0

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The reward that is a float and action is {0,1}. The
        function returns a array of length (max_reward_val-min_reward_val)*number_actions.

    Examples
    --------
    >>> rew = RewardDecksAllInfo(max_reward_val=10, min_reward_val=1, number_actions=2)
    >>> rew.process_feedback(6, 0, [1])
    [1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    >>> rew.process_feedback(6, 1, [1])
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1.]
    """
    max_reward_val: int = 10
    min_reward_val: int = 1
    number_actions: int = 2

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> np.ndarray:
        """

        Returns
        -------
        modelFeedback:
        """
        num_diff_rewards = self.max_reward_val - self.min_reward_val + 1
        reward_proc = np.ones(num_diff_rewards * self.number_actions)
        reward_proc[num_diff_rewards*last_action + feedback - 1] += 1
        return reward_proc.T


class RewardDecksDualInfo(Rewards):
    """
    Processes the decks reward for models expecting the reward information
    from two possible actions.

    """
    maxRewardVal: float = 10
    epsilon: float = 1

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> np.ndarray:
        """

        Returns
        -------
        modelFeedback:
        """
        divisor = self.maxRewardVal + self.epsilon
        rew = (feedback / divisor) * (1 - last_action) + (1 - (feedback / divisor)) * last_action
        reward_processed = np.array([rew, 1-rew])
        return reward_processed


class RewardDecksDualInfoLogistic(Rewards):
    """
    Processes the decks rewards for models expecting the reward information
    from two possible actions.


    """
    max_reward_val: float = 10
    min_reward_val: float = 1
    epsilon: float = 0.3

    def process_feedback(self,
                         feedback: float,
                         last_action: int,
                         stimuli: List[int]
                         ) -> np.ndarray:
        """

        Returns
        -------
        modelFeedback:
        """
        mid = (self.max_reward_val + self.min_reward_val) / 2
        x = np.exp(self.epsilon * (feedback-mid))

        rew = (x/(1+x))*(1-last_action) + (1-(x/(1+x)))*last_action
        reward_processed = np.array([rew, 1-rew])
        return reward_processed
