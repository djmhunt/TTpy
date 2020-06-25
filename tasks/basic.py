# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Note: A simple example of a task class with all the necessary components
"""
import copy

import numpy as np

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])


class Basic(Task):
    """
    An example of a task with all the necessary components, but nothing changing

    Parameters
    ----------
    trials : int
        The number of trials in the task
    """

    valid_actions = [0, 1]
    number_cues = 1

    def __init__(self, trials: int = 100):

        self.nbr_of_trials = trials

        self._trial = -1  # start at -1 so first call to next will yield _trial 0
        self._action: Optional[Action] = None  # placeholder for what _action is taken

    def next_trialstep(self) -> Tuple[int, List[Action]]:
        """
        the task class is an iterator [link to iterator documentation]
        this function produces the next stimulus for the task iterator

        Returns
        -------
        stimulus : None
        next_valid_actions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self._trial += 1

        if self._trial == self.nbr_of_trials:
            raise StopIteration

        next_stimulus = 1
        next_valid_actions = self.valid_actions

        return next_stimulus, next_valid_actions

    def action_feedback(self, action: Action) -> int:
        """
        Receives the next action from the participant and responds to the action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : int
        """
        self._action = action

        return 1


class StimulusBasicSimple(Stimulus):
    """
    Processes the stimulus cues for models expecting just the event

    """

    def process_stimulus(self, observation: int) -> Tuple[int, int]:
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


class RewardBasicDirect(Rewards):
    """
    Processes the reward for models expecting just the reward
    """

    def process_feedback(self,
                         feedback: Union[int, float, Action],
                         last_action: Action,
                         stimuli: List[float]
                         ) -> Union[float, np.ndarray]:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback

