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

    def __init__(self, trials: int = 100):

        super(Basic, self).__init__()

        self.nbr_of_trials = trials

        self.parameters["Trials"] = self.nbr_of_trials

        self.trial = -1  # start at -1 so first call to next will yield trial 0
        self.action: Optional[Action] = None  # placeholder for what action is taken

        self.action_history = [-1] * self.nbr_of_trials

    def __next__(self) -> Tuple[int, List[Action]]:
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

        self.trial += 1

        if self.trial == self.nbr_of_trials:
            raise StopIteration

        next_stimulus = 1
        next_valid_actions = [0, 1]

        return next_stimulus, next_valid_actions

    def receive_action(self, action):
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
        return 1

    def proceed(self):
        """
        Updates the task after feedback
        """
        pass

    def return_task_state(self) -> Dict[str, Any]:
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standard_result_output()

        results["participantActions"] = copy.copy(self.action_history)

        return results

    def store_state(self) -> None:
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.action_history[self.trial] = self.action


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

