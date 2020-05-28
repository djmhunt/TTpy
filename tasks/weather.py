# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Probabilistic classification learning in amnesia.
            Knowlton, B. J., Squire, L. R., & Gluck, M. a. (1994).
            Learning & Memory(Cold Spring Harbor, N.Y.), 1(2), 106–120.
            http://doi.org/10.1101/lm.1.2.106
"""
import numpy as np

from numpy import nan

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])

cue_sets = {"Pickering": [[1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0],
                          [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1],
                          [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 1],
                          [0, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                          [1, 1, 1, 0], [1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1],
                          [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0],
                          [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0],
                          [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1],
                          [1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1],
                          [1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1],
                          [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]]}
default_cues = cue_sets["Pickering"]

actuality_lists = {"Pickering": [2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2,
                                 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2,
                                 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1,
                                 2, 1, 1, 1, 1, 2, 2, 2, nan, nan, nan, nan, nan,
                                 nan, nan, nan, nan, nan, nan, nan, nan, nan],
                   "TestRew": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
                               0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
                               1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
                               0, 0, 0, 1, 1, 1, nan, nan, nan, nan, nan, nan,
                               nan, nan, nan, nan, nan, nan, nan, nan]}
default_actualities = actuality_lists["Pickering"]


class Weather(Task):
    """
    Based on the 1994 paper "Probabilistic classification learning in amnesia."

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Parameters
    ----------
    cue_probabilities : array of int, optional
        If generating data, the likelihood of each cue being associated with each actuality. Each row of the array
        describes one actuality, with each column representing one cue. Each column is assumed sum to 1
    number_cues : int, optional
        The number of cues
    learning_length : int, optional
        The number of trials in the learning phase. Default is 200
    test_length : int, optional
        The number of trials in the test phase. Default is 100
    actualities : array of int, optional
        The actual reality the cues pointed to; the correct response the participant is trying to get correct
    cues : array of floats, optional
        The stimulus cues used to guess the actualities
    """
    def __init__(self,
                 cue_probabilities: Optional[List[float]] = None,
                 learning_length: Optional[int] = 200,
                 test_length: Optional[int] = 100,
                 number_cues: Optional[int] = None,
                 cues: List[List[Union[int, float]]] = None,
                 actualities=None):

        super(Weather, self).__init__()

        if cue_probabilities is None:
            cue_probabilities = [[0.2, 0.8, 0.2, 0.8], [0.8, 0.2, 0.8, 0.2]]

        if number_cues is None:
            number_cues = len(cue_probabilities)
        if cues is None:
            cues = generate_cues(number_cues, learning_length + test_length)
        if not actualities:
            actualities = generate_actualities(cue_probabilities, cues, learning_length, test_length)

        if isinstance(cues, str):
            if cues in cue_sets:
                self.cues = cue_sets[cues]
            else:
                raise Exception("Unknown cue sets")
        else:
            self.cues = cues

        if isinstance(actualities, str):
            if actualities in actuality_lists:
                self.actualities = actuality_lists[actualities]
            else:
                raise Exception("Unknown actualities list")
        else:
            self.actualities = actualities

        self.task_length = len(self.cues)

        self.parameters["Actualities"] = self.actualities
        self.parameters["Cues"] = self.cues

        # Set draw count
        self.trial_step: int = -1
        self.action: Action = None

        # Recording variables
        self.record_action: List[int] = [-1] * self.task_length

    def __next__(self) -> Tuple[List[int], List[Action]]:
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : Tuple
            The current cues
        next_valid_actions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.trial_step += 1

        if self.trial_step == self.task_length:
            raise StopIteration

        next_stim = self.cues[self.trial_step]
        next_valid_actions = [0, 1]

        return next_stim, next_valid_actions

    def receive_action(self, action) -> None:
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model
        """

        self.action = action

    def feedback(self) -> Union[int, float]:
        """
        Feedback to the action from the participant
        """

        response = self.actualities[self.trial_step]

        self.store_state()

        return response

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

        results = self.standard_result_output()

        results["Actions"] = self.record_action

        return results

    def store_state(self) -> None:
        """ Stores the state of all the important variables so that they can be
        output later """

        self.record_action[self.trial_step] = self.action


def generate_cues(number_cues: int, task_length: int) -> List[List[int]]:
    """

    Parameters
    ----------
    number_cues
    task_length

    Returns
    -------
    cues
    """

    cues = []
    for t in range(task_length):
        c = []
        while np.sum(c) in [0, number_cues]:
            c = list((np.random.rand(number_cues) > 0.5) * 1)
        cues.append(c)

    return cues


def generate_actualities(cue_probabilities: Optional[List[float]],
                         cues: List[List[Union[int, float]]],
                         learning_length: int,
                         test_length: int
                         ) -> List[Union[int, float]]:
    """

    Parameters
    ----------
    cue_probabilities
    cues
    learning_length
    test_length

    Returns
    -------
    actions
    """
    actions = []

    if cue_probabilities is None:
        probabilities = {1: {0: 0.75}, 2: {0: 1, 1: 0.5}, 3: {2: 0.75}}
        for t in range(learning_length):
            c = np.array(cues[t])
            s = np.sum(c.reshape([2, 2]), 1)
            prob = probabilities[np.sum(s)][np.prod(s)]
            a = np.argmax(s)
            p = np.array([1-a, a]) * (prob-(1-prob)) + (1-prob)
            action = np.random.choice([0, 1], p=p)
            actions.append(action)
    else:
        cue_prob_array = np.array(cue_probabilities)
        for t in range(learning_length):
            visible_cue_probs = cues[t] * cue_prob_array
            act_prob = np.sum(visible_cue_probs, 1)
            action = np.random.choice([0, 1], p=act_prob / np.sum(act_prob))
            actions.append(action)

    actions.extend([nan] * test_length)

    return actions


class StimulusWeatherDirect(Stimulus):
    """
    Processes the weather stimuli for models expecting just the event

    """

    def process_stimulus(self, observation: List[int]) -> Tuple[List[int], List[int]]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
            The elements present of the stimulus
        stimuliActivity : float or list of float
            The activity of each of the elements

        """

        return observation, observation


class RewardsWeatherDirect(Rewards):
    """
    Processes the weather reward for models expecting the reward feedback

    """

    def process_feedback(self, feedback: float, last_action: Action, stimuli: List[float]) -> float:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback


class RewardWeatherDiff(Rewards):
    """
    Processes the weather reward for models expecting reward corrections
    """

    def process_feedback(self, feedback: Action, last_action: Action, stimuli: List[float]) -> int:
        """

        Returns
        -------
        modelFeedback:
        """

        if feedback == last_action:
            return 1
        else:
            return 0


class RewardWeatherDualCorrection(Rewards):
    """
    Processes the decks reward for models expecting the reward correction
    from two possible actions.
    """
    epsilon: float = 1

    def process_feedback(self, feedback: int, last_action: Action, stimuli: List[float]) -> np.ndarray:
        """

        Returns
        -------
        modelFeedback:
        """
        reward_proc = np.zeros((2, len(stimuli))) + self.epsilon
        reward_proc[feedback, stimuli] = 1
        return np.array(reward_proc)
