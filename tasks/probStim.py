# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
import numpy as np

from numpy import nan

from numpy import float as npfloat

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])

# TODO: Create a set of test cues
cue_sets = {"Test": []}
defaultCues = cue_sets["Test"]

actuality_lists = {}


class ProbStim(Task):
    """
    Basic probabilistic stimulus task. Sometimes referred to as a biased coins task

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Parameters
    ----------
    actualities: int, optional
        The actual reality the cues pointed to. The correct response the participant is trying to get correct
    cues: array of floats, optional
        The cues used to guess the actualities
    task_length: int, optional
        If no provided cues, it is the number of trialsteps for the generated set of cues. Default ``100``
    number_cues: int, optional
        If no provided cues, it is the number of distinct stimuli for the generated set of cues. Default ``4``
    correct_prob: float in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0.
        The default is ``0.8``
    correct_probabilities: list or array of floats in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0 for
        each of the different stimuli. Default ``[correct_prob, 1-correct_prob] * num_cues/2`` with an additional
        correct_prob if there are an odd number of cues
    rewardless_trial_steps: int, optional
        If no actuality is provided, it is the number of actualities at the end of the tasks that will have a
        ``None`` reward. Default ``2*numStimuli``
    """

    number_cues = 1
    valid_actions = [0, 1]

    def __init__(self,
                 cues: Optional[Union[str, List[List[float]], np.ndarray]] = None,
                 actualities: Optional[Union[str, List[float], np.ndarray]] = None,
                 task_length: Optional[int] = 100,
                 number_cues: Optional[int] = 4,
                 correct_prob: Optional[float] = 0.8,
                 correct_probabilities: Optional[List[float]] = None,
                 rewardless_trial_steps: Optional[Union[str, int, List[float]]] = None):

        if isinstance(cues, str):
            if cues in cue_sets:
                self.cues = cue_sets[cues]
                self.task_length = len(self.cues)
                number_cues = len(self.cues[0])
            else:
                raise Exception("Unknown cue sets")
        elif isinstance(cues, (list, np.ndarray)):
            self.cues = cues
            self.task_length = len(self.cues)
            number_cues = len(self.cues[0])
        else:
            self.task_length = task_length
            stimuli = np.zeros((self.task_length, number_cues))
            stimuli[list(range(self.task_length)), np.random.randint(number_cues, size=self.task_length)] = 1
            self.cues = stimuli

        if isinstance(actualities, str):
            if actualities in actuality_lists:
                self.actualities = actuality_lists[actualities]
                rewardless_trial_steps = np.sum(np.isnan(np.array(self.actualities, dtype=npfloat)))
            else:
                raise Exception("Unknown actualities list")
        elif isinstance(actualities, (list, np.ndarray)):
            self.actualities = actualities
            rewardless_trial_steps = np.sum(np.isnan(np.array(actualities, dtype=npfloat)))
        else:
            corr_prob_default = [correct_prob, 1 - correct_prob] * (number_cues // 2)
            corr_prob_default.extend([correct_prob] * (number_cues % 2))

            if correct_probabilities is None:
                correct_probabilities = corr_prob_default
            if rewardless_trial_steps is None:
                rewardless_trial_steps = 2 * number_cues
            corr_choice_prob = np.sum(self.cues * correct_probabilities, 1)
            correct_choice = list((np.random.rand(self.task_length) < corr_choice_prob) * 1)
            correct_choice[-rewardless_trial_steps:] = [nan] * rewardless_trial_steps
            self.actualities = correct_choice

        self.number_cues = number_cues
        self._trial_step: int = -1
        self._action: Action = None

    def next_trialstep(self) -> Tuple[List[Union[int, float]], List[Action]]:
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

        self._trial_step += 1

        if self._trial_step == self.task_length:
            raise StopIteration

        next_stim = list(self.cues[self._trial_step])
        next_valid_actions = self.valid_actions

        return next_stim, next_valid_actions

    def action_feedback(self, action: Action) -> Union[int, float]:
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

        response = self.actualities[self._trial_step]
        self._action = action

        return response


class StimulusProbStimDirect(Stimulus):
    """
    Processes the stimuli for models expecting just the event

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


class RewardProbStimDiff(Rewards):
    """
    Processes the reward for models expecting reward corrections
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


class RewardProbStimDualCorrection(Rewards):
    """
    Processes the reward for models expecting the reward correction
    from two possible actions.
    """
    epsilon = 1

    def process_feedback(self, feedback: int, last_action: Action, stimuli: List[float]) -> np.ndarray:
        """

        Returns
        -------
        modelFeedback:
        """
        reward_processed = np.zeros((2, len(stimuli))) + self.epsilon
        reward_processed[feedback, stimuli] = 1
        return np.array(reward_processed)

