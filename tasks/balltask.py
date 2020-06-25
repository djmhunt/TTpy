# -*- coding: utf-8 -*-
"""
pyhpdm version of the balltask task
TODO: describe tasks
"""
import copy

import numpy as np

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])


class Balltask(Task):
    # TODO: Describe parameters
    # each bag always contains balls of same color
    bag_colours = ['red', 'green', 'blue']
    valid_actions = list(range(len(bag_colours)))
    number_cues = 3

    def __init__(self,
                 nbr_of_bags: int = 6,
                 bag_colours: Optional[List[str]] = None,
                 balls_per_bag: Optional[int] = 3):

        if bag_colours is not None:
            self.bag_colours = bag_colours

        # check for counterbalance
        assert(nbr_of_bags % len(self.bag_colours) == 0), "nbr of bags should be multiple of color count"
        
        bag_sequence = np.repeat(self.bag_colours, nbr_of_bags / len(self.bag_colours))
        self.bag_sequence = np.random.permutation(bag_sequence)

        self.nbr_of_bags = nbr_of_bags
        self.balls_per_bag = balls_per_bag
        self.nbr_of_trials = nbr_of_bags * balls_per_bag
        self.valid_actions = list(range(0, len(self.bag_colours)))

        # variables internal to a task instance
        self._trial = -1
        self._bag = -1
        self._action = None
        self._ball_colour = None
#        self.reward = None

    def next_trialstep(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : (0, 1, 2) representing red, green, blue in default case
            but can be many colors. it's assumed this always goes in same order
            left to right as bag_colors parameter

        Raises
        ------
        StopIteration
        """

        self._trial += 1
        
        if self._trial == self.nbr_of_trials:
            raise StopIteration

        # on first trial, bag is 0, go to next bag when all balls in bag are shown
        if self._trial == 0:
            self._bag = 0
        elif self._trial % self.balls_per_bag == 0:
            self._bag += 1

        next_stimulus = self.bag_sequence[self._bag]
        self._ball_colour = next_stimulus

        next_valid_actions = self.valid_actions  # (0, 1, 2) for RGB
#        next_valid_actions = tuple(self.bag_colors)

        return next_stimulus, next_valid_actions

    def action_feedback(self, action: Action) -> None:
        """
        Responds to the action from the participant
        balltask has no rewards so we return None

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : None, int or float
        """
        self._action = action

        return None


class StimulusBalltaskSimple(Stimulus):
    """
    Processes the stimulus cues for models expecting just the event

    """

    # TODO! change below to work for more colors than 3

    def process_stimulus(self, observation: str) -> Tuple[List[int], List[int]]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
            The elements present of the stimulus
        stimuliActivity : float or list of float
            The activity of each of the elements

        """
        if observation == "red":
            return [1, 0, 0], [1, 0, 0]
        if observation == "green":
            return [0, 1, 0], [0, 1, 0]
        if observation == "blue":
            return [0, 0, 1], [0, 0, 1]


class RewardBalltaskDirect(Rewards):
    """
    Processes the reward for models expecting just the reward
    """

    def process_feedback(self, feedback, last_action: Action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback
