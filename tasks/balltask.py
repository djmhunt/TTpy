# -*- coding: utf-8 -*-
"""
pyhpdm version of the balltask task
TODO: describe tasks
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import copy

import numpy as np

from tasks.taskTemplate import Task

from model.modelTemplate import Stimulus, Rewards

class Balltask(Task):
    # TODO: Describe parameters
    # each bag always contains balls of same color
    def __init__(self, nbr_of_bags=6, bag_colors=['red', 'green', 'blue'], balls_per_bag=3):

        super(Balltask, self).__init__()

        # TODO: change nbr_of_bags default to 90
        
        # check for counterbalance
        assert(nbr_of_bags % len(bag_colors) == 0), "nbr of bags should be multiple of color count"
        
        bag_sequence = np.repeat(bag_colors, nbr_of_bags / len(bag_colors))
        bag_sequence = np.random.permutation(bag_sequence)

        self.parameters["nbr_of_bags"] = nbr_of_bags
        self.parameters["bag_colors"] = bag_colors
        self.parameters["balls_per_bag"] = balls_per_bag
        self.parameters["bag_sequence"] = list(bag_sequence)
        self.parameters["nbr_of_trials"] = nbr_of_bags * balls_per_bag

        # variables internal to a task instance
        self.trial = -1
        self.bag = -1
        self.action = None
        self.ballcolor = None
#        self.reward = None
        
        self.action_history = [-1] * self.parameters['nbr_of_trials']
#        self.reward_history = [-1] * self.parameters['nbr_of_trials']
        self.ball_history = [""] * self.parameters['nbr_of_trials']

    def next(self):
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

        self.trial += 1
        
        if self.trial == self.parameters['nbr_of_trials']:
            raise StopIteration

        # on first trial, bag is 0, go to next bag when all balls in bag are shown
        if self.trial == 0:
            self.bag = 0
        elif self.trial % self.parameters['balls_per_bag'] == 0:
            self.bag += 1

        next_stimulus = self.parameters['bag_sequence'][self.bag]
        self.ballcolor = next_stimulus
        
        valid_actions = np.arange(0, len(self.parameters['bag_colors']))
        next_valid_actions = tuple(valid_actions)  # (0, 1, 2) for RGB
#        next_valid_actions = tuple(self.parameters['bag_colors'])

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
        balltask has no rewards so we return None
        """
        self.storeState()

        return None

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
        history : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        history = self.standardResultOutput()

        history['participant_actions'] = copy.copy(self.action_history)
        history['ballcolor'] = copy.copy(self.ball_history)
        
        return history

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.action_history[self.trial] = self.action
        self.ball_history[self.trial] = self.ballcolor


class StimulusBalltaskSimple(Stimulus):
    """
    Processes the stimulus cues for models expecting just the event

    """

    # TODO! change below to work for more colors than 3

    def processStimulus(self, observation):
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
            return (1, 0, 0), (1, 0, 0)
        if observation == "green":
            return (0, 1, 0), (0, 1, 0)
        if observation == "blue":
            return (0, 0, 1), (0, 0, 1)


class RewardBalltaskDirect(Rewards):
    """
    Processes the reward for models expecting just the reward
    """

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback
