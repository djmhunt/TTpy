# -*- coding: utf-8 -*-
"""
pyhpdm version of the balltask experiment
TODO: describe experiment
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from experiment.experimentTemplate import experiment


class Balltask(experiment):

    Name = "balltask"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        # temp variables to increase readability
        nbr_of_bags = kwargs.pop("nbr_of_bags", 6)  # TODO: change to 90
        bag_colors = kwargs.pop("bag_colors", ['red', 'green', 'blue'])  # each bag always contains balls of same color
        balls_per_bag = kwargs.pop("balls_per_bag", 3)
        
        # check for counterbalance
        assert(nbr_of_bags % len(bag_colors) == 0), "nbr of bags should be multiple of color count"
        
        bag_sequence = np.repeat(bag_colors, nbr_of_bags / len(bag_colors))
        bag_sequence = np.random.permutation(bag_sequence)

        self.parameters = {
            "name"         : self.Name,
            "nbr_of_bags"  : nbr_of_bags,
            "bag_colors"   : bag_colors,
            "balls_per_bag": balls_per_bag,
            "bag_sequence" : list(bag_sequence),
            "nbr_of_trials": nbr_of_bags * balls_per_bag
        }

        # variables internal to an experiment instance
        self.trial = -1
        self.bag = -1
        self.action = None
        self.ballcolor = None
#        self.reward = None
        
        self.action_history = [-1] * self.parameters['nbr_of_trials']
#        self.reward_history = [-1] * self.parameters['nbr_of_trials']
        self.ball_history = [""] * self.parameters['nbr_of_trials'] 

        return self


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
        """
        self.action = action


    def feedback(self):
        """
        Responds to the action from the participant
        balltask has no rewards so we return None
        """
        self.storeState()
        return None


    def procede(self):
        """
        Updates the experiment after feedback
        """
        pass


    def outputEvolution(self):
        """
        Saves files containing all the relevant data for this experiment run
        """

        history = self.parameters.copy()
        history['participant_actions'] = self.action_history.copy()
        history['ballcolor'] = self.ball_history.copy()
        
        return history


    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.action_history[self.trial] = self.action
        self.ball_history[self.trial] = self.ballcolor


def balltaskStimulusDirect():
    """
    Processes the stimulus cues for models expecting just the event

    Returns
    -------
    basicStim : function
        The function returns a tuple of ``1`` and the observation.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2
    """

    # TODO! change below to work for more colors than 3

    def balltaskStimulus(observation):
        if observation == "red":
            return (1, 0, 0), (1, 0, 0)
        if observation == "green":
            return (0, 1, 0), (0, 1, 0)
        if observation == "blue":
            return (0, 0, 1), (0, 0, 1)

    balltaskStimulus.Name = "balltaskStimulusDirect"

    return balltaskStimulus


def balltaskRewardDirect():
    """
    Processes the reward for models expecting just the reward

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2
    """

    def balltaskReward(reward, action, stimuli):
        return reward

    balltaskReward.Name = "balltaskRewardDirect"

    return balltaskReward
