# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Note: A simple example of a task class with all the necessary components
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import copy

from tasks.taskTemplate import Task

from model.modelTemplate import Stimulus, Rewards


class Basic(Task):
    """
    An example of a task with all the necessary components, but nothing changing

    Parameters
    ----------
    trials : int
        The number of trials in the task

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    """

    def __init__(self, trials=100):

        super(Basic, self).__init__()

        self.nbr_of_trials = trials

        self.parameters["Trials"] = self.nbr_of_trials

        self.trial = -1  # start at -1 so first call to next will yield trial 0
        self.action = None  # placeholder for what action is taken

        self.action_history = [-1] * self.nbr_of_trials

    def next(self):
        """
        the task class is an iterator [link to iterator documentation]
        this function produces the next stimulus for the task iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.trial += 1

        if self.trial == self.nbr_of_trials:
            raise StopIteration

        nextStimulus = 1
        nextValidActions = (0, 1)

        return nextStimulus, nextValidActions

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
        return 1

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

        results["participantActions"] = copy.copy(self.action_history)

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.action_history[self.trial] = self.action


class StimulusBasicSimple(Stimulus):
    """
    Processes the stimulus cues for models expecting just the event

    """

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
        return 1, 1


class RewardBasicDirect(Rewards):
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

