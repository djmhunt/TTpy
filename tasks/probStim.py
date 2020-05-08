# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
import numpy as np

from numpy import nan

from numpy import float as npfloat


from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

# TODO: Create a set of test cues
cue_sets = {"Test": []}
defaultCues = cue_sets["Test"]

actuality_lists = {}


class Probstim(Task):
    """
    Basic probabilistic

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    actualities: int, optional
        The actual reality the cues pointed to. The correct response the participant is trying to get correct
    cues: array of floats, optional
        The cues used to guess the actualities
    trialsteps: int, optional
        If no provided cues, it is the number of trialsteps for the generated set of cues. Default ``100``
    num_cues: int, optional
        If no provided cues, it is the number of distinct stimuli for the generated set of cues. Default ``4``
    correct_prob: float in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0.
        The default is ``0.8``
    correct_probabilities: list or array of floats in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0 for
        each of the different stimuli. Default ``[corrProb, 1-corrProb] * (numStimuli//2) + [corrProb] * (numStimuli%2)``
    rewardless_trialsteps: int, optional
        If no actuality is provided, it is the number of actualities at the end of the tasks that will have a
        ``None`` reward. Default ``2*numStimuli``
    """

    def __init__(self,
                 cues=None,
                 actualities=None,
                 trialsteps=100,
                 num_cues=4,
                 correct_prob=0.8,
                 correct_probabilities=None,
                 rewardless_trialsteps=None):

        super(Probstim, self).__init__()

        if isinstance(cues, str):
            if cues in cue_sets:
                self.cues = cue_sets[cues]
                self.T = len(self.cues)
                num_cues = len(self.cues[0])
            else:
                raise Exception("Unknown cue sets")
        elif isinstance(cues, (list, np.ndarray)):
            self.cues = cues
            self.T = len(self.cues)
            num_cues = len(self.cues[0])
        else:
            self.T = trialsteps
            stimuli = np.zeros((self.T, num_cues))
            stimuli[list(range(self.T)), np.random.randint(num_cues, size=self.T)] = 1
            self.cues = stimuli

        if isinstance(actualities, str):
            if actualities in actuality_lists:
                self.actualities = actuality_lists[actualities]
                rewardless_trialsteps = np.sum(np.isnan(np.array(self.actualities, dtype=npfloat)))
            else:
                raise Exception("Unknown actualities list")
        elif isinstance(actualities, (list, np.ndarray)):
            self.actualities = actualities
            rewardless_trialsteps = np.sum(np.isnan(np.array(actualities, dtype=npfloat)))
        else:
            corr_prob_default = [correct_prob, 1 - correct_prob] * (num_cues // 2) + [correct_prob] * (num_cues % 2)
            if not correct_probabilities:
                correct_probabilities = corr_prob_default
            if not rewardless_trialsteps:
                rewardless_trialsteps = 2 * num_cues
            corr_choice_prob = np.sum(self.cues * correct_probabilities, 1)
            correct_choice = list((np.random.rand(self.T) < corr_choice_prob) * 1)
            correct_choice[-rewardless_trialsteps:] = [nan] * rewardless_trialsteps
            self.actualities = correct_choice

        self.parameters["actualities"] = np.array(self.actualities)
        self.parameters["cues"] = np.array(self.cues)
        self.parameters["trialsteps"] = self.T
        self.parameters["rewardless_trialsteps"] = rewardless_trialsteps
        self.parameters["number_cues"] = num_cues

        # Set draw count
        self.t = -1
        self.action = None

        # Recording variables
        self.recAction = [-1] * self.T

    def __next__(self):
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

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        next_stim = self.cues[self.t]
        next_valid_actions = (0, 1)

        return next_stim, next_valid_actions

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
        Feedback to the action from the participant
        """

        response = self.actualities[self.t]

        self.storeState()

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

        results = self.standardResultOutput()

        results["Actions"] = self.recAction

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action


class StimulusProbStimDirect(Stimulus):
    """
    Processes the stimuli for models expecting just the event

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

        return observation, observation


class RewardProbStimDiff(Rewards):
    """
    Processes the reward for models expecting reward corrections
    """

    def processFeedback(self, feedback, last_action, stimuli):
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

    def processFeedback(self, feedback, last_action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        reward_processed = np.zeros((2, len(stimuli))) + self.epsilon
        reward_processed[feedback, stimuli] = 1
        return np.array(reward_processed)

