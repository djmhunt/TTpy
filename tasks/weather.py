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

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

cueSets = {"Pickering": [[1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0],
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
defaultCues = cueSets["Pickering"]

actualityLists = {"Pickering": [2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2,
                                1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2,
                                2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1,
                                2, 1, 1, 1, 1, 2, 2, 2, nan, nan, nan, nan, nan,
                                nan, nan, nan, nan, nan, nan, nan, nan, nan],
                  "TestRew": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
                              0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
                              1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
                              0, 0, 0, 1, 1, 1, nan, nan, nan, nan, nan, nan,
                              nan, nan, nan, nan, nan, nan, nan, nan]}
defaultActualities = actualityLists["Pickering"]


class Weather(Task):
    """
    Based on the 1994 paper "Probabilistic classification learning in amnesia."

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    cueProbs : array of int, optional
        If generating data, the likelihood of each cue being associated with each actuality. Each row of the array
        describes one actuality, with each column representing one cue. Each column is assumed sum to 1
    number_cues : int, optional
        The number of cues
    learningLen : int, optional
        The number of trials in the learning phase. Default is 200
    testLen : int, optional
        The number of trials in the test phase. Default is 100
    actualities : array of int, optional
        The actual reality the cues pointed to; the correct response the participant is trying to get correct
    cues : array of floats, optional
        The stimulus cues used to guess the actualities
    """
    defaultCueProbs = [[0.2, 0.8, 0.2, 0.8], [0.8, 0.2, 0.8, 0.2]]

    def __init__(self, cueProbs=defaultCueProbs, learningLen=200, testLen=100, number_cues=None, cues=None, actualities=None):

        super(Weather, self).__init__()

        if not number_cues:
            number_cues = np.shape(cueProbs)
        if not cues:
            cues = genCues(number_cues, learningLen+testLen)
        if not actualities:
            actualities = genActualities(cueProbs, cues, learningLen, testLen)

        if isinstance(cues, str):
            if cues in cueSets:
                self.cues = cueSets[cues]
            else:
                raise Exception("Unknown cue sets")
        else:
            self.cues = cues

        if isinstance(actualities, str):
            if actualities in actualityLists:
                self.actualities = actualityLists[actualities]
            else:
                raise Exception("Unknown actualities list")
        else:
            self.actualities = actualities

        self.T = len(self.cues)

        self.parameters["Actualities"] = np.array(self.actualities)
        self.parameters["Cues"] = np.array(self.cues)

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
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        nextStim = self.cues[self.t]
        nextValidActions = (0, 1)

        return nextStim, nextValidActions

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


def genCues(number_cues, taskLen):
    """

    Parameters
    ----------
    cueProbs
    taskLen

    Returns
    -------
    cues
    """

    cues = []
    for t in range(taskLen):
        c = []
        while np.sum(c) in [0, number_cues]:
            c = (np.random.rand(number_cues) > 0.5) * 1
        cues.append(c)

    return np.array(cues)


def genActualities(cueProbs, cues, learningLen, testLen):
    """

    Parameters
    ----------
    cueProbs
    cues
    learningLen
    testLen

    Returns
    -------
    actions
    """
    actions = []

    if cueProbs is None:
        probs = {1: {0: 0.75}, 2: {0: 1, 1: 0.5}, 3: {2: 0.75}}
        for t in range(learningLen):
            c = cues[t]
            s = np.sum(c.reshape([2, 2]), 1)
            prob = probs[np.sum(s)][np.prod(s)]
            a = np.argmax(s)
            p = np.array([1-a, a]) * (prob-(1-prob)) + (1-prob)
            action = np.random.choice([0, 1], p=p)
            actions.append(action)
    else:
        for t in range(learningLen):
            visibleCueProbs = cues[t] * cueProbs
            actProb = np.sum(visibleCueProbs, 1)
            action = np.random.choice([0, 1], p=actProb / np.sum(actProb))
            actions.append(action)

    actions.extend([float("Nan")] * testLen)

    return np.array(actions)


class StimulusWeatherDirect(Stimulus):
    """
    Processes the weather stimuli for models expecting just the event

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


class RewardsWeatherDirect(Rewards):
    """
    Processes the weather reward for models expecting the reward feedback

    """

    def processFeedback(self, feedback, lastAction, stimuli):
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

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """

        if feedback == lastAction:
            return 1
        else:
            return 0


class RewardWeatherDualCorrection(Rewards):
    """
    Processes the decks reward for models expecting the reward correction
    from two possible actions.
    """
    epsilon = 1

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        rewardProc = np.zeros((2, len(stimuli))) + self.epsilon
        rewardProc[feedback, stimuli] = 1
        return np.array(rewardProc)
