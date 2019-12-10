# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Probabilistic classification learning in amnesia.
            Knowlton, B. J., Squire, L. R., & Gluck, M. a. (1994).
            Learning & Memory(Cold Spring Harbor, N.Y.), 1(2), 106–120.
            http://doi.org/10.1101/lm.1.2.106
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from numpy import nan

from experiment.experimentTemplate import Experiment

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


class Weather(Experiment):
    """
    Based on the 1994 paper "Probabilistic classification learning in amnesia."

    Many methods are inherited from the experiment.experiment.experiment class.
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
    numCues : int, optional
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

    def __init__(self, cueProbs=defaultCueProbs, learningLen=200, testLen=100, numCues=None, cues=None, actualities=None, **kwargs):

        super(Weather, self).__init__(**kwargs)

        if not numCues:
            numCues = np.shape(cueProbs)
        if not cues:
            cues = genCues(numCues, learningLen+testLen)
        if not actualities:
            actualities = genActualities(cueProbs, cues, learningLen, testLen)

        if isinstance(cues, basestring):
            if cues in cueSets:
                self.cues = cueSets[cues]
            else:
                raise Exception("Unknown cue sets")
        else:
            self.cues = cues

        if isinstance(actualities, basestring):
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

    def next(self):
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
        Updates the experiment after feedback
        """

        pass

    def returnTaskState(self):
        """
        Returns all the relevant data for this experiment run

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


def genCues(numCues, taskLen):
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
    for t in xrange(taskLen):
        c = []
        while np.sum(c) in [0, numCues]:
            c = (np.random.rand(numCues) > 0.5) * 1
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
        for t in xrange(learningLen):
            c = cues[t]
            s = np.sum(c.reshape([2, 2]), 1)
            prob = probs[np.sum(s)][np.prod(s)]
            a = np.argmax(s)
            p = np.array([1-a, a]) * (prob-(1-prob)) + (1-prob)
            action = np.random.choice([0, 1], p=p)
            actions.append(action)
    else:
        for t in xrange(learningLen):
            visibleCueProbs = cues[t] * cueProbs
            actProb = np.sum(visibleCueProbs, 1)
            action = np.random.choice([0, 1], p=actProb / np.sum(actProb))
            actions.append(action)

    actions.extend([float("Nan")] * testLen)

    return np.array(actions)


def weatherStimDirect():
    """
    Processes the weather stimuli for models expecting just the event

    Returns
    -------
    weatherStim : function
        The function returns the event

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.QLearn, model.OpAL
    """

    def weatherStim(observation):

        return observation, observation

    weatherStim.Name = "weatherStimDirect"
    return weatherStim


def weatherStimAllAction(numActions):
    """
    Processes the weather stimuli for models expecting feedback from all
    possible actions

    Parameters
    ----------
    numActions : int
        The number of actions the participant can perform. Assumes the lowest
        valued action can be represented as 0

    Returns
    -------
    weatherStim : function
        The function expects to be passed a tuple containing the event and the
        last action. The event that is a list of {0,1} and action is {0,1}. The
        function returns a array of length (len(event))*numActions.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.BP, model.EP, model.MSRev, model.decision.binary.decEtaSet

    Examples
    --------
    >>> from experiment.weather import weatherStimAllAction
    >>> stim = weatherStimAllAction(2)
    >>> stim(np.array([1, 0, 0, 1]), 0)
    ((1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    >>> stim(np.array([1, 0, 0, 1]), 1)
    ((0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0))
    """

    def weatherStim(observation, action):
        """
        For stimuli processing

        Parameters
        ----------
        observation :  None or list of ints or floats
            The last observation that was recorded
        action : int
            The action chosen by the participant

        Returns
        -------
        activeStimuli : tuple of {0,1}
        stimulus : tuple of floats
            The events processed into a form to be used for updating the expectations
        """

        obsSize = np.size(observation)

        s = np.ones((numActions, obsSize))
        a = np.zeros((numActions, obsSize))

        s[action, :] = observation
        a[action, :] = observation

        stimulus = tuple(s.flatten())
        activeStimuli = tuple(a.flatten())

        return activeStimuli, stimulus

    weatherStim.Name = "weatherStimAllAction"
    weatherStim.Params = {"numActions": numActions}
    return weatherStim


def weatherRewDirect():
    """
    Processes the weather reward for models expecting the reward feedback

    Parameters
    ----------

    Returns
    -------
    weatherRew : function
        The function expects to be passed a tuple containing the reward the
        last action and the last stimuli. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def weatherRew(reward, action, stimuli):

        return reward

    weatherRew.Name = "weatherRewDirect"
    return weatherRew


def weatherRewDiff():
    """
    Processes the weather reward for models expecting reward corrections

    Parameters
    ----------

    Returns
    -------
    weatherRew : function
        The function expects to be passed a tuple containing the reward the
        last action and the last stimuli. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.QLearn
    """

    def weatherRew(reward, action, stimuli):

        if reward == action:
            return 1
        else:
            return 0

    weatherRew.Name = "weatherRewDiff"
    return weatherRew


def weatherRewDualCorrection(epsilon):
    """
    Processes the decks reward for models expecting the reward correction
    from two possible actions.

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward the
        last action and the last stimuli. The reward that is a float and
        action is {0,1}. The function returns a list of length 2.

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    def weatherRew(reward, action, stimuli):
        rewardProc = np.zeros((2, len(stimuli))) + epsilon
        rewardProc[reward, stimuli] = 1
        return np.array(rewardProc)

    weatherRew.Name = "deckRewDualInfo"
    weatherRew.Params = {"epsilon": epsilon}
    return weatherRew
