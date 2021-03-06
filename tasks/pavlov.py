# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Value and prediction error in medial frontal cortex: integrating the single-unit and systems levels of analysis.
                `Silvetti, M., Seurinck, R., & Verguts, T. (2011)`.
                Frontiers in Human Neuroscience, 5(August), 75.
                doi:10.3389/fnhum.2011.00075
"""
import numpy as np

from tasks.taskTemplate import Task

from model.modelTemplate import Stimulus, Rewards

# TODO: Update pavlov to work with the current framework

class Pavlov(Task):

    """
    Based on the Silvetti et al 2011 paper `"Value and prediction error in
    medial frontal cortex: integrating the single-unit and systems levels of
    analysis."`

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    rewMag : float, optional
        The size of the stimulus. Default 4
    rewProb : array of floats, optional
        The probabilities of each stimulus producing a reward.
        Default [0.85,0.33]
    stimMag : float, optional
        The size of the stimulus. Default 1
    stimDur : int, optional
        The duration, in tens of ms, that the stimulus is produced for. This
        should be longer than rewDur since rewDur is set to end when stimDur
        ends. Default 200
    rewDur : int, optional
        The duration, in tens of ms, that the reward is produced for.
        Default 40
    simDur : int, optional
        The duration, in tens of ms, that each stimulus event is run for.
        Default 300
    stimRepeats : int, optional
        The number of times a stimulus is introduced. Default 72

    """

    def __init__(self, rewMag=4, rewProb=np.array([0.87, 0.33]), stimMag=1, stimDur=20, rewDur=4, simDur=30, stimRepeats=7):

        super(Pavlov, self).__init__()

        self.rewMag = rewMag
        self.rewProb = rewProb
        self.stimMag = stimMag
        self.stimDur = stimDur  # Stimulus duration
        self.rewDur = rewDur  # duration of reward
        self.simLen = simDur  # the length of the simulation
        self.stimRepeats = stimRepeats  # The number of learning runs
#        simLoop = kwargs.pop('simLoopLen',100) #The number of learning loops are run

        self.index = -1

        self.parameters["rewMag"] = self.rewMag
        self.parameters["rewProb"] = self.rewProb
        self.parameters["stimMag"] = self.stimMag
        self.parameters["stimDur"] = self.stimDur
        self.parameters["rewDur"] = self.rewDur
        self.parameters["simLen"] = self.simLen
        self.parameters["stimRepeats"] = self.stimRepeats

        self.cSet, self.stimChoice = self._getStim(self.stimRepeats, self.stimMag)
        self.rewSigSet, self.rewVals = self._getRew(self.stimChoice, self.simLen, self.stimRepeats, self.stimDur, self.rewDur, self.rewMag, self.rewProb)

        self.recActions = []

    def __next__(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        nextStim : tuple of c, rewSig and stimDur, described below
        c : list of floats
            Contains the inputs for each of the stimuli
        rewSig : list of lists of floats
            Each list contains the rewards at each time
        stimDur : int
        nextValidActions : Tuple of ints
            The list of valid actions that the model can respond with. Set to
            ``None``, as there are no actions.


        Raises
        ------
        StopIteration
        """

        self.index += 1

        if self.index == self.stimRepeats:
            raise StopIteration

        nextStim = (self.cSet[self.index], self.stimDur)
        nextValidActions = None

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

        self.storeState()

    def feedback(self):
        """
        Responds to the action from the participant
        """

        rewSig = self.rewSigSet[self.index]

        return rewSig

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

        results["choices"] = self.cSet
        results["stimuli"] = self.stimChoice
        results["rewards"] = self.rewVals
        results["actions"] = self.recActions

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        self.recActions.append(self.action)

    def _getStim(self, stimRepeats, stimMag):
        stimChoice = np.random.choice([0, 1], size=(stimRepeats, 1))
        cSet = np.array([[1-sc[0],sc[0]] for sc in stimChoice])*stimMag

        return cSet, stimChoice

    def _getRew(self, stimChoice, simLen, stimRepeats, stimDur, rewDur, rewMag, rewProb):

        rewVals = (np.random.random((stimRepeats, 1)) < rewProb[stimChoice])*rewMag
        rewSig1 = np.zeros((stimRepeats, stimDur-rewDur))
        rewSig2 = np.ones((stimRepeats, rewDur))*rewVals
        rewSig3 = np.zeros((stimRepeats, simLen-stimDur))
        rewSigSet = np.concatenate((rewSig1, rewSig2, rewSig3), 1)

        return rewSigSet, rewVals

def pavlovStimTemporal():
    """
    Passes the pavlov stimuli to models that cope with stimuli and rewards
    that have a duration.

    Returns
    -------
    pavlovStim : function
        The function expects to be passed an event with three components:
        ``(stim,rew,stimDur)``and an action (unused) and yield a series of
        events ``t,c,r```.
        ``stim`` is the value of the stimulus. It is expected to be a list-like
        object. ``rew`` is a list containing the reward for each trialstep.
        The reward is expected to be a float. ``stimDur`` is the duration of
        the stimulus, an ``int``. This should be less than the length of ``rew``.
        ``c`` the stimulus. ``r`` the reward. ``t`` is the time

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def pavlovStim(event, action, lastObservation=None):

        cStim = event[0]
        rewSig = event[1]
        stimDur = event[2]

        cStimZeros = np.zeros((len(cStim)))

        for t, r in enumerate(rewSig):

            if t < stimDur:
                c = cStim
            else:
                c = cStimZeros

            yield t, c, r

    pavlovStim.Name = "pavlovStimTemporal"

    return pavlovStim

