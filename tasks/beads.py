# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    `Moore, S. C., & Sellen, J. L. (2006)`.
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from tasks.taskTemplate import Task

from model.modelTemplate import Stimulus, Rewards

# Bead Sequences:
beadSequences = {"MooreSellen": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}
defaultBeads = beadSequences["MooreSellen"]


class Beads(Task):
    """Based on the Moore & Sellen Beads task

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    N : int, optional
        Number of beads that could potentially be shown
    beadSequence : list or array of {0,1}, optional
        The sequence of beads to be shown. Bead sequences can also be embedded
        in the code and then referred to by name. The only current one is
        `MooreSellen`, the default sequence.
    """

    def __init__(self, N=None, beadSequence=defaultBeads):

        super(Beads, self).__init__()

        if isinstance(beadSequence, basestring):
            if beadSequence in beadSequences:
                self.beads = beadSequences[beadSequence]
            else:
                raise Exception("Unknown bead sequence")
        else:
            self.beads = beadSequence

        if N:
            self.T = N
        else:
            self.T = len(self.beads)

        self.parameters["N"] = self.T
        self.parameters["beadSequence"] = self.beads

        # Set trialstep count
        self.t = -1

        # Recording variables

        self.recBeads = [-1]*self.T
        self.recAction = [-1]*self.T
        self.firstDecision = 0

    def next(self):
        """ Produces the next bead for the iterator

        Returns
        -------
        bead : {0,1}
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        self.storeState()

        nextStim = self.beads[self.t]
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

        self.recAction[self.t] = action

        if action and not self.firstDecision:
            self.firstDecision = self.t + 1

    def returnTaskState(self):
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standardResultOutput()

        results["Observables"] = np.array(self.recBeads)
        results["Actions"] = self.recAction
        results["FirstDecision"] = self.firstDecision

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        self.recBeads[self.t] = self.beads[self.t]


def generateSequence(numBeads, oneProb, switchProb):
    """
    Designed to generate a sequence of beads with a probability of switching
    jar at any time.

    Parameters
    ----------
    numBeads : int
        The number of beads in the sequence
    oneProb : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar.
    switchProb : float in ``[0,1]``
        The probability that the drawn beads change the jar they are being
        drawn from

    Returns
    -------
    sequence : list of ``{0,1}``
        The generated sequence of beads
    """

    sequence = np.zeros(numBeads)

    probs = np.random.rand(numBeads, 2)
    bead = 1

    for i in range(numBeads):
        if probs[i, 1] < switchProb:
            bead = 1-bead

        if probs[i, 0] < oneProb:
            sequence[i] = bead
        else:
            sequence[i] = 1-bead

    return sequence


class StimulusBeadDirect(Stimulus):
    """
    Processes the beads stimuli for models expecting just the event

    """

    def processStimulus(self, observation):
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
        stimuliActivity : float or list of float

        """
        return 1, observation


class StimulusBeadDualDirect(Stimulus):
    """
    Processes the beads stimuli for models expecting a tuple of ``[event,1-event]``

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
        stimulus = np.array([observation, 1-observation])
        return 1, stimulus


class StimulusBeadDualInfo(Stimulus):
    """
    Processes the beads stimuli for models expecting the reward information
    from two possible actions

    Parameters
    ----------
    oneProb : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar. ``event_info`` is calculated as
        ``oneProb*event + (1-oneProb)*(1-event)``
    """
    oneProb = [0, 1]

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
        stim = self.oneProb*observation + (1-self.oneProb)*(1-observation)
        stimulus = np.array([stim, 1-stim])
        return 1, stimulus


class RewardBeadDirect(Rewards):
    """
    Processes the beads reward for models expecting just the reward
    """

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback


