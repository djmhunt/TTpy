# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
            Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
            Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
            doi:10.1073/pnas.0706111104

"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

import itertools

from experiment.experimentTemplate import Experiment
from model.modelTemplate import Stimulus, Rewards


class ProbSelect(Experiment):
    """
    Probabilistic selection task based on Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
                                        Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
                                        Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
                                        doi:10.1073/pnas.0706111104

    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    rewardProb : float in range [0,1], optional
        The probability that a reward is given for choosing action A. Default
        is 0.7
    actRewardProb : dictionary, optional
        A dictionary of the potential actions that can be taken and the
        probability of a reward.
        Default {0:rewardProb, 1:1-rewardProb, 2:0.5, 3:0.5}
    learnActPairs : list of tuples, optional
        The pairs of actions shown together in the learning phase.
    learningLen : int, optional
        The number of trials in the learning phase. Default is 240
    testLen : int, optional
        The number of trials in the test phase. Default is 60
    rewardSize : float, optional
        The size of reward given if successful. Default 1
    number_actions : int, optional
        The number of actions that can be chosen at any given time, chosen at
        random from actRewardProb. Default 4

    Notes
    -----
    The experiment is broken up into two sections: a learning phase and a
    transfer phase. Participants choose between pairs of four actions: A, B, M1
    and M2. Each provides a reward with a different probability: A:P>0.5,
    B:1-P<0.5, M1=M2=0.5. The transfer phase has all the action pairs but no
    feedback. This class only covers the learning phase, but models are
    expected to be implemented as if there is a transfer phase.

    """

    def __init__(self, rewardProb=0.7,
                 learningActPairs=[(0, 1), (2, 3)],
                 actRewardProb=None,
                 learningLen=240,
                 testLen=60,
                 number_actions=None,
                 rewardSize=1):

        if not actRewardProb:
            actRewardProb = {0: rewardProb,
                             1: 1-rewardProb,
                             2: 0.5,
                             3: 0.5}

        if not number_actions:
            number_actions = len(actRewardProb)

        super(ProbSelect, self).__init__()

        self.parameters["rewardProb"] = rewardProb
        self.parameters["actRewardProb"] = actRewardProb
        self.parameters["learningActPairs"] = learningActPairs
        self.parameters["learningLen"] = learningLen
        self.parameters["testLen"] = testLen
        self.parameters["number_actions"] = number_actions
        self.parameters["rewardSize"] = rewardSize

        self.t = -1
        self.rewardProb = rewardProb
        self.actRewardProb = actRewardProb
        self.learningActPairs = learningActPairs
        self.learningLen = learningLen
        self.rewardSize = rewardSize
        self.T = learningLen + testLen
        self.action = None
        self.rewVal = -1
        self.number_actions = number_actions
        self.choices = actRewardProb.keys()

        self.actT = genActSequence(actRewardProb, learningActPairs, learningLen, testLen)

        # Recording variables
        self.recRewVal = [-1] * self.T
        self.recAction = [-1] * self.T

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of length 2 of ints
            The list of valid actions that the model can respond with.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        nextStim = None
        nextValidActions = self.actT[self.t]

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
        Responds to the action from the participant
        """
        # The probability of success varies depending on if it is choice

        if self.t < self.learningLen:
            actRewProb = self.actRewardProb[self.action]

            if actRewProb >= np.random.rand(1):
                reward = self.rewardSize
            else:
                reward = 0
        else:
            reward = float('Nan')

        self.rewVal = reward

        self.storeState()

        return reward

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

        results["rewVals"] = np.array(self.recRewVal)
        results["Actions"] = np.array(self.recAction)
        results["validAct"] = np.array(self.actT)

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action
        self.recRewVal[self.t] = self.rewVal


def genActSequence(actRewardProb, learningActPairs, learningLen, testLen):

    pairNums = range(len(learningActPairs))
    actPairs = np.array(learningActPairs)

    pairs = np.random.choice(pairNums, size=learningLen, replace=True)
    actSeq = list(actPairs[pairs])

    for t in xrange(testLen):
        pairs = np.random.choice(pairNums, size=2, replace=False)
        elements = np.random.choice([0, 1], size=2, replace=True)

        pair = [actPairs[p, e] for p, e in itertools.izip(pairs, elements)]
        actSeq.append(pair)

    return actSeq


class StimulusProbSelectDirect(Stimulus):
    """
    Processes the selection stimuli for models expecting just the event

    Examples
    --------
    >>> stim = StimulusProbSelectDirect()
    >>> stim.processStimulus(1)
    (1, 1)
    >>> stim.processStimulus(0)
    (1, 1)
    """

    def processStimulus(self, observation):
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
        stimuliActivity : float or list of float

        """
        return 1, 1


class RewardProbSelectDirect(Rewards):
    """
    Processes the probabilistic selection reward for models expecting just the reward

    """

    def processFeedback(self, reward, action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return reward
