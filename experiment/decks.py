# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Regulatory fit effects in a choice task
                `Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007)`.
                Psychonomic Bulletin & Review, 14(6), 1125–32.
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from experiment.experimentTemplate import Experiment

from model.modelTemplate import Stimulus, Rewards

deckSets = {"WorthyMaddox": np.array([[2,  2,  1,  1,  2,  1,  1,  3,  2,  6,  2,  8,  1,  6,  2,  1,  1,
                                    5,  8,  5, 10, 10,  8,  3, 10,  7, 10,  8,  3,  4,  9, 10,  3,  6,
                                    3,  5, 10, 10, 10,  7,  3,  8,  5,  8,  6,  9,  4,  4,  4, 10,  6,
                                    4, 10,  3, 10,  5, 10,  3, 10, 10,  5,  4,  6, 10,  7,  7, 10, 10,
                                    10,  3,  1,  4,  1,  3,  1,  7,  1,  3,  1,  8],
                                   [7, 10,  5, 10,  6,  6, 10, 10, 10,  8,  4,  8, 10,  4,  9, 10,  8,
                                    6, 10, 10, 10,  4,  7, 10,  5, 10,  4, 10, 10,  9,  2,  9,  8, 10,
                                    7,  7,  1, 10,  2,  6,  4,  7,  2,  1,  1,  1,  7, 10,  1,  4,  2,
                                    1,  1,  1,  4,  1,  4,  1,  1,  1,  1,  3,  1,  4,  1,  1,  1,  5,
                                    1,  1,  1,  7,  2,  1,  2,  1,  4,  1,  4,  1]])}
defaultDecks = deckSets["WorthyMaddox"]


class Decks(Experiment):
    """
    Based on the Worthy&Maddox 2007 paper "Regulatory fit effects in a choice task.

    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    draws: int, optional
        Number of cards drawn by the participant
    decks: array of floats, optional
        The decks of cards
    discard: bool
        Defines if you discard the card not chosen or if you keep it.
    """

    def __init__(self, draws=None, decks=defaultDecks, discard=False):

        super(Decks, self).__init__()

        self.discard = discard

        if isinstance(decks, basestring):
            if decks in deckSets:
                self.decks = deckSets[decks]
            else:
                raise Exception("Unknown deck sets")
        else:
            self.decks = decks

        if draws:
            self.T = draws
        else:
            self.T = len(self.decks[0])

        self.parameters["Draws"] = self.T
        self.parameters["Discard"] = self.discard
        self.parameters["Decks"] = self.decks

        # Set draw count
        self.t = -1
        self.cardValue = None
        self.action = None
        if self.discard:
            self.drawn = -1
        else:
            self.drawn = [-1, -1]

        # Recording variables

        self.recCardVal = [-1]*self.T
        self.recAction = [-1]*self.T

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        nextStim = None
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
        Responds to the action from the participant
        """

        deckDrawn = self.action

        if self.discard:
            cardDrawn = self.drawn + 1
            self.drawn = cardDrawn
        else:
            cardDrawn = self.drawn[deckDrawn] + 1
            self.drawn[deckDrawn] = cardDrawn

        self.cardValue = self.decks[deckDrawn, cardDrawn]

        self.storeState()

        return self.cardValue

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
        results["cardValue"] = self.recCardVal
        results["finalDeckDraws"] = self.drawn

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.recAction[self.t] = self.action
        self.recCardVal[self.t] = self.cardValue


class StimulusDecksLinear(Stimulus):

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


class RewardDecksLinear(Rewards):
    """
    Processes the decks reward for models expecting just the reward

    """

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback


class RewardDecksNormalised(Rewards):
    """
    Processes the decks reward for models expecting just the reward, but in range [0,1]

    Parameters
    ----------
    maxReward : int, optional
        The highest value a reward can have. Default ``10``

    See Also
    --------
    model.OpAL
    """
    maxReward = 10

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback / self.maxReward


class RewardDecksPhi(Rewards):
    """
    Processes the decks reward for models expecting just the reward, but in range [0, 1]

    Parameters
    ----------
    phi : float
        The scaling value of the reward
    """

    phi = 1

    def processFeedback(self, feedback, lastAction, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback * self.phi


class RewardDecksAllInfo(Rewards):
    """
    Processes the decks reward for models expecting the reward information
    from all possible actions

    Parameters
    ----------
    maxRewardVal : int
        The highest value a reward can have
    minRewardVal : int
        The lowest value a reward can have
    number_actions : int
        The number of actions the participant can perform. Assumes the lowest
        valued action is 0

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The reward that is a float and action is {0,1}. The
        function returns a array of length (maxRewardVal-minRewardVal)*number_actions.

    Attributes
    ----------
    Name : string
        The identifier of the function

    Examples
    --------
    >>> rew = RewardDecksAllInfo(10, 1, 2)
    >>> rew.processFeedback(6, 0, 1)
    array([1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> rew.processFeedback(6, 1, 1)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1.])
    """
    maxRewardVal = 10
    minRewardVal = 1
    number_actions = 2

    def processFeedback(self, reward, action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        numDiffRewards = self.maxRewardVal - self.minRewardVal + 1
        rewardProc = np.ones(numDiffRewards * self.number_actions)
        rewardProc[numDiffRewards*action + reward - 1] += 1
        return rewardProc.T


class RewardDecksDualInfo(Rewards):
    """
    Processes the decks reward for models expecting the reward information
    from two possible actions.


    """
    maxRewardVal = 10
    epsilon = 1

    def processFeedback(self, reward, action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        divisor = self.maxRewardVal + self.epsilon
        rew = (reward / divisor) * (1 - action) + (1 - (reward / divisor)) * action
        rewardProc = [[rew], [1-rew]]
        return np.array(rewardProc)


class RewardDecksDualInfoLogistic(Rewards):
    """
    Processes the decks rewards for models expecting the reward information
    from two possible actions.


    """
    maxRewardVal = 10
    minRewardVal = 1
    epsilon = 0.3

    def processFeedback(self, reward, action, stimuli):
        """

        Returns
        -------
        modelFeedback:
        """
        mid = (self.maxRewardVal + self.minRewardVal) / 2
        x = np.exp(self.epsilon * (reward-mid))

        rew = (x/(1+x))*(1-action) + (1-(x/(1+x)))*action
        rewardProc = [[rew], [1-rew]]
        return np.array(rewardProc)

