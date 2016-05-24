# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Regulatory fit effects in a choice task
                `Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007)`.
                Psychonomic Bulletin & Review, 14(6), 1125–32.
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""
from __future__ import division, print_function

from numpy import array, zeros, exp
from numpy.random import rand
from experimentTemplate import experiment
from experimentPlot import experimentPlot


### decks
deckSets = {"WorthyMaddox": array([[2,  2,  1,  1,  2,  1,  1,  3,  2,  6,  2,  8,  1,  6,  2,  1,  1,
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


class Decks(experiment):
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
    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to
        its documentation for more details.
    """

    Name = "decks"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        T = kwargs.pop('draws', None)
        decks = kwargs.pop("decks", defaultDecks)

        self.plotArgs = kwargs.pop('plotArgs',{})

        if isinstance(decks, str):
            if decks in deckSets:
                self.decks = deckSets[decks]
            else:
                raise "Unknown deck sets"
        else:
            self.decks = decks

        if T:
            self.T = T
        else:
            self.T = len(self.decks[0])

        self.parameters = {"Name": self.Name,
                           "Draws": self.T,
                           "Decks": self.decks}

        # Set draw count
        self.t = -1
        self.drawn = -1  # [-1,-1]
        self.cardValue = None
        self.action = None

        # Recording variables

        self.recCardVal = [-1]*self.T
        self.recAction = [-1]*self.T

        return self

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
        """

        self.action = action

    def feedback(self):
        """
        Responds to the action from the participant
        """

        deckDrawn = self.action
        cardDrawn = self.drawn + 1  # [deckDrawn] + 1

        self.cardValue = self.decks[deckDrawn, cardDrawn]

#        self.drawn[deckDrawn] = cardDrawn
        self.drawn = cardDrawn

        self.storeState()

        return self.cardValue

    def procede(self):
        """
        Updates the experiment after feedback
        """

        pass

    def outputEvolution(self):
        """
        Plots and saves files containing all the relavent data for this
        experiment run
        """

        results = {"Name": self.Name,
                   "Actions": self.recAction,
                   "cardValue": self.recCardVal,
                   "Decks": self.decks,
                   "Draws": self.T,
                   "finalDeckDraws": self.drawn}

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action
        self.recCardVal[self.t] = self.cardValue


def deckStimDirect():
    """
    Processes the decks stimuli for models expecting just the event

    Returns
    -------
    deckStim : function
        The function expects to be passed a tuple containing the observation and the
        last action. The function returns a tuple of ``1`` and the observation.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2, model.decision.binary.decEta
    """

    def deckStim(observation, action):
        return 1, observation

    deckStim.Name = "deckStimDirect"
    return deckStim


def deckRewDirect():
    """
    Processes the decks reward for models expecting just the reward

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
    model.qLearn, model.qLearn2, model.decision.binary.decEta
    """

    def deckRew(reward, action, stimuli):
        return reward

    deckRew.Name = "deckRewDirect"
    return deckRew


def deckRewDirectNormal(maxRewardVal):
    """
    Processes the decks reward for models expecting just the reward, but in range [0,1]

    Parameters
    ----------
    maxRewardVal : int
        The highest value a reward can have

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
    model.opal, model.opals
    """

    def deckRew(reward, action, stimuli):
        return reward / maxRewardVal

    deckRew.Name = "deckRewDirectNormal"
    return deckRew


def deckRewAllInfo(maxRewardVal, minRewardVal, numActions):
    """
    Processes the decks reward for models expecting the reward information
    from all possible actions

    Parameters
    ----------
    maxRewardVal : int
        The highest value a reward can have
    minRewardVal : int
        The lowest value a reward can have
    numActions : int
        The number of actions the participant can perform. Assumes the lowest
        valued action is 0

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The reward that is a float and action is {0,1}. The
        function returns a array of length (maxRewardVal-minRewardVal)*numActions.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.BP, model.EP, model.MS_rev, model.decision.binary.decIntEtaReac

    Examples
    --------
    >>> from experiment.decks import deckRewAllInfo
    >>> rew = deckRewAllInfo(10,1,2)
    >>> rew(6,0)
    array([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> rew(6,1)
    array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    """
    numDiffRewards = maxRewardVal-minRewardVal+1
    respZeros = zeros(numDiffRewards * numActions)

    def deckRew(reward, action, stimuli):
        rewardProc = respZeros.copy() + 1
        rewardProc[numDiffRewards*action + reward - 1] += 1
        return rewardProc.T

    deckRew.Name = "deckRewAllInfo"
    deckRew.Params = {"maxRewardVal": maxRewardVal,
                       "minRewardVal": minRewardVal,
                       "numActions": numActions}
    return deckRew


def deckRewDualInfo(maxRewardVal, epsilon):
    """
    Processes the decks reward for models expecting the reward information
    from two possible actions.

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The reward that is a float and action is {0,1}. The
        function returns a list of length 2.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.BP, model.EP, model.MS, model.MS_rev
    """
    divisor = maxRewardVal + epsilon

    def deckRew(reward, action, stimuli):
        rew = (reward / divisor) * (1 - action) + (1 - (reward / divisor)) * action
        rewardProc = [[rew], [1-rew]]
        return array(rewardProc)

    deckRew.Name = "deckRewDualInfo"
    deckRew.Params = {"epsilon": epsilon}
    return deckRew


def deckRewDualInfoLogistic(maxRewardVal, minRewardVal, epsilon):
    """
    Processes the decks rewards for models expecting the reward information
    from two possible actions.

    Returns
    -------
    deckRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The reward is a float and action is {0,1}. The
        function returns a list of length 2.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.BP, model.EP, model.MS, model.MS_rev
    """
    mid = (maxRewardVal + minRewardVal)/2

    def deckRew(reward, action, stimuli):

        x = exp(epsilon * (reward-mid))

        rew = (x/(1+x))*(1-action) + (1-(x/(1+x)))*action
        rewardProc = [[rew], [1-rew]]
        return array(rewardProc)

    deckRew.Name = "deckRewDualInfoLogistic"
    deckRew.Params = {"midpoint": mid,
                       "epsilon": epsilon}
    return deckRew
