# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division, print_function, unicode_literals, absolute_import

from numpy import array, zeros, exp, size, ones, nan, ndarray, isnan, sum
from numpy.random import rand, randint
from numpy import float as npfloat
from types import NoneType

from experiment.experimentTemplate import experiment
# from plotting import dataVsEvents, paramDynamics
from experiment.experimentPlot import experimentPlot

# from utils import varyingParams

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


class Probstim(experiment):
    """
    Basic probabilistic

    Many methods are inherited from the experiment.experiment.experiment class.
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
    numStim: int, optional
        If no provided cues, it is the number of distinct stimuli for the generated set of cues. Default ``4``
    correctProb: float in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0.
        The default is ``0.8``
    correctProbs: list or array of floats in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0 for
        each of the different stimuli. Default ``[corrProb, 1-corrProb] * (numStim//2) + [corrProb] * (numStim%2)``
    rewardlessT: int, optional
        If no actualities provided, it is the number of actualities at the end of the experiment that will have a
        ``None`` reward. Default ``2*numStim``
    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to
        its documentation for more details.
    """

    Name = "probStim"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        cues = kwargs.pop("cues", None)
        actualities = kwargs.pop("actualities", None)

        self.plotArgs = kwargs.pop('plotArgs', {})

        if isinstance(cues, basestring):
            if cues in cueSets:
                self.cues = cueSets[cues]
                self.T = len(self.cues)
                numStim = len(self.cues[0])
            else:
                raise "Unknown cue sets"
        elif isinstance(cues, (list, ndarray)):
            self.cues = cues
            self.T = len(self.cues)
            numStim = len(self.cues[0])
        else:
            self.T = kwargs.pop("trialsteps", 100)
            numStim = kwargs.pop("numStim", 4)
            stim = zeros((self.T, numStim))
            stim[range(self.T), randint(numStim, size=self.T)] = 1
            self.cues = stim

        if isinstance(actualities, str):
            if actualities in actualityLists:
                self.actualities = actualityLists[actualities]
                rewardlessT = sum(isnan(array(self.actualities, dtype=npfloat)))
            else:
                raise "Unknown actualities list"
        elif isinstance(actualities, (list, ndarray)):
            self.actualities = actualities
            rewardlessT = sum(isnan(array(actualities, dtype=npfloat)))
        else:
            corrProbBasis = kwargs.pop("correctProb", 0.8)
            corrProbDefault = [corrProbBasis, 1-corrProbBasis] * (numStim//2) + [corrProbBasis] * (numStim%2)
            correctProbs = kwargs.pop("correctProbs", corrProbDefault)
            rewardlessT = kwargs.pop("rewardlessT", 2*numStim)
            corrChoiceProb = sum(self.cues * correctProbs, 1)
            correctChoice = list((rand(self.T) < corrChoiceProb) * 1)
            correctChoice[-rewardlessT:] = [nan] * rewardlessT
            self.actualities = correctChoice

        self.parameters = {"Name": self.Name,
                           "Actualities": array(self.actualities),
                           "Cues": array(self.cues),
                           "numtrialsteps": self.T,
                           "numRewardless": rewardlessT,
                           "numCues": numStim}


        # Set draw count
        self.t = -1
        self.action = None

        # Recording variables
        self.recAction = [-1] * self.T

        return self

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
        action : int
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

    def procede(self):
        """
        Updates the experiment after feedback
        """

        pass

    def outputEvolution(self):
        """
        Plots and saves files containing all the relevant data for this
        experiment run
        """

        results = self.parameters.copy()

        results["Actions"] = self.recAction

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action


def probstimDirect():
    """
    Processes the stimuli for models expecting just the event

    Returns
    -------
    probstim : function
        The function expects to be passed a tuple containing the event and the
        last action. The function returns the event

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2, model.opal, model.opals, model.decision.binary.decEta
    """

    def probstim(observation):

        return observation, observation

    probstim.Name = "probstimDirect"
    return probstim


def probrewDiff():
    """
    Processes the reward for models expecting reward corrections

    Parameters
    ----------

    Returns
    -------
    probrew : function
        The function expects to be passed a tuple containing the reward the
        last action and the last stimuli. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2, model.decision.binary.decEta
    """

    def probrew(reward, action, stimuli):

        if reward == action:
            return 1
        else:
            return 0

    probrew.Name = "probrewDiff"
    return probrew


def probrewDualCorrection(epsilon):
    """
    Processes the reward for models expecting the reward correction
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

    See Also
    --------
    model.BP, model.EP, model.MS, model.MS_rev
    """

    def probrew(reward, action, stimuli):
        rewardProc = zeros((2, len(stimuli))) + epsilon
        rewardProc[reward, stimuli] = 1
        return array(rewardProc)

    probrew.Name = "probstimDualCorrection"
    probrew.Params = {"epsilon": epsilon}
    return probrew
