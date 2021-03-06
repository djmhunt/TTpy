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
cueSets = {"Test": []}
defaultCues = cueSets["Test"]

actualityLists = {}


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
    numStimuli: int, optional
        If no provided cues, it is the number of distinct stimuli for the generated set of cues. Default ``4``
    correctProb: float in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0.
        The default is ``0.8``
    correctProbs: list or array of floats in [0,1], optional
        If no actualities provided, it is the probability of the correct answer being answer 1 rather than answer 0 for
        each of the different stimuli. Default ``[corrProb, 1-corrProb] * (numStimuli//2) + [corrProb] * (numStimuli%2)``
    rewardlessT: int, optional
        If no actualities provided, it is the number of actualities at the end of the tasks that will have a
        ``None`` reward. Default ``2*numStimuli``
    """

    def __init__(self,
                 cues=None,
                 actualities=None,
                 trialsteps=100,
                 numStimuli=4,
                 correctProb=0.8,
                 correctProbabilities=None,
                 rewardlessT=None):

        super(Probstim, self).__init__()

        if isinstance(cues, str):
            if cues in cueSets:
                self.cues = cueSets[cues]
                self.T = len(self.cues)
                numStimuli = len(self.cues[0])
            else:
                raise Exception("Unknown cue sets")
        elif isinstance(cues, (list, np.ndarray)):
            self.cues = cues
            self.T = len(self.cues)
            numStimuli = len(self.cues[0])
        else:
            self.T = trialsteps
            numStimuli = numStimuli
            stimuli = np.zeros((self.T, numStimuli))
            stimuli[list(range(self.T)), np.random.randint(numStimuli, size=self.T)] = 1
            self.cues = stimuli

        if isinstance(actualities, str):
            if actualities in actualityLists:
                self.actualities = actualityLists[actualities]
                rewardlessT = np.sum(np.isnan(np.array(self.actualities, dtype=npfloat)))
            else:
                raise Exception("Unknown actualities list")
        elif isinstance(actualities, (list, np.ndarray)):
            self.actualities = actualities
            rewardlessT = np.sum(np.isnan(np.array(actualities, dtype=npfloat)))
        else:
            corrProbDefault = [correctProb, 1-correctProb] * (numStimuli // 2) + [correctProb] * (numStimuli % 2)
            if not correctProbabilities:
                correctProbabilities = corrProbDefault
            if not rewardlessT:
                rewardlessT = 2 * numStimuli
            corrChoiceProb = np.sum(self.cues * correctProbabilities, 1)
            correctChoice = list((np.random.rand(self.T) < corrChoiceProb) * 1)
            correctChoice[-rewardlessT:] = [nan] * rewardlessT
            self.actualities = correctChoice

        self.parameters["Actualities"] = np.array(self.actualities)
        self.parameters["Cues"] = np.array(self.cues)
        self.parameters["numtrialsteps"] = self.T
        self.parameters["numRewardless"] = rewardlessT
        self.parameters["number_cues"] = numStimuli

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


class RewardProbStimDualCorrection(Rewards):
    """
    Processes the reward for models expecting the reward correction
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

