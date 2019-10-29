# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Note: A simple example of an experiment class with all the necessary components
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from experiment.experimentTemplate import experiment

class Basic(experiment):

    Name = "basic"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        self.nbr_of_trials = kwargs.pop("trials", 100)

        self.parameters = {"Name": self.Name,
                           "Trials": self.nbr_of_trials}

        self.trial = -1  # start at -1 so first call to next will yield trial 0
        self.action = None  # placeholder for what action is taken

        self.action_history = [-1] * self.nbr_of_trials

        return self

    def next(self):
        """
        the experiment class is an iterator [link to iterator documentation]
        this function produces the next stimulus for the experiment iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.trial += 1

        if self.trial == self.nbr_of_trials:
            raise StopIteration

        nextStimulus = 1
        nextValidActions = (0, 1)

        return nextStimulus, nextValidActions

    def receiveAction(self, action):
        """
        Receives the next action from the participant
        """
        self.action = action

    def feedback(self):
        """
        Responds to the action from the participant
        """
        return 1

    def procede(self):
        """
        Updates the experiment after feedback
        """
        pass

    def outputEvolution(self):
        """
        Saves files containing all the relevant data for this experiment run
        """

        results = self.parameters.copy()

        results["participantActions"] = self.action_history.copy()

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.action_history[self.trial] = self.action

def basicStimulusDirect():
    """
    Processes the stimulus cues for models expecting just the event

    Returns
    -------
    basicStimulus : function
        The function returns a tuple of ``1`` and the observation.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2
    """

    def basicStimulus(observation):
        return 1, 1

    basicStimulus.Name = "basicStimulusDirect"
    return basicStimulus


def basicRewardDirect():
    """
    Processes the reward for models expecting just the reward

    Returns
    -------
    basicReward : function
        The function expects to be passed a tuple containing the reward and the
        last action. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2
    """

    def basicReward(reward, action, stimuli):
        return reward

    basicReward.Name = "basicRewardDirect"
    return basicReward
