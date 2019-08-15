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

        self.T = kwargs.pop("trials", 100)

        self.parameters = {"Name": self.Name,
                           "Trials": self.T}

        self.t = -1
        self.action = None

        self.recAction = [-1] * self.T

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

        if self.t ==self.T
            raise StopIteration

        nextStim = 1
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
        return 1

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

        results = self.parameters.copy()

        results["partActions"] = self.recAction.copy()

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later
        """

        self.recAction[self.t] = self.action

def basicStimDirect():
    """
    Processes the stimulus cues for models expecting just the event

    Returns
    -------
    basicStim : function
        The function returns a tuple of ``1`` and the observation.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2
    """

    def basicStim(observation):
        return 1, 1

    basicStim.Name = "basicStimDirect"
    return basicStim


def basicRewDirect():
    """
    Processes the reward for models expecting just the reward

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
    model.qLearn, model.qLearn2
    """

    def basicRew(reward, action, stimuli):
        return reward

    basicRew.Name = "basicRewDirect"
    return basicRew
