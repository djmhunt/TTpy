# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Probabilistic classification learning in amnesia.
            Knowlton, B. J., Squire, L. R., & Gluck, M. a. (1994).
            Learning & Memory(Cold Spring Harbor, N.Y.), 1(2), 106–120.
            http://doi.org/10.1101/lm.1.2.106
"""
from __future__ import division, print_function

from numpy import array, zeros, exp, size, ones, nan
from numpy.random import rand
from experimentTemplate import experiment
# from plotting import dataVsEvents, varDynamics
from experimentPlot import experimentPlot

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

actualityLists = {"Pickering": [2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2,
                                1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, nan,
                                nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]}
defaultActualities = actualityLists["Pickering"]


class Weather(experiment):
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
    actualities: int, optional
        The actual reality the cues pointed to; the correct response the participant is trying to get correct
    cues: array of floats, optional
        The cues used to guess the actualities
    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to
        its documentation for more details.
    """

    Name = "weather"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        cues = kwargs.pop("cues", defaultCues)
        actualities = kwargs.pop("actualities", defaultActualities)

        self.plotArgs = kwargs.pop('plotArgs', {})

        if isinstance(cues, str):
            if cues in cueSets:
                self.cues = cueSets[cues]
            else:
                raise "Unknown cue sets"
        else:
            self.cues = cues

        if isinstance(actualities, str):
            if actualities in actualityLists:
                self.actualities = actualityLists[actualities]
            else:
                raise "Unknown actualities list"
        else:
            self.actualities = actualities

        self.T = len(self.cues)

        self.parameters = {"Name": self.Name,
                           "Actualities": array(self.actualities),
                           "Cues": array(self.cues)}

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
        Plots and saves files containing all the relavent data for this
        experiment run
        """

        results = self.parameters.copy()

        results["Actions"] = self.recAction

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action


def weatherStimFeedObs():
    """
    Processes the weather stimuli for models expecting just the event

    Returns
    -------
    weatherStim : function
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

    def weatherStim(feedback, action, lastObservation=None):
        if feedback == action:
            return lastObservation
        else:
            return zeros(size(lastObservation))

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
    model.BP, model.EP, model.MS_rev, model.decision.binary.decEtaSet

    Examples
    --------
    >>> from experiment.weather import weatherStimAllAction
    >>> stim = weatherStimAllAction(2)
    >>> stim(0, 0, lastObservation=array([0.5,0.8]))
    array([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> stim(1, 0, lastObservation=array([0.5,0.8]))
    array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    """

    def weatherStim(feedback, action, lastObservation=None):
        """
        For stimuli processing

        Parameters
        ----------
        feedback : int
            The action that was the correct one
        action : int
            The action chosen by the participant
        lastObservation : None or list of ints or floats
            The last observation that was recorded. Generally this is optional, but not here

        Returns
        -------
        stimulus : tuple of floats
            The events processed into a form to be used for updating the expectations
        """

        s = ones((numActions, size(lastObservation)))
        s[feedback, :] = lastObservation
        stimulus = s.flatten()

        return tuple(stimulus)

    weatherStim.Name = "weatherStimAllAction"
    weatherStim.Params = {"numActions": numActions}
    return weatherStim
