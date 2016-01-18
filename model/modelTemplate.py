# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from numpy import array, size, isnan, ones
from types import NoneType

from modelSetPlot import modelSetPlot
from modelPlot import modelPlot
from model.decision.binary import decSingle
from utils import callableDetailsString


class model(object):

    """
    The model class is a general template for a model. It also contains
    universal methods used by all models.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting
    """

    Name = "model"

    def __init__(self, **kwargs):
        """"""
        self.numCritics = kwargs.pop('numCritics', 2)
        self.prior = kwargs.pop('prior', ones(self.numCritics) * 0.5)

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decSingle(expResponses=tuple(range(1, self.numCritics + 1))))

        self.currAction = 1
        self.decision = None
        self.validActions = None
        self.lastObservation = None
        self.probabilities = array(self.prior)

        self.parameters = {"Name": self.Name,
                           "numCritics": self.numCritics,
                           "prior": self.prior,
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

        # Recorded information

        self.recAction = []
        self.recEvents = []

    def __eq__(self, other):

        if self.Name == other.Name:
            return True
        else:
            return False

    def __ne__(self, other):

        if self.Name != other.Name:
            return True
        else:
            return False

    def __hash__(self):

        return hash(self.Name)

    def action(self):
        """
        Returns the action of the model

        Returns
        -------
        action : integer or None
        """

        self.currAction = self.decision

        self.storeState()

        return self.currAction

    def observe(self, state):
        """
        Receives the latest observation

        Parameters
        ----------
        state : tuple of ({int | float | tuple},{tuple of int | None})
            The stimulus from the experiment followed by the tuple of valid 
            actions. Returns without doing anything if the value of the 
            stimulus is ``None``.

        """

        self._updateObservation(state)

    def _updateObservation(self, state):
        """Processes updates to new actions"""

        events, validActions = state
        lastEvents = self.lastObservation
        self.validActions = validActions

        if type(validActions) is NoneType:
            # If the model is not expected to act,
            # even for a dummy action,
            # so there will be no feedback

            if type(events) is not NoneType:
                self._processEvent(events)
            self.lastObservation = None
        else:
            # If the model is expected to act,
            # store any observations for updating the model after the action feedback
            # and calculate the next action

            # If the last observation still has not been processed,
            # process it
            if type(lastEvents) is not NoneType:
                self._processEvent(lastEvents)

            # Store stimuli, regardless if it is an event or a None type
            self.lastObservation = events

            self._processAction(events)

    def feedback(self, response):
        """
        Receives the reaction to the action

        Parameters
        ----------
        response : float
            The stimulus from the experiment. Returns without doing anything if
            the value of response is `None`.
        """

        self._updateReaction(response)

    def _updateReaction(self, events):
        """Processes updates to feedback"""

        # If there is feedback
        if type(events) is not NoneType:
            self._processEvent(events, lastObservation=self.lastObservation)
            self.lastObservation = None

    def _processEvent(self, events, lastObservation=None):

        if size(events) == 0 or isnan(events):
            event = array([None] * self.numCritics)
            self.recEvents.append(event)
            return

        event = self.stimFunc(events, self.currAction, lastObservation=lastObservation)

        self.recEvents.append(event)

        self._updateModel(event)

    def _updateModel(self, event):
        """
        Parameters
        ----------

        event : list, dict or float
            Whatever suits the model best

        """

        # There is no model here

    def _processAction(self, events):

        self.decision, self.decProbabilities = self.decisionFunc(self.probabilities, self.currAction, stimulus=events, validResponses=self.validActions)

    def outputEvolution(self):
        """
        Returns all the relevant data for this model

        Returns
        -------
        results : dictionary
        """

        results = self.parameters.copy()

        results["Actions"] = array(self.recAction)
        results["Events"] = array(self.recEvents)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)

    def params(self):
        """
        Returns the parameters of the model

        Returns
        -------
        parameters : dictionary
        """

        return self.parameters

    def plot(self):
        """
        Returns a plotting class relevant for this model

        Returns
        -------
        modelPlot : model.modelPlot
        """

        return self.modelPlot

    def plotSet(self):
        """
        Returns a plotting class relevant analysis of sets of results from this
        model

        Returns
        -------
        modelSetPlot : model.modelSetPlot
        """

        return self.modelSetPlot

    class modelPlot(modelPlot):

        """Abstract class for the creation of plots relevant to a model"""

    class modelSetPlot(modelSetPlot):

        """Abstract class for the creation of plots relevant to a set of models"""


def blankStim():
    """
    Default stimulus processor. Does nothing.Returns ([1,0], None)

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return [1, 0], None

    blankStimFunc.Name = "blankStim"
    return blankStimFunc

