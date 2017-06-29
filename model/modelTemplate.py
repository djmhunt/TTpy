# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from numpy import array, size, isnan, ones, reshape, sum
from types import NoneType

from model.modelSetPlot import modelSetPlot
from model.modelPlot import modelPlot
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

    Parameters
    ----------
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numCues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numCues
    probActions : bool, optional
        Defines if the probabilities calculated by the model are for each
        action-stimulus pair or for actions. That is, if the stimuli values for
        each action are combined before the probability calculation.
        Default ``True``
    prior : array of floats in ``[0,1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((self.numActions, self.numCues)) / self.numCritics)``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision.
    """

    Name = "modelTemplate"

    def __init__(self, **kwargs):
        """"""
        kwargRemains = self.genStandardParameters(kwargs)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', decSingle(expResponses=tuple(range(1, self.numActions + 1))))

        self.genStandardParameterDetails()

        # Recorded information
        self.genStandardResultsStore()


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

        return self.currAction

    def observe(self, state):
        """
        Receives the latest observation and decides what to do with it

        There are five possible states:
        Observation
        Observation Action
        Observation Action Feedback
        Action Feedback
        Observation Feedback

        Parameters
        ----------
        state : tuple of ({int | float | tuple},{tuple of int | None})
            The stimulus from the experiment followed by the tuple of valid
            actions. Passes the values onto a processing function,
            self._updateObservation``.

        """

        events, validActions = state

        lastEvents = self.lastObservation
        self.validActions = validActions

        # If the last observation still has not been processed,
        # and there has been no feedback, then process it.
        # There may have been an action but feedback was NoneType
        # Since we have another observation it is time to learn from the previous one
        if type(lastEvents) is not NoneType:
            self.processEvent(self.currAction)
            self.storeState()

        self.lastObservation = events

        # Find the reward expectations
        self.expectedRewards, self.stimuli, self.stimuliFilter = self.rewardExpectation(events)

        expectedProbs = self.actorStimulusProbs()

        # If the model is not expected to act, even for a dummy action,
        # then the currentAction can be set to the rest value
        # Otherwise choose an action
        if type(validActions) is NoneType:
            # TODO implement the capacity to set what the defaultNonAction is
            self.currAction = self.defaultNonAction
        else:
            self.currAction, self.decProbabilities = self.chooseAction(expectedProbs, self.currAction, events, validActions)

    def feedback(self, response):
        """
        Receives the reaction to the action and processes it

        Parameters
        ----------
        response : float
            The response from the experiment after an action. Returns without doing
            anything if the value of response is `None`.
        """

        # If there is feedback
        if type(response) is not NoneType:
            self.processEvent(self.currAction, response)
            self.lastObservation = None
            self.storeState()

    def processEvent(self, action=None, response=None):
        """
        Integrates the information from a stimulus, action, response set, regardless
        of which of the three elements are present.

        Parameters
        ----------
        stimuli : {int | float | tuple | None}
            The stimuli received
        action : int, optional
            The chosen action of the model. Default ``None``
        response : float, optional
            The response from the experiment after an action. Default ``None``
        """
        self.recReward.append(response)

        # If there was a reward passed but it was empty, there is nothing to update
        if type(response) is not NoneType and (size(response) == 0 or isnan(response)):
            return

        # Find the reward expectation
        expectedReward = self.expectedRewards[action]
        self.expectedReward = expectedReward

        # If there was no reward, the the stimulus is the learnt 'reward'
        if type(response) is NoneType:
            response = self.stimuli

        # Find the significance of the discrepancy between the response and the expected response
        delta = self.delta(response, expectedReward, action, self.stimuli)

        # Use that discrepency to update the model
        self.updateModel(delta, action, self.stimuliFilter)

    def rewardExpectation(self, observation):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ---------
        stimuli : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        expectedRewards : float
            The expected reward for each action
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        # Calculate expectation by identifying the relevant stimuli for the action
        # First identify the expectations relevant to the action
        # Filter them weighted by the stimuli
        # Calculate the combined value
        # Return the value

        # stimuli = self.stimFunc(response, action, lastObservation=stimuli)
        return 0, 0, 0

    def delta(self, reward, expectation, action, stimuli):
        """
        Calculates the comparison between the reward and the expectation

        Parameters
        ----------
        reward : float
            The reward value
        expectation : float
            The expected reward value
        action : int
            The chosen action
        stimuli : {int | float | tuple | None}
            The stimuli received

        Returns
        -------
        delta
        """

        modReward = self.rewFunc(reward, action, stimuli)

        return 0

    def updateModel(self, delta, action, stimuliFilter):
        """
        Parameters
        ----------

        stimuli : list, dict or float
            Whatever suits the model best
        delta :
            The difference between the reward and the expected reward

        """

        # There is no model here

    def calcProbabilities(self, actionValues):
        """
        Calculate the probabilities associated with the action

        Parameters
        ----------
        actionValues : 1D ndArray of floats

        Returns
        -------
        probArray : 1D ndArray of floats
            The probabilities associated with the actionValues

        """

        # There is no model here

        return 0

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        return 0

    def chooseAction(self, probabilities, lastAction, events, validActions):
        """
        Chooses the next action and returns the associated probabilities

        Parameters
        ----------
        probabilities : list of floats
            The probabilities associated with each combinations
        lastAction : int
            The last chosen action
        events : list of floats
            The stimuli. If probActions is True then this will be unused as the probabilities will already be
        validActions

        Returns
        -------
        newAction : int
            The chosen action
        decProbabilities : list of floats
            The weights for the different actions

        """

        decision, decProbabilities = self.decisionFunc(probabilities, lastAction, stimulus=events, validResponses=validActions)
        self.decision = decision

        return decision, decProbabilities

    def actStimMerge(self, actStimuliParam, stimFilter=1):
        """
        Takes the parameter to be merged by stimulli and filters it by the stimuli values

        Parameters
        ----------
        actStimuliParam : list of floats
            The list of values representing each action stimuli pair, where the stimuli will have their filtered
             values merged together.
        stimFilter : array of floats or a float, optional
            The list of active stimuli with their weightings or one weight for all.
            Default ``1``

        Returns
        -------
        actionParams : list of floats
            The parameter values associated with each action

        """

        actionParamSets = reshape(actStimuliParam, (self.numActions, self.numCues))
        actionParamSets = actionParamSets * stimFilter
        actionParams = sum(actionParamSets, axis=1, keepdims=True)

        return actionParams

    def outputEvolution(self):
        """
        Returns all the relevant data for this model

        Returns
        -------
        results : dictionary
        """

        results = self.standardResultOutput()

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()

    def standardResultOutput(self):
        """
        Returns the relevant data expected from a model as well as the parameters for the current model

        Returns
        -------
        results : dictionary
            A dictionary of details about the

        """

        results = self.parameters.copy()

        results["simID"] = self.simID
        results["Actions"] = array(self.recAction)
        results["Stimuli"] = array(self.recStimuli).T
        results["Rewards"] = array(self.recReward)
        results["Expectations"] = array(self.recExpectations).T
        results["ExpectedRewards"] = array(self.recExpectedReward).T
        results["ValidActions"] = array(self.recValidActions).T
        results["Decisions"] = array(self.recDecision)
        results["UpdatedProbs"] = array(self.recProbabilities).T
        results["ActionProb"] = array(self.recActionProb)
        results["DecisionProbs"] = array(self.recActionProbs)

        return results

    def storeStandardResults(self):
        """
        Updates the store of standard results found across models
        """

        self.recAction.append(self.currAction)
        self.recValidActions.append(self.validActions[:])
        self.recDecision.append(self.decision)
        self.recExpectations.append(self.expectations.flatten())
        self.recExpectedRewards.append(self.expectedRewards.flatten())
        self.recExpectedReward.append(self.expectedReward.flatten())
        self.recStimuli.append(self.stimuli)
        self.recProbabilities.append(self.probabilities.flatten())
        self.recActionProbs.append(self.decProbabilities.copy())
        self.recActionProb.append(self.decProbabilities[self.currAction])

    def genStandardParameters(self, kwargs):
        """Initialises the standard parameters and variables for a model
        """

        self.probActions = kwargs.pop('probActions', True)
        self.numActions = kwargs.pop('numActions', 2)
        self.numCues = kwargs.pop('numCues', 1)
        self.numCritics = kwargs.pop('numCritics', self.numActions * self.numCues)

        if self.probActions:
            defaultPrior = ones(self.numActions) / self.numActions
        else:
            defaultPrior = ones((self.numActions, self.numCues)) / self.numCritics
        self.prior = kwargs.pop('prior', defaultPrior)

        self.stimuli = ones(self.numCues)
        self.stimuliFilter = ones(self.numCues)

        self.currAction = None
        self.decision = None
        self.validActions = None
        self.lastObservation = None

        self.probabilities = array(self.prior)
        self.decProbabilities = array(self.prior)
        self.expectedRewards = ones(self.numActions)
        self.expectedReward = 1

        return kwargs

    def genStandardParameterDetails(self):
        """
        Generates the standard parameters descibing the model as implemented.
        """

        self.parameters = {"Name": self.Name,
                           "numActions": self.numActions,
                           "numCues": self.numCues,
                           "numCritics": self.numCritics,
                           "probActions": self.probActions,
                           "prior": self.prior.copy(),
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

    def genStandardResultsStore(self):
        """Set up the dictionary that stores the standard variables used to track a model
        """

        self.recAction = []
        self.recStimuli = []
        self.recReward = []
        self.recExpectations = []
        self.recExpectedReward = []
        self.recExpectedRewards = []
        self.recValidActions = []
        self.recDecision = []
        self.recProbabilities = []
        self.recActionProbs = []
        self.recActionProb = []
        self.simID = None

    def params(self):
        """
        Returns the parameters of the model

        Returns
        -------
        parameters : dictionary
        """

        return self.parameters

    def setsimID(self, simID):
        """

        Parameters
        ----------
        simID : float

        Returns
        -------

        """

        self.simID = simID

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


def blankRew():
    """
    Default reward processor. Does nothing. Returns reward

    Returns
    -------
    blankRewFunc : function
        The function expects to be passed the reward and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankRewFunc(reward):
        return reward

    blankRewFunc.Name = "blankRew"
    return blankRewFunc

