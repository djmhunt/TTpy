# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

from types import NoneType

from model.decision.discrete import weightProb
from utils import callableDetailsString


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


class Model(object):

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
    actionCodes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
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

    #Name = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_Name(cls):
        return cls.__name__

    def __init__(self, numActions=2, numCues=1, numCritics=None, actionCodes=None, nonAction=None, prior=None, stimFunc=blankStim(),
                 rewFunc=blankRew(), decFunc=None, **kwargs):
        """"""
        self.Name = self.get_Name()

        self.numActions = numActions
        self.numCues = numCues
        if numCritics is None:
            numCritics = self.numActions * self.numCues
        self.numCritics = numCritics

        if actionCodes is None:
            actionCodes = {k: k for k in xrange(self.numActions)}
        self.actionCode = actionCodes

        self.defaultNonAction = nonAction

        if prior is None:
            prior = np.ones(self.numActions) / self.numActions
        self.prior = prior

        self.stimuli = np.ones(self.numCues)
        self.stimuliFilter = np.ones(self.numCues)

        self.currAction = None
        self.decision = None
        self.validActions = None
        self.lastObservation = None

        self.probabilities = np.array(self.prior)
        self.decProbabilities = np.array(self.prior)
        self.expectedRewards = np.ones(self.numActions)
        self.expectedReward = np.array([1])

        self.stimFunc = stimFunc
        self.rewFunc = rewFunc
        if not decFunc:
            decFunc = weightProb(range(self.numActions))
        self.decisionFunc = decFunc
        self.stimFunc = self._eventModifier(self.stimFunc, kwargs)
        self.rewFunc = self._eventModifier(self.rewFunc, kwargs)
        self.decisionFunc = self._eventModifier(self.decisionFunc, kwargs)

        self.parameters = {"Name": self.Name,
                           "numActions": self.numActions,
                           "numCues": self.numCues,
                           "numCritics": self.numCritics,
                           "prior": self.prior.copy(),
                           "nonAction": self.defaultNonAction,
                           "actionCode": self.actionCode.copy(),
                           "stimFunc": callableDetailsString(self.stimFunc),
                           "decFunc": callableDetailsString(self.decisionFunc)}

        # Recorded information
        self.recAction = []
        self.recActionSymbol = []
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

        return self.currActionSymbol

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
        lastAction = self.currAction
        if type(validActions) is NoneType:
            self.currAction = self.defaultNonAction
        else:
            self.currAction, self.decProbabilities = self.chooseAction(expectedProbs, lastAction, events, validActions)

        # Now that the action has been chosen, add any reinforcement of the previous choice in the expectations
        self.lastChoiceReinforcement()

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

        # If there were any last reflections to do on the action chosen before processing the new event, now is the last
        # chance to do it
        self.choiceReflection()

        # If there was a reward passed but it was empty, there is nothing to update
        if type(response) is not NoneType and (np.size(response) == 0 or np.isnan(response)):
            return

        # Find the reward expectation
        expectedReward = self.expectedRewards[action]
        self.expectedReward = expectedReward

        # If there was no reward, the the stimulus is the learnt 'reward'
        if type(response) is NoneType:
            response = self.stimuli

        # Find the significance of the discrepancy between the response and the expected response
        delta = self.delta(response, expectedReward, action, self.stimuli)

        # Use that discrepancy to update the model
        self.updateModel(delta, action, self.stimuli, self.stimuliFilter)

    def rewardExpectation(self, stimuli):
        """Calculate the expected reward for each action based on the stimuli

        This contains parts that are experiment dependent

        Parameters
        ----------
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
        return 0, stimuli, 0

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

    def updateModel(self, delta, action, stimuli, stimuliFilter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this trialstep

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
        validActions : 1D list or array
            The actions permitted during this trialstep

        Returns
        -------
        newAction : int
            The chosen action
        decProbabilities : list of floats
            The weights for the different actions

        """

        if np.isnan(probabilities).any():
            raise ValueError("probabilities contain NaN")
        decision, decProbabilities = self.decisionFunc(probabilities, lastAction, stimulus=events, validResponses=validActions)
        self.decision = decision
        self.currActionSymbol = decision
        decisionCode = self.actionCode[decision]

        return decisionCode, decProbabilities

    def overrideActionChoice(self, action):
        """
        Provides a method for overriding the model action choice. This is used when fitting models to participant actions.

        Parameters
        ----------
        action : int
            Action chosen by external source to same situation
        """

        self.currActionSymbol = action
        self.currAction = self.actionCode[action]

    def choiceReflection(self):
        """
        Allows the model to update its state once an action has been chosen.
        """

    def lastChoiceReinforcement(self):
        """
        Allows the model to update the reward expectation for the previous trialstep given the choice made in this trialstep

        Returns
        -------

        """

    def actStimMerge(self, actStimuliParam, stimFilter=1):
        """
        Takes the parameter to be merged by stimuli and filters it by the stimuli values

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

        actionParamSets = np.reshape(actStimuliParam, (self.numActions, self.numCues))
        actionParamSets = actionParamSets * stimFilter
        actionParams = np.sum(actionParamSets, axis=1, keepdims=True)

        return actionParams

    def returnTaskState(self):
        """
        Returns all the relevant data for this model

        Returns
        -------
        results : dictionary
        """

        results = self.standardResultOutput()

        return results.copy()

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
        results["Actions"] = np.array(self.recAction)
        results["Stimuli"] = np.array(self.recStimuli).T
        results["Rewards"] = np.array(self.recReward)
        results["Expectations"] = np.array(self.recExpectations).T
        results["ExpectedReward"] = np.array(self.recExpectedReward).T
        results["ExpectedRewards"] = np.array(self.recExpectedRewards).T
        results["ValidActions"] = np.array(self.recValidActions).T
        results["Decisions"] = np.array(self.recDecision)
        results["UpdatedProbs"] = np.array(self.recProbabilities).T
        results["ActionProb"] = np.array(self.recActionProb)
        results["DecisionProbs"] = np.array(self.recActionProbs)

        return results

    def storeStandardResults(self):
        """
        Updates the store of standard results found across models
        """

        self.recAction.append(self.currAction)
        self.recActionSymbol.append(self.currActionSymbol)
        self.recValidActions.append(self.validActions[:])
        self.recDecision.append(self.decision)
        self.recExpectations.append(self.expectations.flatten())
        self.recExpectedRewards.append(self.expectedRewards.flatten())
        self.recExpectedReward.append(self.expectedReward.flatten())
        self.recStimuli.append(self.stimuli)
        self.recProbabilities.append(self.probabilities.flatten())
        self.recActionProbs.append(self.decProbabilities.copy())
        self.recActionProb.append(self.decProbabilities[self.currActionSymbol])

    def params(self):
        """
        Returns the parameters of the model

        Returns
        -------
        parameters : dictionary
        """

        return self.parameters.copy()

    def __repr__(self):

        return self.params()

    def setsimID(self, simID):
        """

        Parameters
        ----------
        simID : float

        Returns
        -------

        """

        self.simID = simID

    @staticmethod
    def _eventModifier(eFunc, kwargs):

        try:
            f = eFunc(kwargs)
            if callable(f):
                return f
            else:
                return eFunc
        except TypeError:
            return eFunc


