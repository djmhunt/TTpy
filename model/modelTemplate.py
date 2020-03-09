# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

import copy
import re
import collections

from model.decision.discrete import weightProb

import utils


class Stimulus(object):
    """
    Stimulus processor class. This acts as an interface between an observation and . Does nothing.

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    # Name = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):

        name = '{}.{}'.format(cls.__module__, cls.__name__)

        return name

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        self.Name = self.get_name()

    def details(self):

        properties = [str(k) + ' : ' + str(v).strip('[]()') for k, v in self.__dict__.iteritems() if k is not "Name"]
        desc = self.Name + " with " + ", ".join(properties)

        return desc

    def processStimulus(self, observation):
        """
        Takes the observation and turns it into a form the model can use

        Parameters
        ----------
        observation :

        Returns
        -------
        stimuliPresent :  int or list of int
        stimuliActivity : float or list of float

        """
        return 1, 1


class Rewards(object):
    """
    This acts as an interface between the feedback from a task and the feedback a model can process

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    # Name = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):

        name = '{}.{}'.format(cls.__module__, cls.__name__)

        return name

    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        self.Name = self.get_name()

    def details(self):

        properties = [str(k) + ' : ' + str(v).strip('[]()') for k, v in self.__dict__.iteritems() if k is not "Name"]
        desc = self.Name + " with " + ", ".join(properties)

        return desc

    def processFeedback(self, feedback, lastAction, stimuli):
        """
        Takes the feedback and turns it into a form to be processed by the model

        Parameters
        ----------
        feedback :
        lastAction :
        stimuli:

        Returns
        -------
        modelFeedback:

        """
        return feedback


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
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    number_cues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    number_critics : integer, optional
        The number of different reaction learning sets.
        Default number_actions*number_cues
    action_codes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0,1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((self.number_actions, self.number_cues)) / self.number_critics)``
    stimulus_shaper_name : string, optional
        The  name of the function that transforms the stimulus into a form the model can
        understand and a string to identify it later. ``stimulus_shaper`` takes priority
    reward_shaper_name : string, optional
        The  name of the function that transforms the reward into a form the model can
        understand. ``rewards_shaper`` takes priority
    decision_function_name : string, optional
        The name of the function that takes the internal values of the model and turns them
        in to a decision. ``decision function`` takes priority
    stimulus_shaper : Stimulus class, optional
        The class that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is Stimulus
    reward_shaper : Rewards class, optional
        The class that transforms the reward into a form the model can
        understand. Default is Rewards
    decision_function : function, optional
        The function that takes the internal values of the model and turns them in to a decision.
        Default is ``weightProb(range(number_actions))``
    stimulus_shaper_properties : list, optional
        The valid parameters of the function. Used to filter the unlisted keyword arguments
        Default is ``None``
    reward_shaper_properties : list, optional
        The valid parameters of the function. Used to filter the unlisted keyword arguments
        Default is ``None``
    decision_function_properties : list, optional
        The valid parameters of the function. Used to filter the unlisted keyword arguments
        Default is ``None``
    """

    #Name = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):
        return cls.__name__

    # TODO:  define and start using non_action
    
    parameter_patterns = []

    def __init__(self, number_actions=2, number_cues=1, number_critics=None,
                 action_codes=None, non_action='None',
                 prior=None,
                 stimulus_shaper=None, stimulus_shaper_name=None, stimulus_shaper_properties=None,
                 reward_shaper=None, reward_shaper_name=None, reward_shaper_properties=None,
                 decision_function=None, decision_function_name=None, decision_function_properties=None,
                 **kwargs):
        """"""
        self.Name = self.get_name()
        
        self.pattern_parameters = self.kwarg_pattern_parameters(kwargs)
        for k, v in self.pattern_parameters.iteritems():
            setattr(self, k, v)

        self.number_actions = number_actions
        self.number_cues = number_cues
        if number_critics is None:
            number_critics = self.number_actions * self.number_cues
        self.number_critics = number_critics

        if action_codes is None:
            action_codes = {k: k for k in xrange(self.number_actions)}
        self.actionCode = action_codes

        self.defaultNonAction = non_action

        if prior is None:
            prior = np.ones(self.number_actions) / self.number_actions
        self.prior = prior

        self.stimuli = np.ones(self.number_cues)
        self.stimuliFilter = np.ones(self.number_cues)

        self.currAction = None
        self.decision = None
        self.validActions = None
        self.lastObservation = None

        self.probabilities = np.array(self.prior)
        self.decProbabilities = np.array(self.prior)
        self.expectedRewards = np.ones(self.number_actions)
        self.expectedReward = np.array([1])

        if stimulus_shaper is not None and issubclass(stimulus_shaper, Stimulus):
            if stimulus_shaper_properties is not None:
                stimulus_shaper_kwargs = {k: v for k, v in kwargs.iteritems() if k in stimulus_shaper_properties}
            else:
                stimulus_shaper_kwargs = kwargs.copy()
            self.stimulus_shaper = stimulus_shaper(**stimulus_shaper_kwargs)
        elif isinstance(stimulus_shaper_name, basestring):
            stimulus_class = utils.find_class(stimulus_shaper_name,
                                              class_folder='tasks',
                                              inherited_class=Stimulus,
                                              excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            stimulus_shaper_kwargs = {k: v for k, v in kwargs.iteritems() if k in utils.getClassArgs(stimulus_class)}
            self.stimulus_shaper = stimulus_class(**stimulus_shaper_kwargs)
        else:
            self.stimulus_shaper = Stimulus()

        if reward_shaper is not None and issubclass(reward_shaper, Rewards):
            if reward_shaper_properties is not None:
                reward_shaper_kwargs = {k: v for k, v in kwargs.iteritems() if k in reward_shaper_properties}
            else:
                reward_shaper_kwargs = kwargs.copy()
            self.reward_shaper = reward_shaper(**reward_shaper_kwargs)
        elif isinstance(reward_shaper_name, basestring):
            reward_class = utils.find_class(reward_shaper_name,
                                            class_folder='tasks',
                                            inherited_class=Rewards,
                                            excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            reward_shaper_kwargs = {k: v for k, v in kwargs.iteritems() if k in utils.getClassArgs(reward_class)}
            self.reward_shaper = reward_class.processFeedback(**reward_shaper_kwargs)
        else:
            self.reward_shaper = Rewards()

        if callable(decision_function):
            if decision_function_properties is not None:
                decision_shaper_kwargs = {k: v for k, v in kwargs.iteritems() if k in decision_function_properties}
            else:
                decision_shaper_kwargs = kwargs.copy()
            self.decision_function = decision_function(**decision_shaper_kwargs)
        elif isinstance(decision_function_name, basestring):
            decision_function = utils.find_function(decision_function_name, 'model/decision')
            decision_function_kwargs = {k: v for k, v in kwargs.iteritems() if k in utils.getFuncArgs(decision_function)}
            self.decision_function = decision_function(**decision_function_kwargs)
        else:
            self.decision_function = weightProb(range(self.number_actions))

        self.parameters = {"Name": self.Name,
                           "number_actions": self.number_actions,
                           "number_cues": self.number_cues,
                           "number_critics": self.number_critics,
                           "prior": copy.copy(self.prior),
                           "non_action": self.defaultNonAction,
                           "actionCode": copy.copy(self.actionCode),
                           "stimulus_shaper": self.stimulus_shaper.details(),
                           "reward_shaper": self.reward_shaper.details(),
                           "decision_function": utils.callableDetailsString(self.decision_function)}
        self.parameters.update(self.pattern_parameters)

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

        # TODO: Expand this to cover the parameters properly
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
            The stimulus from the task followed by the tuple of valid
            actions. Passes the values onto a processing function,
            self._updateObservation``.

        """

        events, validActions = state

        lastEvents = self.lastObservation
        self.validActions = validActions

        # If the last observation still has not been processed,
        # and there has been no feedback, then process it.
        # There may have been an action but feedback was None
        # Since we have another observation it is time to learn from the previous one
        if lastEvents is not None:
            self.processEvent(self.currAction)
            self.storeState()

        self.lastObservation = events

        # Find the reward expectations
        self.expectedRewards, self.stimuli, self.stimuliFilter = self.rewardExpectation(events)

        expectedProbs = self.actorStimulusProbs()

        # If the model is not expected to act, use a dummy action,
        # Otherwise choose an action
        lastAction = self.currAction
        if validActions is self.defaultNonAction:
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
            The response from the task after an action. Returns without doing
            anything if the value of response is `None`.
        """

        # If there is feedback
        if response is not None:
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
            The response from the task after an action. Default ``None``
        """
        self.recReward.append(response)

        # If there were any last reflections to do on the action chosen before processing the new event, now is the last
        # chance to do it
        self.choiceReflection()

        # If there was a reward passed but it was empty, there is nothing to update
        if response is not None and (np.size(response) == 0 or np.isnan(response)):
            return

        # Find the reward expectation
        expectedReward = self.expectedRewards[action]
        self.expectedReward = expectedReward

        # If there was no reward, the the stimulus is the learnt 'reward'
        if response is None:
            response = self.stimuli

        # Find the significance of the discrepancy between the response and the expected response
        delta = self.delta(response, expectedReward, action, self.stimuli)

        # Use that discrepancy to update the model
        self.updateModel(delta, action, self.stimuli, self.stimuliFilter)

    def rewardExpectation(self, stimuli):
        """Calculate the expected reward for each action based on the stimuli

        This contains parts that are task dependent

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

        # stimuli = self.stimulus_shaper_name(lastObservation)
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

        modReward = self.reward_shaper.processFeedback(reward, action, stimuli)

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
        decision, decProbabilities = self.decision_function(probabilities, lastAction, trial_responses=validActions)
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

        actionParamSets = np.reshape(actStimuliParam, (self.number_actions, self.number_cues))
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
        results["ExpectedReward"] = np.array(self.recExpectedReward).flatten()
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

        params = self.params()
        name = params.pop('Name')

        label = ["{}(".format(name)]
        label.extend(["{}={}, ".format(k, repr(v)) for k, v in params.iteritems()])
        label.append(")")

        representation = ' '.join(label)

        return representation

    def setsimID(self, simID):
        """

        Parameters
        ----------
        simID : float

        Returns
        -------

        """

        self.simID = simID
    
    @classmethod
    def pattern_parameters_match(cls, *args):
        """
        Validates if the parameters are described by the model patterns

        Parameters
        ----------
        *args : strings
            The potential parameter names

        Returns
        -------
        pattern_parameters : list
            The args that match the patterns in parameter_patterns
        """

        pattern_parameters = []
        for pattern in cls.parameter_patterns:
            pattern_parameters.extend(sorted([k for k in args if re.match(pattern, k)]))

        return pattern_parameters

    def kwarg_pattern_parameters(self, kwargs):
        """
        Extracts the kwarg parameters that are described by the model patterns

        Parameters
        ----------
        kwargs : dict
            The class initialisation kwargs

        Returns
        -------
        pattern_parameter_dict : dict
            A subset of kwargs that match the patterns in parameter_patterns
        """

        pattern_parameter_keys = self.pattern_parameters_match(*kwargs.keys())

        pattern_parameter_dict = collections.OrderedDict()
        for k in pattern_parameter_keys:
            pattern_parameter_dict[k] = kwargs.pop(k)

        return pattern_parameter_dict
