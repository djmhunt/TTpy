# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import numpy as np

import copy
import re
import collections

from typing import Union, Tuple, List, Any, Optional, ClassVar, Dict, Callable, NewType

from model.decision.discrete import weightProb

import utils

Action = NewType('Action', Union[int, str])


class Stimulus(object):
    """
    Stimulus processor class. This acts as an interface between an observation and . Does nothing.

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    # Name: ClassVar[str] = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):

        name = '{}.{}'.format(cls.__module__, cls.__name__)

        return name

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.Name = self.get_name()

    def details(self) -> str:
        """
        Provides a description of the stimulus class properties, starting with its name

        Returns
        -------
        description: str
            The description
        """

        properties = [str(k) + ' : ' + str(v).strip('[]()') for k, v in self.__dict__.items() if k != "Name"]
        description = self.Name + " with " + ", ".join(properties)

        return description

    def process_stimulus(self, observation: Union[str, float, List[float]]
                         ) -> Tuple[Union[int, List[int]], Union[float, List[float]]]:
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

    # Name: ClassVar[str] = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):

        name = '{}.{}'.format(cls.__module__, cls.__name__)

        return name

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.Name = self.get_name()

    def details(self) -> str:
        """
        Provides a description of the reward class properties, starting with its name

        Returns
        -------
        description: str
            The description
        """

        properties = [str(k) + ' : ' + str(v).strip('[]()') for k, v in self.__dict__.items() if k != "Name"]
        description = self.Name + " with " + ", ".join(properties)

        return description

    def process_feedback(self,
                         feedback: Union[int, float, Action],
                         last_action: Action,
                         stimuli: List[float]
                         ) -> Union[float, np.ndarray]:
        """
        Takes the feedback and turns it into a form to be processed by the model

        Parameters
        ----------
        feedback :
        last_action :
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
        Default is ``weightProb(list(range(number_actions)))``
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

    # Name: ClassVar[str] = __qualname__ ## TODO: start using when moved to Python 3. See https://docs.python.org/3/glossary.html#term-qualified-name

    @classmethod
    def get_name(cls):
        return cls.__name__

    parameter_patterns = []

    def __init__(self, *,
                 number_actions: Optional[int] = 2,
                 number_cues: Optional[int] = 1,
                 number_critics: Optional[int] = None,
                 action_codes: Optional[Dict[Action, int]] = None,
                 non_action: Optional[Union[int, float, str]] = 'None',
                 prior: Optional[Union[list, np.ndarray]] = None,
                 stimulus_shaper: Optional['Stimulus'] = None,  # What kind of callable?
                 stimulus_shaper_name: Optional[str] = None,
                 stimulus_shaper_properties: Dict[str, Any] = None,
                 reward_shaper: Optional['Rewards'] = None,
                 reward_shaper_name: Optional[str] = None,
                 reward_shaper_properties: Dict[str, Any] = None,
                 decision_function: Optional[Callable] = None,
                 decision_function_name: Optional[str] = None,
                 decision_function_properties: Dict[str, Any] = None,
                 **kwargs):
        """"""
        self.Name = self.get_name()
        
        self.pattern_parameters = self.kwarg_pattern_parameters(kwargs)
        for k, v in self.pattern_parameters.items():
            setattr(self, k, v)

        self.number_actions = number_actions
        self.number_cues = number_cues

        if action_codes is None:
            action_codes = {k: k for k in range(self.number_actions)}
        else:
            self.number_actions = len(action_codes)
        self.action_code = action_codes

        if number_critics is None:
            number_critics = self.number_actions * self.number_cues
        self.number_critics = number_critics

        self.default_non_action = non_action

        if prior is None:
            prior = np.ones(self.number_actions) / self.number_actions
        self.prior = prior

        self.stimuli = np.ones(self.number_cues)
        self.stimuli_filter = np.ones(self.number_cues)

        self.current_action = None
        self.curr_action_symbol: Optional[Action] = None
        self.decision = None
        self.valid_actions: Optional[List[Action]] = None
        self.last_observation = None

        self.probabilities = np.array(self.prior)
        self.decision_probabilities = np.array(self.prior)
        self.expected_rewards = np.ones(self.number_actions)
        self.expectedReward = np.array([1])

        if stimulus_shaper is not None and issubclass(stimulus_shaper, Stimulus):
            if stimulus_shaper_properties is not None:
                stimulus_shaper_kwargs = {k: v for k, v in kwargs.items() if k in stimulus_shaper_properties}
            else:
                stimulus_shaper_kwargs = kwargs.copy()
            self.stimulus_shaper = stimulus_shaper(**stimulus_shaper_kwargs)
        elif isinstance(stimulus_shaper_name, str):
            stimulus_class = utils.find_class(stimulus_shaper_name,
                                              class_folder='tasks',
                                              inherited_class=Stimulus,
                                              excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            stimulus_shaper_kwargs = {k: v for k, v in kwargs.items() if k in utils.get_class_args(stimulus_class)}
            self.stimulus_shaper = stimulus_class(**stimulus_shaper_kwargs)
        else:
            self.stimulus_shaper = Stimulus()

        if reward_shaper is not None and issubclass(reward_shaper, Rewards):
            if reward_shaper_properties is not None:
                reward_shaper_kwargs = {k: v for k, v in kwargs.items() if k in reward_shaper_properties}
            else:
                reward_shaper_kwargs = kwargs.copy()
            self.reward_shaper = reward_shaper(**reward_shaper_kwargs)
        elif isinstance(reward_shaper_name, str):
            reward_class = utils.find_class(reward_shaper_name,
                                            class_folder='tasks',
                                            inherited_class=Rewards,
                                            excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            reward_shaper_kwargs = {k: v for k, v in kwargs.items() if k in utils.get_class_args(reward_class)}
            self.reward_shaper = reward_class.process_feedback(**reward_shaper_kwargs)
        else:
            self.reward_shaper = Rewards()

        if callable(decision_function):
            if decision_function_properties is not None:
                decision_shaper_kwargs = {k: v for k, v in kwargs.items() if k in decision_function_properties}
            else:
                decision_shaper_kwargs = kwargs.copy()
            self.decision_function = decision_function(**decision_shaper_kwargs)
        elif isinstance(decision_function_name, str):
            decision_function = utils.find_function(decision_function_name, 'model/decision')
            decision_function_kwargs = {k: v for k, v in kwargs.items()
                                        if k in utils.get_function_args(decision_function)}
            self.decision_function = decision_function(**decision_function_kwargs)
        else:
            self.decision_function = weightProb(list(range(self.number_actions)))

        self.parameters = {"Name": self.Name,
                           "number_actions": self.number_actions,
                           "number_cues": self.number_cues,
                           "number_critics": self.number_critics,
                           "prior": copy.copy(self.prior),
                           "non_action": self.default_non_action,
                           "action_code": copy.copy(self.action_code),
                           "stimulus_shaper": self.stimulus_shaper.details(),
                           "reward_shaper": self.reward_shaper.details(),
                           "decision_function": utils.callableDetailsString(self.decision_function)}
        self.parameters.update(self.pattern_parameters)

        # Recorded information
        self.rec_action = []
        self.rec_action_symbol = []
        self.rec_stimuli = []
        self.rec_reward = []
        self.rec_expectations = []
        self.rec_expected_reward = []
        self.rec_expected_rewards = []
        self.rec_valid_actions = []
        self.rec_decision = []
        self.rec_probabilities = []
        self.rec_action_probabilities = []
        self.rec_action_probability = []
        self.simID: Optional[str] = None

    def __eq__(self, other: 'Model') -> bool:

        # TODO: Expand this to cover the parameters properly
        if self.Name == other.Name:
            return True
        else:
            return False

    def __ne__(self, other: 'Model') -> bool:

        if self.Name != other.Name:
            return True
        else:
            return False

    def __hash__(self) -> int:

        return hash(self.Name)

    def observe(self, state: Tuple[Any, Optional[List[Action]]]) -> Action:
        """
        Receives the latest observation and returns the chosen action

        There are five possible states:
        Observation
        Observation Action
        Observation Action Feedback
        Action Feedback
        Observation Feedback

        Parameters
        ----------
        state : tuple of ({int | float | tuple},{tuple of int or str | None})
            The stimulus from the task followed by the tuple of valid
            actions.

        Returns
        -------
        action : integer or None

        """

        events, valid_actions = state

        last_events = self.last_observation
        self.valid_actions = valid_actions

        # If the last observation still has not been processed,
        # and there has been no feedback, then process it.
        # There may have been an action but feedback was None
        # Since we have another observation it is time to learn from the previous one
        if last_events is not None:
            self.process_event(self.current_action)
            self.store_state()

        self.last_observation = events

        # Find the reward expectations
        self.expected_rewards, self.stimuli, self.stimuli_filter = self.reward_expectation(events)

        expected_probabilities = self.actor_stimulus_probs()

        # If the model is not expected to act, use a dummy action,
        # Otherwise choose an action
        last_action = self.current_action
        if valid_actions is self.default_non_action:
            self.current_action = self.default_non_action
        else:
            self.current_action, self.decision_probabilities = self.choose_action(expected_probabilities,
                                                                                  last_action,
                                                                                  events,
                                                                                  valid_actions)

        # Now that the action has been chosen, add any reinforcement of the previous choice in the expectations
        self.last_choice_reinforcement()

        return self.curr_action_symbol

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
            self.process_event(self.current_action, response)
            self.last_observation = None
            self.store_state()

    def process_event(self, action: Action = None, response: Optional[float] = None):
        """
        Integrates the information from a stimulus, action, response set, regardless
        of which of the three elements are present.

        Parameters
        ----------
        action : int, optional
            The chosen action of the model. Default ``None``
        response : float, optional
            The response from the task after an action. Default ``None``
        """
        self.rec_reward.append(response)

        # If there were any last reflections to do on the action chosen before processing the new event, now is the last
        # chance to do it
        self.choice_reflection()

        # If there was a reward passed but it was empty, there is nothing to update
        if response is not None and (np.size(response) == 0 or np.isnan(response)):
            return

        # Find the reward expectation
        expected_reward = self.expected_rewards[action]
        self.expectedReward = expected_reward

        # If there was no reward, the the stimulus is the learnt 'reward'
        if response is None:
            response = self.stimuli

        # Find the significance of the discrepancy between the response and the expected response
        delta = self.delta(response, expected_reward, action, self.stimuli)

        # Use that discrepancy to update the model
        self.update_model(delta, action, self.stimuli, self.stimuli_filter)

    def reward_expectation(self, observation: Any) -> Tuple[List[float], List[float], List[bool]]:
        """Calculate the expected reward for each action based on the stimuli

        This contains parts that are task dependent

        Parameters
        ----------
        observation :
            The set of stimuli

        Returns
        -------
        expected_rewards : list of float
            The expected reward for each action
        observation : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        # Calculate expectation by identifying the relevant stimuli for the action
        # First identify the expectations relevant to the action
        # Filter them weighted by the stimuli
        # Calculate the combined value
        # Return the value

        active_stimuli, stimuli = self.stimulus_shaper.process_stimulus(observation)
        return [0], stimuli, active_stimuli

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

        # modReward = self.reward_shaper.process_feedback(reward, action, stimuli)

        return 0

    def update_model(self, delta, action, stimuli, stimuli_filter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuli_filter : list of bool
            A list describing if a stimulus cue is present in this trialstep

        """

        # There is no model here
        pass

    def calculate_probabilities(self, action_values: np.ndarray) -> np.ndarray:
        """
        Calculate the probabilities associated with the action

        Parameters
        ----------
        action_values : 1D ndArray of floats

        Returns
        -------
        prob_array : 1D ndArray of floats
            The probabilities associated with the actionValues

        """

        # There is no model here

        return np.zeros(self.number_actions)

    def actor_stimulus_probs(self) -> np.ndarray:
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        return np.zeros(self.number_actions)

    def choose_action(self,
                      probabilities: List[float],
                      lastAction: Optional[Action],
                      events,
                      valid_actions: Optional[List[Action]]
                      ) -> Tuple[int, Dict[Action, float]]:
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
        valid_actions : 1D list or array
            The actions permitted during this trialstep

        Returns
        -------
        newAction : int
            The chosen action
        dec_probabilities : list of floats
            The weights for the different actions

        """

        if np.isnan(probabilities).any():
            raise ValueError("probabilities contain NaN")
        decision, dec_probabilities = self.decision_function(probabilities, lastAction, trial_responses=valid_actions)
        self.decision = decision
        self.curr_action_symbol = decision
        decision_code = self.action_code[decision]

        return decision_code, dec_probabilities

    def override_action_choice(self, action: Action) -> None:
        """
        Provides a method for overriding the model action choice. This is used when fitting models to participant actions.

        Parameters
        ----------
        action : int
            Action chosen by external source to same situation
        """

        self.curr_action_symbol = action
        self.current_action = self.action_code[action]

    def choice_reflection(self) -> None:
        """
        Allows the model to update its state once an action has been chosen.
        """

        pass

    def last_choice_reinforcement(self) -> None:
        """
        Allows the model to update the reward expectation for the previous trialstep given the choice made
        in this trialstep

        Returns
        -------

        """
        pass

    def action_cue_merge(self, act_cue_param, cue_filter=1):
        """
        Takes the parameter to be merged by stimuli and filters it by the stimuli values

        Parameters
        ----------
        act_cue_param : list of floats
            The list of values representing each action stimuli pair, where the stimuli will have their filtered
             values merged together.
        cue_filter : array of floats or a float, optional
            The list of active stimulus cues with their weightings or one weight for all.
            Default ``1``

        Returns
        -------
        action_params : list of floats
            The parameter values associated with each action

        """

        action_param_sets = np.reshape(act_cue_param, (self.number_actions, self.number_cues))
        action_param_sets = action_param_sets * cue_filter
        action_params = np.sum(action_param_sets, axis=1, keepdims=True)

        return action_params

    def return_task_state(self):
        """
        Returns all the relevant data for this model

        Returns
        -------
        results : dictionary
        """

        results = self.standard_results_output()

        return results.copy()

    def store_state(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.store_standard_results()

    def standard_results_output(self) -> Dict[str, Any]:
        """
        Returns the relevant data expected from a model as well as the parameters for the current model

        Returns
        -------
        results : dictionary
            A dictionary of details about the model run

        """

        results = self.parameters.copy()

        results["simID"] = self.simID
        results["Actions"] = np.array(self.rec_action)
        results["Stimuli"] = np.array(self.rec_stimuli).T
        results["Rewards"] = np.array(self.rec_reward)
        results["Expectations"] = np.array(self.rec_expectations).T
        results["ExpectedReward"] = np.array(self.rec_expected_reward).flatten()
        results["ExpectedRewards"] = np.array(self.rec_expected_rewards).T
        results["ValidActions"] = np.array(self.rec_valid_actions).T
        results["Decisions"] = np.array(self.rec_decision)
        results["UpdatedProbs"] = np.array(self.rec_probabilities).T
        results["ActionProb"] = np.array(self.rec_action_probability)
        results["DecisionProbs"] = np.array(self.rec_action_probabilities)

        return results

    def store_standard_results(self) -> None:
        """
        Updates the store of standard results found across models
        """

        self.rec_action.append(self.current_action)
        self.rec_action_symbol.append(self.curr_action_symbol)
        self.rec_valid_actions.append(self.valid_actions[:])
        self.rec_decision.append(self.decision)
        self.rec_expectations.append(self.expectations.flatten())
        self.rec_expected_rewards.append(self.expected_rewards.flatten())
        self.rec_expected_reward.append(self.expectedReward.flatten())
        self.rec_stimuli.append(self.stimuli)
        self.rec_probabilities.append(self.probabilities.flatten())
        self.rec_action_probabilities.append(self.decision_probabilities.copy())
        self.rec_action_probability.append(self.decision_probabilities[self.curr_action_symbol])

    def params(self) -> Dict[str, Any]:
        """
        Returns the parameters of the model

        Returns
        -------
        parameters : dictionary
        """

        return self.parameters.copy()

    def __repr__(self) -> str:

        params = self.params()
        name = params.pop('Name')

        label = ["{}(".format(name)]
        label.extend(["{}={}, ".format(k, repr(v)) for k, v in params.items()])
        label.append(")")

        representation = ' '.join(label)

        return representation

    def set_simID(self, simID: str) -> None:
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

    def kwarg_pattern_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
