# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import numpy as np

import copy
import re
import collections
import abc
import warnings

from typing import Union, Tuple, List, Any, Optional, ClassVar, Dict, Callable, NewType

from model.decision.discrete import weightProb

import utils

Action = NewType('Action', Union[int, str])


class Modulator(object):

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

        properties = [f"{k} : {str(v).strip('[]()')}" for k, v in self.__dict__.items() if k != "Name"]
        description = self.Name + " with " + ", ".join(properties)

        return description


class Stimulus(Modulator, metaclass=abc.ABCMeta):
    """
    Stimulus processor class. This acts as an interface between an observation and . Does nothing.

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    #@classmethod
    #def __subclasshook__(cls, subclass):
    #    return (hasattr(subclass, 'process_stimulus') and
    #            callable(subclass.process_stimulus) or
    #            NotImplemented)

    #@abc.abstractmethod
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


class Rewards(Modulator, metaclass=abc.ABCMeta):
    """
    This acts as an interface between the feedback from a task and the feedback a model can process

    Attributes
    ----------
    Name : string
        The identifier of the function
    """

    #@classmethod
    #def __subclasshook__(cls, subclass):
    #    return (hasattr(subclass, 'process_feedback') and
    #            callable(subclass.process_feedback) or
    #            NotImplemented)

    #@abc.abstractmethod
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
        feedback : float or np.ndarray

        """
        return feedback


class ModelMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):

        self = cls.__new__(cls)
        super(cls, self).__init__(*args, **kwargs)
        self.__init__(*args, **kwargs)

        self.__model_information_setup__()

        return self


class Model(metaclass=ModelMeta):
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

        self.number_actions = number_actions
        self.number_cues = number_cues

        if action_codes is None:
            action_codes = {k: k for k in range(self.number_actions)}
        else:
            self.number_actions = len(action_codes)
        self.action_code = action_codes
        self.action_code_reversed = {v: k for k, v in action_codes.items()}

        if number_critics is None:
            number_critics = self.number_actions * self.number_cues
        self.number_critics = number_critics

        self.default_non_action = non_action

        if prior is None:
            prior = np.ones(self.number_actions) / self.number_actions
        self.prior = prior

        self._stimuli = np.ones(self.number_cues)
        self._stimuli_filter = np.ones(self.number_cues)

        self._current_action = None
        self._current_action_symbol: Optional[Action] = None
        self._decision = None
        self._valid_actions: Optional[List[Action]] = None
        self.last_observation = None
        self._received_response = None

        self.probabilities = np.array(self.prior)
        #self.ActionProb: float = np.nan
        self._decision_probabilities = {self.action_code_reversed[i]: p for i, p in enumerate(self.prior)}
        self._expected_rewards = np.ones(self.number_actions)
        self._expected_reward = np.array([1])

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
            warnings.warn('No stimulus shaper has been defined. Using the default shaper may lead to errors', UserWarning)
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
            warnings.warn('No reward shaper has been defined. Using the default shaper may lead to errors', UserWarning)
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

        self.simID: Optional[str] = None

    def __model_information_setup__(self):

        class_variables = vars(self).copy()
        self.record = collections.defaultdict(list)
        self.parameters = {}
        for k, v in class_variables.items():
            if k.startswith('_'):
                self.record[k.strip('_')].append(v)
            else:
                self.parameters[k] = v

        self.parameters["stimulus_shaper"] = self.stimulus_shaper.details()
        self.parameters["reward_shaper"] = self.reward_shaper.details()
        self.parameters["decision_function"] = utils.callableDetailsString(self.decision_function)

    def __eq__(self, other: 'Model') -> bool:

        # TODO: Expand this to cover the parameters properly
        if self.__repr__() == other.__repr__():
            return True
        else:
            return False

    def __ne__(self, other: 'Model') -> bool:

        if self.__eq__(other):
            return False
        else:
            return True

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
        self._valid_actions = valid_actions

        # If the last observation still has not been processed,
        # and there has been no feedback, then process it.
        # There may have been an action but feedback was None
        # Since we have another observation it is time to learn from the previous one
        if last_events is not None:
            self.process_event(self._current_action)
            self.store_state()

        self.last_observation = events

        # Find the reward expectations
        self._expected_rewards, self._stimuli, self._stimuli_filter = self.reward_expectation(events)

        expected_probabilities = self.actor_stimulus_probs()

        # If the model is not expected to act, use a dummy action,
        # Otherwise choose an action
        last_action = self._current_action
        if valid_actions is self.default_non_action:
            self._current_action = self.default_non_action
            self._ActionProb = np.nan
        else:
            self._current_action, self._decision_probabilities = self.choose_action(expected_probabilities,
                                                                                    last_action,
                                                                                    events,
                                                                                    valid_actions)
            self.ActionProb = self._decision_probabilities[self._current_action_symbol]

        # Now that the action has been chosen, add any reinforcement of the previous choice in the expectations
        self.last_choice_reinforcement()

        return self._current_action_symbol

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
            self.process_event(self._current_action, response)
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
        self._received_response = response

        # If there were any last reflections to do on the action chosen before processing the new event, now is the last
        # chance to do it
        self.choice_reflection()

        # If there was a reward passed but it was empty, there is nothing to update
        if response is not None and (np.size(response) == 0 or np.isnan(response)):
            return

        # Find the reward expectation
        expected_reward = self._expected_rewards[action]
        self._expected_reward = expected_reward

        # If there was no reward, the the stimulus is the learnt 'reward'
        if response is None:
            response = self._stimuli

        # Find the significance of the discrepancy between the response and the expected response
        delta = self.delta(response, expected_reward, action, self._stimuli)

        # Use that discrepancy to update the model
        self.update_model(delta, action, self._stimuli, self._stimuli_filter)

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
        self._decision = decision
        self._current_action_symbol = decision
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

        self._current_action_symbol = action
        self._current_action = self.action_code[action]

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

    def return_state(self) -> Dict[str, Any]:
        """
        Returns the relevant data expected from a model as well as the parameters for the current model

        Returns
        -------
        results : dictionary
            A dictionary of details about the model run

        """
        results = dict(self.record)
        results.update(self.parameters)
        return results

    def store_state(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """
        ignore = ['parameters', 'record']

        current_state = vars(self).copy()
        for k in set(current_state.keys()).difference(ignore).difference(self.parameters.keys()):
            v = current_state[k]
            if isinstance(v, np.ndarray):
                v_new = v.flatten()
            else:
                v_new = copy.copy(v)

            if k.startswith('_'):
                self.record[k.strip('_')].append(v_new)
            else:
                self.record[k].append(v_new)

    def __repr__(self) -> str:

        params = self.paramaters.copy()
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
    def pattern_parameters_match(cls, *args, patterns: List[str]=[]) -> List[str]:
        """
        Validates if the strings are described by any of the patterns

        Parameters
        ----------
        *args : strings
            The potential parameter names
        patterns : list of strings
            list of regular expression patterns to be used by re.match

        Returns
        -------
        pattern_parameters : list
            The args that match the patterns
        """

        pattern_parameters = []
        for pattern in patterns:
            pattern_parameters.extend(sorted([k for k in args if re.match(pattern, k)]))

        return pattern_parameters

    def add_pattern_parameters(self, kwargs: Dict[str, Any], patterns: List[str]=[]) -> Dict[str, Any]:
        """
        Examine a dictionary of parameters, finds those that match a pattern and add them as properties to the model

        Parameters
        ----------
        kwargs : dict with strings as keys
            Generally the class initialisation kwargs
        patterns : list of strings
            list of regular expression patterns to be used by re.match

        Returns
        -------
        pattern_parameter_dict : dict
            A subset of kwargs that match the patterns
        """

        pattern_parameter_keys = self.pattern_parameters_match(*kwargs.keys(), patterns=patterns)

        pattern_parameter_dict = collections.OrderedDict()
        for k in pattern_parameter_keys:
            v = kwargs.pop(k)
            pattern_parameter_dict[k] = v
            setattr(self, k, v)

        return pattern_parameter_dict
