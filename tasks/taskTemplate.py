# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
import collections
import abc

from typing import Union, Tuple, List, Dict, Any, Optional, NewType, ClassVar

Action = NewType('Action', Union[int, str])


class TaskMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):

        self = super(TaskMeta, cls).__call__(*args, **kwargs)

        self.__task_setup__()

        return self


class Task(metaclass=TaskMeta):
    """The abstract tasks class from which all others inherit

    Many general methods for tasks are found only here
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'next_trialstep') and
                callable(subclass.next_trialstep) and
                hasattr(subclass, 'action_feedback') and
                callable(subclass.action_feedback) or
                NotImplemented)

    @property
    @abc.abstractmethod
    def number_cues(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def valid_actions(self):
        return NotImplementedError

    @abc.abstractmethod
    def next_trialstep(self) -> Tuple[Optional[List[Union[int, float]]], List[Action]]:
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        next_valid_actions : Tuple of ints
            The list of valid actions that the model can respond with.

        Raises
        ------
        StopIteration
        """

        # Since there is nothing to iterate over, just return the final state

        raise NotImplementedError

    @abc.abstractmethod
    def action_feedback(self, action: Action) -> Optional[Union[int, float]]:
        """
        Receives the next action from the participant and responds to the action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : None, int or float
        """

        raise NotImplementedError

    def __iter__(self):
        """
        Returns the iterator for the tasks
        """

        return self

    def __next__(self) -> Tuple[Optional[List[Union[int, float]]], List[Action]]:
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : list of floats or ints
        next_valid_actions : Tuple of ints
            The list of valid actions that the model can respond with

        Raises
        ------
        StopIteration
        """

        stimulus, next_valid_actions = self.next_trialstep()

        return stimulus, next_valid_actions

    def __eq__(self, other: 'Task') -> bool:

        if self.Name == other.Name:
            return True
        else:
            return False

    def __ne__(self, other: 'Task') -> bool:

        if self.Name != other.Name:
            return True
        else:
            return False

    def __hash__(self) -> int:

        return hash(self.Name)

    @classmethod
    def get_name(cls) -> str:
        """
        Returns the name of the class
        """

        return cls.__name__

    def __repr__(self) -> str:

        params = self.parameters.copy()
        name = params.pop('Name')

        label = [f'{name}(']
        label.extend([f'{k}={repr(v)}, ' for k, v in params.items()])
        label.append(')')

        representation = ' '.join(label)

        return representation

    def __task_setup__(self):

        self.Name = self.__class__.__name__

        if hasattr(self, 'valid_actions'):
            if isinstance(self.valid_actions, collections.abc.Iterable):
                self.number_actions: int = len(self.valid_actions)
            else:
                raise TypeError(f'The valid_actions needs to by iterable, found {type(self.valid_actions)}')
        else:
            raise NotImplementedError('The number_actions needs to be specified in the task initialisation ')

        if not hasattr(self, 'number_cues'):
            raise NotImplementedError('The number_cues attribute needs to be specified in the task initialisation')
        elif not isinstance(self.number_cues, int):
            raise TypeError(f'The number_cues attribute should be an int')

        class_variables = vars(self).copy()
        self.record = collections.defaultdict(list)
        self.parameters = {}
        for k, v in class_variables.items():
            if k.startswith('_'):
                self.record[k.strip('_')].append(v)
            else:
                self.parameters[k] = v

    def feedback(self, action: Action) -> Optional[Union[int, float]]:
        """
        Receives the next action from the participant and responds to the action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model

        Returns
        -------
        feedback : None, int or float
        """

        feedback_value = self.action_feedback(action)

        self.store_state()

        return feedback_value

    def return_state(self) -> Dict[str, Any]:
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = dict(self.record)

        results.update(self.parameters)

        return results

    def store_state(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        ignore = ['parameters', 'record']

        current_state = vars(self).copy()
        for k in set(current_state.keys()).difference(ignore).difference(self.parameters.keys()):
            if k.startswith('_'):
                self.record[k.strip('_')].append(current_state[k])
            else:
                self.record[k].append(current_state[k])
