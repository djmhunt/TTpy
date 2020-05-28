# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from typing import Union, Tuple, List, Dict, Any, Optional, NewType

Action = NewType('Action', Union[int, str])


class Task(object):
    """The abstract tasks class from which all others inherit

    Many general methods for tasks are found only here

    Parameters
    ----------


    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    """

    def __init__(self):

        self.Name = self.get_name()

        self.parameters = {"Name": self.Name
                           }

        self.record_actions = []

    def __iter__(self):
        """
        Returns the iterator for the tasks
        """

        return self

    def __next__(self) -> Tuple[List[Union[int, float]], List[Action]]:
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints
            The list of valid actions that the model can respond with. Set to
            ``None``, as they never vary.

        Raises
        ------
        StopIteration
        """

        # Since there is nothing to iterate over, just return the final state

        raise StopIteration

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

        params = self.params()
        name = params.pop('Name')

        label = ["{}(".format(name)]
        label.extend(["{}={}, ".format(k, repr(v)) for k, v in params.items()])
        label.append(")")

        representation = ' '.join(label)

        return representation

    def receive_action(self, action: Action) -> None:
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model
        """

        self.record_actions.append(action)

    def proceed(self) -> None:
        """
        Updates the task before the next trialstep
        """

        pass

    def feedback(self) -> Union[int, float]:
        """
        Responds to the action from the participant

        Returns
        -------
        feedback : None, int or float

        """
        pass

    def return_task_state(self) -> Dict[str, Any]:
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standard_result_output()

        results["Actions"] = self.record_actions

        return results

    def store_state(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        pass

    def standard_result_output(self) -> Dict[str, Any]:

        results = self.parameters.copy()

        return results

    def params(self) -> Dict[str, Any]:
        """
        Returns the parameters of the task as a dictionary

        Returns
        -------
        parameters : dict
            The parameters of the task
        """

        return self.parameters.copy()
