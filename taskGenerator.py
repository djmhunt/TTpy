# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import copy
import collections
import warnings

import utils

from tasks.taskTemplate import Task


class TaskGeneration(object):

    """
    Generates task class instances based on a task and a set of varying parameters

    Parameters
    ----------
    task_name : string
        The name of the file where a tasks.taskTemplate.Task class can be found
    parameters : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across task instances. When a parameter
        contains a list, an instance of the task will be created for every combination of this parameter with all
        the others. Default ``None``
    other_options : dictionary of float, string or binary valued elements
        These contain all the the task options that describe the task being studied but do not vary across
        task instances. Default ``None``
    """

    def __init__(self, task_name, parameters=None, other_options=None):

        self.count = -1

        task_class = utils.find_class(task_name,
                                      class_folder='tasks',
                                      inherited_class=Task,
                                      excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
        valid_task_args = utils.get_class_args(task_class)

        self.task_class = task_class

        if not parameters:
            parameters = {}

        parameter_keys = list(parameters.keys())
        for p in parameter_keys:
            if p not in valid_task_args:
                raise KeyError(
                        '{} is not a valid property for model ``{}``. Use {}'.format(p, task_name,
                                                                                     valid_task_args))

        parameter_combinations = []
        for p in utils.listMergeGen(*list(parameters.values())):
            pc = {k: copy.copy(v) for k, v in zip(parameter_keys, p)}
            parameter_combinations.append(pc)
        self.parameter_combinations = parameter_combinations

        if other_options:
            checked_options = {}
            for k, v in other_options.items():
                if k not in valid_task_args:
                    raise KeyError('{} is not a valid property for task ``{}``. Use {}'.format(k,
                                                                                               task_name,
                                                                                               valid_task_args))
                elif k in parameter_keys:
                    warnings.warn("task parameter {} has been defined twice".format(k))
                else:
                    checked_options[k] = v
            self.other_options = checked_options
        else:
            self.other_options = {}

        if parameter_combinations:
            self.count_max = len(parameter_combinations)
        else:
            self.count_max = 1

    def __iter__(self):
        """ 
        Returns the iterator for the creation of tasks
        """

        self.count = -1

        return self

    def __next__(self):
        """ 
        Produces the next task instance for the iterator

        Returns
        -------
        instance : tasks.taskTemplate.Task instance
        """

        self.count += 1
        if self.count >= self.count_max:
            raise StopIteration

        return self.new_task(self.count)

    def iter_task_ID(self):
        """
        Yields the tasks IDs. To be used with self.new_task(expID) to receive the next tasks instance

        Returns
        -------
        expID : int
            The ID number that refers to the next tasks parameter combination.
        """

        for c in range(self.count_max):
            yield c

    def new_task(self, task_number):
        """
        Produces the next tasks instance

        Parameters
        ----------
        task_number : int
            The number of the tasks instance to be initialised

        Returns
        -------
        instance : tasks.taskTemplate.Task instance
        """

        if task_number >= self.count_max:
            return None

        properties = copy.copy(self.parameter_combinations[task_number])
        properties.update(copy.copy(self.other_options))

        return self.task_class(**properties)
