# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import copy
import warnings

from typing import Tuple, Dict, Any, Optional, Type

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

    def __init__(self,
                 task_name: str,
                 parameters: Optional[Dict[str, Any]] = None,
                 other_options: Optional[Dict[str, Any]] = None):

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
                raise KeyError(f'{p} is not a valid property for model ``{task_name}``. Use {valid_task_args}')

        parameter_combinations = []
        for p in utils.listMergeGen(*list(parameters.values())):
            pc = {k: copy.copy(v) for k, v in zip(parameter_keys, p)}
            parameter_combinations.append(pc)
        self.parameter_combinations = parameter_combinations

        if other_options:
            checked_options = {}
            for k, v in other_options.items():
                if k not in valid_task_args:
                    raise KeyError(f'{k} is not a valid property for task ``{task_name}``. Use {valid_task_args}')
                elif k in parameter_keys:
                    warnings.warn(f'task parameter {k} has been defined twice')
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

    def __next__(self) -> Tuple[Type[Task], Dict[str, Any], Dict[str, Any]]:
        """ 
        Produces the next task instance for the iterator

        Returns
        -------
        instance : tasks.taskTemplate.Task instance
        """

        self.count += 1
        if self.count >= self.count_max:
            raise StopIteration

        varying_properties = copy.copy(self.parameter_combinations[self.count])
        static_properties = copy.copy(self.other_options)


        return self.task_class, varying_properties, static_properties
