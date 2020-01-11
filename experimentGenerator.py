# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import itertools
import copy
import collections
import warnings

import utils

from experiment.experimentTemplate import Experiment


class ExperimentGen(object):

    """
    Generates experiment class instances based on an experiment and a set of varying parameters

    Parameters
    ----------
    experiment_name : string
        The name of the file where a experiment.experimentTemplate.Experiment class can be found
    parameters : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across experiment instances. When a parameter
        contains a list, an instance of the experiment will be created for every combination of this parameter with all
        the others. Default ``None``
    other_options : dictionary of float, string or binary valued elements
        These contain all the the experiment options that describe the experiment being studied but do not vary across
        experiment instances. Default ``None``
    """

    def __init__(self, experiment_name, parameters=None, other_options=None):

        self.count = -1

        experiment_class = utils.find_class(experiment_name,
                                            class_folder='experiment',
                                            inherited_class=Experiment,
                                            excluded_files=['experimentTemplate', '__init__', 'experimentGenerator'])
        valid_experiment_args = utils.getClassArgs(experiment_class)

        self.experiment_class = experiment_class

        if not parameters:
            parameters = {}

        parameter_keys = parameters.keys()
        for p in parameter_keys:
            if p not in valid_experiment_args:
                raise KeyError(
                        '{} is not a valid property for model ``{}``. Use {}'.format(p, experiment_name,
                                                                                     valid_experiment_args))

        parameter_combinations = []
        for p in utils.listMergeGen(*parameters.values()):
            pc = collections.OrderedDict((k, copy.copy(v)) for k, v in itertools.izip(parameter_keys, p))
            parameter_combinations.append(pc)
        self.parameter_combinations = parameter_combinations

        if other_options:
            checked_options = collections.OrderedDict()
            for k, v in other_options.iteritems():
                if k not in valid_experiment_args:
                    raise KeyError('{} is not a valid property for experiment ``{}``. Use {}'.format(k, experiment_name,
                                                                                                     valid_experiment_args))
                elif k in parameter_keys:
                    warnings.warn("experiment parameter {} has been defined twice".format(k))
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
        Returns the iterator for the creation of experiments
        """

        self.count = -1

        return self

    def next(self):
        """ 
        Produces the next experiment instance for the iterator

        Returns
        -------
        instance : experiment.experimentTemplate.experiment instance
        """

        self.count += 1
        if self.count >= self.count_max:
            raise StopIteration

        return self.new_experiment(self.count)

    def iter_experiment_ID(self):
        """
        Yields the experiment IDs. To be used with self.new_experiment(expID) to receive the next experiment instance

        Returns
        -------
        expID : int
            The ID number that refers to the next experiment parameter combination.
        """

        for c in range(self.count_max):
            yield c

    def new_experiment(self, experiment_number):
        """
        Produces the next experiment instance

        Parameters
        ----------
        experiment_number : int
            The number of the experiment instance to be initialised

        Returns
        -------
        instance : experiment.experiment.experiment instance
        """

        if experiment_number >= self.count_max:
            return None

        properties = copy.copy(self.parameter_combinations[experiment_number])
        properties.update(copy.copy(self.other_options))

        return self.experiment_class(**properties)
