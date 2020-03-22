# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import itertools
import collections
import copy
import warnings

import utils

from model.modelTemplate import Model, Stimulus, Rewards


class ModelGen(object):

    """
    Generates model class instances based on a model and a set of varying parameters

    Parameters
    ----------
    model_name : string
        The name of the file where a model.modelTemplate.Model class can be found
    parameters : dictionary containing floats or lists of floats, optional
        Parameters are the options that you are or are likely to change across 
        model instances. When a parameter contains a list, an instance of the 
        model will be created for every combination of this parameter with 
        all the others. Default ``None``
    other_options : dictionary of float, string or binary valued elements, optional
        These contain all the the model options that define the version 
        of the model being studied. Default ``None``
    """

    def __init__(self, model_name, parameters=None, other_options=None):

        self.count = -1

        if parameters is None:
            parameters = {}
        if other_options is None:
            other_options = {}

        model_class = utils.find_class(model_name,
                                       class_folder='model',
                                       inherited_class=Model,
                                       excluded_files=['modelTemplate', '__init__', 'modelGenerator'])
        valid_model_args = utils.get_class_args(model_class)
        valid_args = copy.copy(valid_model_args)

        if 'stimulus_shaper_name' in parameters:
            raise NotImplementedError(
                    "This system has not been created for changing stimulus shapers. Please put it in the ``other_options``")
        stimulus_shaper_name = other_options.pop('stimulus_shaper_name', None)
        if stimulus_shaper_name:
            stimFunc = utils.find_class(stimulus_shaper_name,
                                        class_folder='tasks',
                                        inherited_class=Stimulus,
                                        excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            valid_stimulus_args = utils.get_class_attributes(stimFunc, ignore=['processStimulus'])
            valid_args.extend(valid_stimulus_args)
        else:
            stimFunc = None

        if 'reward_shaper_name' in parameters:
            raise NotImplementedError(
                "This system has not been created for changing reward shapers. Please put it in the ``other_options``")
        reward_shaper_name = other_options.pop('reward_shaper_name', None)
        if reward_shaper_name:
            rewardFunc = utils.find_class(reward_shaper_name,
                                          class_folder='tasks',
                                          inherited_class=Rewards,
                                          excluded_files=['taskTemplate', '__init__', 'taskGenerator'])
            valid_reward_args = utils.get_class_attributes(rewardFunc, ignore=['processFeedback'])
            valid_args.extend(valid_reward_args)
        else:
            rewardFunc = None

        if 'decision_function_name' in parameters:
            raise NotImplementedError(
                    "This system has not been created for changing decision functions. Please put it in the ``other_options``")
        decision_function_name = other_options.pop('decision_function_name', None)
        if decision_function_name:
            decisionFunc = utils.find_function(decision_function_name, 'model/decision', excluded_files=['__init__'])
            valid_decision_args = utils.get_function_args(decisionFunc)
            valid_args.extend(valid_decision_args)
        else:
            decisionFunc = None

        self.model_class = model_class

        if not parameters:
            parameters = {}

        parameter_keys = list(parameters.keys())
        for p in parameter_keys:
            if p not in valid_args and len(model_class.pattern_parameters_match(p)) == 0:
                raise KeyError(
                    '{} is not a valid property for model ``{}``. Those available are {}'.format(p, model_name,
                                                                                                 valid_args))

        parameter_combinations = []
        for p in utils.listMergeGen(*list(parameters.values())):
            pc = collections.OrderedDict((k, copy.copy(v)) for k, v in zip(parameter_keys, p))
            parameter_combinations.append(pc)
        self.parameter_combinations = parameter_combinations

        if other_options:
            checked_options = collections.OrderedDict()
            for k, v in other_options.items():
                if k not in valid_args:
                    raise KeyError('{} is not a valid property for model ``{}``. Those available are {}'.format(k,
                                                                                                                model_name,
                                                                                                                valid_args))
                elif k in parameter_keys:
                    warnings.warn("model parameter {} has been defined twice".format(k))
                else:
                    checked_options[k] = v
            self.other_options = checked_options
            self.other_options['stimulus_shaper_properties'] = valid_stimulus_args
            self.other_options['reward_shaper_properties'] = valid_reward_args
            self.other_options['decision_function_properties'] = valid_decision_args
        else:
            self.other_options = {}

        if stimFunc:
            self.other_options['stimulus_shaper'] = stimFunc
        if rewardFunc:
            self.other_options['reward_shaper'] = rewardFunc
        if decisionFunc:
            self.other_options['decision_function'] = decisionFunc

        if parameter_combinations:
            self.count_max = len(parameter_combinations)
        else:
            self.count_max = 1

    def __iter__(self):
        """ 
        Returns the iterator for the creation of models
        """

        self.count = -1

        return self

    def __next__(self):
        """
        Produces the next item for the iterator
        
        Returns
        -------
        models : list of model.model.model instances
        """

        self.count += 1
        if self.count >= self.count_max:
            raise StopIteration

        properties = copy.copy(self.parameter_combinations[self.count])
        other_options = copy.copy(self.other_options)
        properties.update(other_options)

        return self.model_class(**properties)

    def iter_details(self):
        """ 
        Yields a list containing a model object and parameters to initialise them
        
        Returns
        -------
        model : model.modelTemplate.Model
            The model to be initialised
        parameters : ordered dictionary of floats or bools
            The model instance parameters
        other_options : dictionary of floats, strings and binary values
        """

        for p in self.parameter_combinations:
            yield self.model_class, p, self.other_options
