# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import yaml
import copy
#import fire
import collections
import inspect

import numpy as np

import simulation
import dataFitting
import utils


class MissingScriptSection(Exception):
    pass

class MissingKeyError(Exception):
    pass

class ArgumentError(Exception):
    pass

SCRIPT_PARAMETERS = {'data_folder': ['data', 'path'],
                     'data_format': ['data', 'format'],
                     'data_file_filter': ['data', 'valid_files'],
                     'data_file_terminal_ID': ['data', 'file_terminal_ID'],
                     'data_read_options': ['data', 'read_options'],
                     'data_split_by': ['data', 'split_by'],
                     'data_group_by': ['data', 'group_by'],
                     'data_extra_processing': ['data', 'extra_processing'],
                     'model_name': ['model', 'name'],
                     'model_changing_properties': ['model', 'parameters'],
                     'model_changing_properties_repetition': ['simulation', 'parameter_repetition'],
                     'task_name': ['task', 'name'],
                     'task_changing_properties': ['task', 'parameters'],
                     'participantID': ['data', 'name'],
                     'participant_choices': ['data', 'choices'],
                     'participant_rewards': ['data', 'rewards'],
                     'model_fit_value': ['fitting', 'measures', 'fitting_variable'],
                     'fit_subset': ['fitting', 'measures', 'trial_subset'],
                     'task_stimuli': ['data', 'stimuli'],
                     'participant_action_options': ['data', 'action_options'],
                     'fit_method': ['fitting', 'method'],
                     'fit_measure': ['fitting', 'measures', 'main'],
                     'fit_measure_args': ['fitting', 'measures', 'parameters'],
                     'fit_extra_measures': ['fitting', 'measures', 'extras'],
                     'participant_varying_model_parameters': ['data', 'varying_model_parameters'],
                     'label': ['saving', 'name'],
                     'save_fitting_progress': ['saving', 'save_fitting_progress'],
                     'output_path': ['saving', 'output_path'],
                     'pickle': ['saving', 'pickle'],
                     'boundary_excess_cost_function': ['saving', 'bound_cost_function'],
                     'min_log_level': ['saving', 'min_log_level'],
                     'numpy_error_level': ['saving', 'numpy_error_level'],
                     'fit_float_error_response_value': ['fitting', 'measures', 'float_error_response_value'],
                     'calculate_covariance': ['fitting', 'calculate_covariance']}

SCRIPT_PARAMETER_GROUPS = {'model_constant_properties': ['model'],
                           'task_constant_properties': ['task'],
                           'fit_method_args': ['fitting', 'measures']}


def run_config(script_file, trusted_file=False):
    """
    Takes a .yaml configuration file and runs a simulation or data fitting as described.

    Parameters
    ----------
    config_file : string
        The file name and path of a ``.yaml`` configuration file.
    trusted_file : bool, optional
        If the config file contains executable code this will only be executed if trusted_file is set to ``True``.
        Default is ``False``
    """

    if trusted_file:
        loader = yaml.UnsafeLoader
    else:
        loader = yaml.FullLoader

    with open(script_file) as file_stream:
        script = yaml.load(file_stream, Loader=loader)

    script_sections = script.keys()

    if 'model' not in script_sections:
        raise MissingScriptSection('A ``model`` should be described in the script')

    run_properties = {'config_file': script_file}

    for label, location in SCRIPT_PARAMETERS.iteritems():
        try:
            value = key_find(script, location)
            if label == 'data_extra_processing':
                if value[:4] == 'def ':
                    compiled_value = compile(value, '<string>', 'exec')
                    eval(compiled_value)
                    function_name = compiled_value.co_names[0]
                    function = [v for k, v in copy.copy(locals()).iteritems() if k == function_name][0]
                    args = utils.getFuncArgs(function)
                    if len(args) != 1:
                        raise ArgumentError('The data extra_processing function must have only one argument. Found {}'.format(args))
                    function.func_code_string = value
                    value = function
                else:
                    raise TypeError('data extra_processing must provide a function')
            run_properties[label] = value
        except MissingKeyError:
            continue

    for label, location in SCRIPT_PARAMETER_GROUPS.iteritems():
        try:
            value = key_find(script, location)
            run_properties[label] = value
        except MissingKeyError:
            continue

    if 'simulation' in script_sections:
        if 'task' not in script_sections:
            raise MissingScriptSection('A ``task`` should be described in the script for a simulation')

        simulation.run(**run_properties)

    elif 'fitting' in script_sections:
        if 'data' not in script_sections:
            raise MissingScriptSection('A ``data`` section should be described in the script')
        dataFitting.run(**run_properties)
    else:
        raise MissingScriptSection('A ``simulation`` or ``fitting`` section is necessary for this script to be understood')


def key_find(script, location):
    """
    Find if the nested dictionary key exists, and if it does, return its value

    Parameters
    ----------
    script : dict
        The nested dictionary
    location : list
        The sequence of dictionary keys

    Returns
    -------
    value : object
        The value found at the location in the script
    """
    sub_script = script
    final_loc = location.pop(-1)
    for loc in location:
        if loc in sub_script:
            sub_script_section = sub_script[loc]
            sub_script = sub_script_section
        else:
            raise MissingKeyError

    if final_loc in sub_script:
        return sub_script.pop(final_loc)
    else:
        raise MissingKeyError


def symplify_dtypes(struct):
    """
    Cleans up complex datatypes used in passed in parameters and transforms them to ones recognised by YAML

    Parameters
    ----------
    struct : object
        The object to be cleaned up

    Returns
    -------
    value : object
        The cleaned up value. Can be a NoneType, bool, int, float, string, list, dict or function
    """
    if isinstance(struct, (bool, int, float, basestring)) or struct is None:
        clean_struct = struct
    elif isinstance(struct, (list, tuple)):
        clean_struct = [symplify_dtypes(s) for s in struct]
    elif isinstance(struct, np.ndarray):
        clean_struct = struct.tolist()
    elif isinstance(struct, (dict, collections.OrderedDict, collections.defaultdict)):
        clean_struct = {}
        for key, value in struct.iteritems():
            clean_struct[key] = symplify_dtypes(value)
    elif callable(struct):
        try:
            clean_struct = inspect.getsource(struct)
        except IOError as err:
            clean_struct = struct.func_code_string
    else:
        raise TypeError('Unexpected parameter type {}'.format(struct))
    return clean_struct


def key_set(prepared_dict, location, value):
    """
    Places a value into a nested dictionary at a given location

    Parameters
    ----------
    prepared_dict : default dict defaulting to dict
        The dictionary into which the object will be placed
    location : list of strings
        The sequence of keys for the nested dictionary
    value: object
        The item to be placed at the location in prepared_dict
    """
    cleaned_value = symplify_dtypes(value)
    sub_dict = prepared_dict
    final_location = location.pop()
    for loc in location:
        if loc not in sub_dict:
            sub_dict[loc] = {}
        sub_dict = sub_dict[loc]
    sub_dict[final_location] = cleaned_value


def write_script(file_path, config):
    """
    Takes the parameters passed to simulation.run or dataFitting.run and returns an appropriate YAML script

    Parameters
    ----------
    file_path : string
        The full path and file name of the YAML file to be written
    config : dict
        The configuration to be written to the YAML file
    """
    config_file = config.pop('config_file', None)

    prepared_dict = {}
    for label, location in SCRIPT_PARAMETER_GROUPS.iteritems():
        if label in config:
            if config[label] is None:
                config.pop(label)
            else:
                key_set(prepared_dict, location, config.pop(label))

    for label, value in config.iteritems():
        location = SCRIPT_PARAMETERS[label]
        key_set(prepared_dict, location, value)

    with open(file_path, 'w') as file_stream:
        yaml.dump(dict(prepared_dict), file_stream, indent=4)


def run_script(script_file, trusted_file=False):
    """
    Takes a .yaml configuration file and runs a simulation or data fitting as described.

    This has been replaced by run_config

    Parameters
    ----------
    config_file : string
        The file name and path of a ``.yaml`` configuration file.
    trusted_file : bool, optional
        If the config file contains executable code this will only be executed if trusted_file is set to ``True``.
        Default is ``False``
    """

    if trusted_file:
        loader = yaml.UnsafeLoader
    else:
        loader = yaml.FullLoader

    with open(script_file) as file_stream:
        script = yaml.load(file_stream, Loader=loader)

    script_sections = script.keys()

    # model section
    if 'model' not in script_sections:
        raise MissingScriptSection('A ``model`` should be described in the script')
    else:
        model_info = copy.copy(script['model'])
        model_name = model_info.pop('name', 'QLearn')
        model_changing_properties = model_info.pop('parameters', None)
        model_constant_properties = {}
        model_constant_properties['stimulus_shaper_name'] = model_info.pop('stimulus_shaper')
        model_constant_properties['reward_shaper_name'] = model_info.pop('rewards_shaper')
        model_constant_properties['decision_function_name'] = model_info.pop('decisions_function')
        model_constant_properties.update(model_info)

    # saving section
    saving_info = script.pop('saving', {})
    label = saving_info.pop('name', None)
    output_path = saving_info.pop('output_path', None)
    pickle = saving_info.pop('pickle', False)
    min_log_level = saving_info.pop('min_log_level', 'INFO')
    numpy_error_level = saving_info.pop('numpy_error_level', 'log')
    save_fitting_progress = saving_info.pop('save_fitting_progress', False)
    bound_cost_function = saving_info.pop('bound_cost_function', None)

    if 'simulation' in script_sections:
        simulation_info = script['simulation']
        if 'parameter_repetition' in simulation_info:
            key = model_changing_properties.keys()[0]
            value = model_changing_properties[key]
            extended_value = np.repeat(value, simulation_info['parameter_repetition'])
            model_changing_properties[key] = extended_value

        if 'task' not in script_sections:
            raise MissingScriptSection('A ``task`` should be described in the script')
        else:
            task_info = script['task']
            task_name = task_info.pop('name', 'Basic')
            task_changing_properties = task_info.pop('parameters', None)
            if len(task_info) > 0:
                task_constant_properties = task_info
            else:
                task_constant_properties = None

        simulation.run(task_name=task_name,
                       task_changing_properties=task_changing_properties,
                       task_constant_properties=task_constant_properties,
                       model_name=model_name,
                       model_changing_properties=model_changing_properties,
                       model_constant_properties=model_constant_properties,
                       label=label,
                       config_file=script_file,
                       output_path=output_path,
                       pickle=pickle,
                       min_log_level=min_log_level,
                       numpy_error_level=numpy_error_level)

    elif 'fitting' in script_sections:
        fitting_info = script['fitting']
        fitting_method = fitting_info.pop('method', 'Evolutionary')
        fitting_measures = fitting_info.pop('measures', {})
        fitting_measure = fitting_measures.pop('main', '-loge')
        fitting_measure_args = fitting_measures.pop('parameters', {})
        fitting_extra_measures = fitting_measures.pop('extras', [])
        fitting_variable = fitting_measures.pop('fitting_variable', 'ActionProb')
        fitting_subset = fitting_measures.pop('trial_subset', None)
        fitting_calculate_covariance = fitting_measures.pop('calculate_covariance', False)
        fitting_float_error_response_value = fitting_measures.pop('float_error_response_value', 1 / 1e100)
        fitting_method_args = fitting_measures

        if 'data' not in script_sections:
            raise MissingScriptSection('A ``data`` section should be described in the script')
        else:
            data_info = script['data']
            data_folder = data_info.pop('path', './')
            data_format = data_info.pop('format', 'csv')
            data_file_filter = data_info.pop('valid_files', None)
            data_participantID = data_info.pop('name', 'Name')
            data_choices = data_info.pop('choices', 'Actions')
            data_rewards = data_info.pop('rewards', 'Rewards')
            data_stimuli = data_info.pop('stimuli', None)
            data_action_options = data_info('action_options', None)
            data_varying_model_parameters = data_info('varying_model_parameters', None)
            data_file_terminal_ID = data_info.pop('file_terminal_ID', True)
            data_read_options = data_info.pop('read_options', None)
            data_split_by = data_info.pop('split_by', None)
            data_group_by = data_info.pop('group_by', None)
            data_extra_processing = data_info.pop('extra_processing', None)

        dataFitting.run(data_folder=data_folder,
                        data_format=data_format,
                        data_file_filter=data_file_filter,
                        data_file_terminal_ID=data_file_terminal_ID,
                        data_read_options=data_read_options,
                        data_split_by=data_split_by,
                        data_group_by=data_group_by,
                        data_extra_processing=data_extra_processing,
                        model_name=model_name,
                        model_changing_properties=model_changing_properties,
                        model_constant_properties=model_constant_properties,
                        participantID=data_participantID,
                        participant_choices=data_choices,
                        participant_rewards=data_rewards,
                        model_fit_value=fitting_variable,
                        fit_subset=fitting_subset,
                        task_stimuli=data_stimuli,
                        participant_action_options=data_action_options,
                        fit_method=fitting_method,
                        fit_method_args=fitting_method_args,
                        fit_measure=fitting_measure,
                        fit_measure_args=fitting_measure_args,
                        fit_extra_measures=fitting_extra_measures,
                        participant_varying_model_parameters=data_varying_model_parameters,
                        label=label,
                        save_fitting_progress=save_fitting_progress,
                        config_file=script_file,
                        output_path=output_path,
                        pickle=pickle,
                        boundary_excess_cost_function=bound_cost_function,
                        min_log_level=min_log_level,
                        numpy_error_level=numpy_error_level,
                        fit_float_error_response_value=fitting_float_error_response_value,
                        calculate_covariance=fitting_calculate_covariance
                        )

    else:
        raise MissingScriptSection('A ``simulation`` or ``fitting`` section is necessary for this script to be understood')

if __name__ == '__main__':
    run_config('./runScripts/runScripts_sim.yaml', trusted_file=True)
#    fire.Fire(run_config)