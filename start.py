# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import yaml
import copy
import fire
import collections
import inspect
import pathlib

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


def generate_run_properties(script: dict, script_file: str) -> dict:
    """
    Takes the dictionary coming from a YAML configuration file and returns a dictionary of parameters for TTpy functions

    Parameters
    ----------
    script : dict
        The contents of the script_file as read in by the YAML interpreter
    script_file : string
        The file name and path of a ``.yaml`` configuration file.

    Returns
    -------
    run_properties : dict
        The dictionary of parameters for a TTpy function
    """
    run_properties = {'config_file_path': script_file}

    for label, location in SCRIPT_PARAMETERS.items():
        try:
            value = key_find(script, location.copy())
            if label == 'data_extra_processing':
                if value[:4] == 'def ':
                    compiled_value = compile(value, '<string>', 'exec')
                    eval(compiled_value)
                    function_name = compiled_value.co_names[0]
                    function = [v for k, v in copy.copy(locals()).items() if k == function_name][0]
                    args = utils.get_function_args(function)
                    if len(args) != 1:
                        raise ArgumentError('The data extra_processing function must have only one argument. Found {}'.format(args))
                    function.func_code_string = value
                    value = function
                else:
                    raise TypeError('data extra_processing must provide a function')
            run_properties[label] = value
        except MissingKeyError:
            continue

    return run_properties


def run_config(script_file, trusted_file=False):
    """
    Takes a .yaml configuration file and runs a simulation or data fitting as described.

    Parameters
    ----------
    script_file : string
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

    script_sections = list(script.keys())

    if 'model' not in script_sections:
        raise MissingScriptSection('A ``model`` should be described in the script')

    run_properties = generate_run_properties(script, script_file)

    for label, location in SCRIPT_PARAMETER_GROUPS.items():
        try:
            value = key_find(script, location.copy())
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


def simplify_dtypes(struct):
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
    if isinstance(struct, (bool, int, float, str)) or struct is None:
        clean_struct = struct
    elif isinstance(struct, (list, tuple)):
        clean_struct = [simplify_dtypes(s) for s in struct]
    elif isinstance(struct, np.ndarray):
        clean_struct = struct.tolist()
    elif isinstance(struct, (dict, collections.OrderedDict, collections.defaultdict)):
        clean_struct = {}
        for key, value in struct.items():
            clean_struct[key] = simplify_dtypes(value)
    elif isinstance(struct, pathlib.PurePath):
        clean_struct = struct.as_posix()
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
    cleaned_value = simplify_dtypes(value)
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

    The parameters passed to simulation.run or dataFitting.run should be those passed to generate_run_properties

    Parameters
    ----------
    file_path : string
        The full path and file name of the YAML file to be written
    config : dict
        The configuration to be written to the YAML file
    """
    config_file = config.pop('config_file_path', None)

    prepared_dict = {}
    for label, location in SCRIPT_PARAMETER_GROUPS.items():
        if label in config:
            if config[label] is None:
                config.pop(label)
            else:
                key_set(prepared_dict, location.copy(), config.pop(label))

    for label, value in config.items():
        location = SCRIPT_PARAMETERS[label].copy()
        key_set(prepared_dict, location, value)

    with open(file_path, 'w') as file_stream:
        yaml.dump(dict(prepared_dict), file_stream, indent=4)


if __name__ == '__main__':
    fire.Fire(run_config)