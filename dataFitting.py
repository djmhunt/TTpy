# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import collections
import copy
#import fire

import pandas as pd

from typing import Any

import outputting
import utils
import data

from fitAlgs.fitSims import FitSim
from fitAlgs.fitAlg import FitAlg
from modelGenerator import ModelGen


class LengthError(Exception):
    pass


class OrderError(Exception):
    pass


def run(data_folder='./',
        data_format='csv',
        data_file_filter=None,
        data_file_terminal_ID=True,
        data_read_options=None,
        data_split_by=None,
        data_group_by=None,
        data_extra_processing=None,
        model_name='QLearn',
        model_changing_properties=None,
        model_constant_properties=None,
        participantID="Name",
        participant_choices='Actions',
        participant_rewards='Rewards',
        model_fit_value='ActionProb',
        fit_subset=None,
        task_stimuli=None,
        participant_action_options=None,
        fit_method='Evolutionary',
        fit_method_args=None,
        fit_measure='-loge',
        fit_measure_args=None,
        fit_extra_measures=None,
        participant_varying_model_parameters=None,
        label=None,
        save_fitting_progress=False,
        config_file=None,
        output_path=None,
        pickle=False,
        boundary_excess_cost_function=None,
        min_log_level='INFO',
        numpy_error_level="log",
        fit_float_error_response_value=1 / 1e100,
        calculate_covariance=False
        ):
    """
    A framework for fitting models to data for tasks, along with
    recording the data associated with the fits.

    Parameters
    ----------
    data_folder : string or list of strings, optional
        The folder where the data can be found. Default is the current folder.
    data_format : string, optional
        The file type of the data, from ``mat``, ``csv``, ``xlsx`` and ``pkl``. Default is ``csv``
    data_file_filter : callable, string, list of strings or None, optional
        A function to process the file names or a list of possible prefixes as strings or a single string.
        Default ``None``, no file names removed
    data_file_terminal_ID : bool, optional
        Is there an ID number at the end of the filename? If not then a more general search will be performed.
        Default ``True``
    data_read_options : dict, optional
        The keyword arguments for the data importing method chosen
    data_split_by : string or list of strings, optional
        If multiple participant datasets are in one file sheet, this specifies the column or columns that can
        distinguish and identify the rows for each participant. Default ``None``
    data_group_by : list of strings, optional
        A list of parts of filenames that are repeated across participants, identifying all the files that should
        be grouped together to form one participants data. The rest of the filename is assumed to identify the
        participant. Default is ``None``
    data_extra_processing : callable, optional
        A function that modifies the dictionary of data read for each participant in such that it is appropriate
        for fitting. Default is ``None``
    model_name : string, optional
        The name of the file where a model.modelTemplate.Model class can be found. Default ``QLearn``
    model_changing_properties : dictionary with values of tuple of two floats, optional
        Parameters are the options that you allow to vary across model fits. Each model parameter is specified as a
        dict key. The value is a tuple containing the upper and lower search bounds, e.g. ``alpha`` has the bounds
        (0, 1). Default ``None``
    model_constant_properties : dictionary of float, string or binary valued elements, optional
        These contain all the the model options that define the version
        of the model being studied. Default ``None``
    participantID : basestring, optional
        The key (label) used to identify each participant. Default ``Name``
    participant_choices : string, optional
        The participant data key of their action choices. Default ``'Actions'``
    participant_rewards : string, optional
        The participant data key of the participant reward data. Default ``'Rewards'``
    model_fit_value : string, optional
        The key to be compared in the model data. Default ``'ActionProb'``
    fit_subset : ``float('Nan')``, ``None``, ``"rewarded"``, ``"unrewarded"``, ``"all"`` or list of int, optional
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing
        - ``"all"`` for fitting all trials
        - ``float('Nan')`` or ``"unrewarded"`` for all those trials whose feedback was ``float('Nan')``
        - ``"rewarded"`` for those who had feedback that was not ``float('Nan')``
        Default ``None``, which means all trials will be used.
    task_stimuli : list of strings or None, optional
        The keys containing the observational parameters seen by the
        participant before taking a decision on an action. Default ``None``
    participant_action_options : string or list of strings or None or one element list with a list, optional
        If a string or list of strings these are treated as dict keys where the valid actions for each trial can
        be found. If None then all trials will use all available actions. If the list contains one list then it will
        be treated as a list of valid actions for each trialstep. Default ``'None'``
    fit_method : string, optional
        The fitting method to be used. The names accepted are those of the modules in the folder fitAlgs containing a
        FitAlg class. Default ``'evolutionary'``
    fit_method_args : dict, optional
        A dictionary of arguments specific to the fitting method. Default ``None``
    fit_measure : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fitting guide. Default ``-loge``
    fit_measure_args : dict, optional
        The parameters used to initialise fitMeasure and extraFitMeasures. Default ``None``
    fit_extra_measures : list of strings, optional
        List of fit measures not used to fit the model, but to provide more information. Any arguments needed for these
        measures should be placed in fitMeasureArgs. Default ``None``
    participant_varying_model_parameters : dict of string, optional
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string. Default ``{}``
    label : string, optional
        The label for the data fitting. Default ``None`` will mean no data is saved to files.
    save_fitting_progress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``
    config_file : string, optional
        The file name and path of a ``.yaml`` configuration file. Overrides all other parameters if found.
        Default ``None``
    output_path : string, optional
        The path that will be used for the run output. Default ``None``
    pickle : bool, optional
        If true the data for each model, and participant is recorded.
        Default is ``False``
    boundary_excess_cost_function : basestring or callable returning a function, optional
        The function is used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound()``
    min_log_level : basestring, optional
        Defines the level of the log from (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``). Default ``INFO``
    numpy_error_level : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr
    fit_float_error_response_value : float, optional
        If a floating point error occurs when running a fit the fitter function
        will return a value for each element of fpRespVal. Default is ``1/1e100`
    calculate_covariance : bool, optional
        Is the covariance calculated. Default ``False``

    See Also
    --------
    modelGenerator : The model factory
    outputting : The outputting functions
    fitAlgs.fitAlg.FitAlg : General class for a method of fitting data
    fitAlgs.fitSims.fitSim : General class for a method of simulating the fitting of data
    data.Data : Data import class
    """
    config = copy.deepcopy(locals())

    if participant_varying_model_parameters is None:
        model_changing_variables = {}
    else:
        model_changing_variables = participant_varying_model_parameters

    # TODO : Validate model_changing_properties with the data and the model
    participants = data.Data.load_data(file_type=data_format,
                                       folders=data_folder,
                                       file_name_filter=data_file_filter,
                                       terminal_ID=data_file_terminal_ID,
                                       split_by=data_split_by,
                                       participantID=participantID,
                                       choices=participant_choices,
                                       feedbacks=participant_rewards,
                                       stimuli=task_stimuli,
                                       action_options=participant_action_options,
                                       group_by=data_group_by,
                                       extra_processing=data_extra_processing,
                                       data_read_options=data_read_options)

    if model_changing_properties:
        model_parameters = collections.OrderedDict()
        for key, value in model_changing_properties.iteritems():
            if len(value) == 2:
                v1, v2 = value
                if v2 < v1:
                    raise OrderError('The bounds specified for model parameter ``{}`` must have the lower bound first'.format(key))
                else:
                    model_parameters[key] = (v1 + v2) / 2
            else:
                raise LengthError("The parameter values for the ``model_changing_properties`` must be presented as a list of the maximum and minimum values. Review those of ``{}``".format(key))
    else:
        model_parameters = model_changing_properties

    models = ModelGen(model_name=model_name,
                      parameters=model_parameters,
                      other_options=model_constant_properties)

    model_simulator = FitSim(participant_choice_property=participants.choices,
                             participant_reward_property=participants.feedbacks,
                             model_fitting_variable=model_fit_value,
                             fit_subset=fit_subset,
                             task_stimuli_property=participants.stimuli,
                             action_options_property=participants.action_options,
                             float_error_response_value=fit_float_error_response_value
                             )

    fitting_method = utils.find_class(fit_method,
                                      class_folder='fitAlgs',
                                      inherited_class=FitAlg,
                                      excluded_files=['boundFunc', 'qualityFunc', 'fitSims'])

    if fit_method_args is None:
        fit_method_args = {}

    fitter = fitting_method(fit_sim=model_simulator,
                            fit_measure=fit_measure,
                            extra_fit_measures=fit_extra_measures,
                            fit_measure_args=fit_measure_args,
                            bounds=model_changing_properties,
                            boundary_excess_cost=boundary_excess_cost_function,
                            calculate_covariance=calculate_covariance,
                            **fit_method_args)

    with outputting.Saving(config=config) as file_name_generator:

        logger = logging.getLogger('Fitting')

        log_fitting_parameters(fitter.info())

        message = 'Beginning the data fitting'
        logger.info(message)

        model_ID = 0
        # Initialise the stores of information
        participant_fits = collections.defaultdict(list)  # type: collections.defaultdict[Any, list]

        for model, model_parameter_variables, model_static_args in models.iter_details():

            for v in model_changing_variables.itervalues():
                model_static_args[v] = "<Varies for each participant>"

            log_model_fitting_parameters(model, model_parameter_variables, model_static_args)

            participantID = participants.participantID
            for participant in participants:

                participant_name = participant[participantID]
                if isinstance(participant_name, (list, tuple)):
                    participant_name = participant_name[0]

                for k, v in model_changing_variables.iteritems():
                    model_static_args[v] = participant[k]

                # Find the best model values from those proposed
                message = "Beginning participant fit for participant {}".format(participant_name)
                logger.info(message)

                model_fitted, fit_quality, fitting_data = fitter.participant(model,
                                                                             model_parameter_variables,
                                                                             model_static_args,
                                                                             participant)

                message = "Participant fitted"
                logger.debug(message)

                log_model_fitted_parameters(model_parameter_variables, model_fitted.params(), fit_quality, participant_name)

                participant_fits = record_participant_fit(participant, participant_name, model_fitted.returnTaskState(),
                                                          str(model_ID), fitting_data, model_changing_variables,
                                                          participant_fits, fileNameGen=file_name_generator,
                                                          pickleData=pickle, saveFittingProgress=save_fitting_progress)

            model_ID += 1

        fit_record(participant_fits, file_name_generator)


# %% Data record functions  
def record_participant_fit(participant, part_name, model_data, model_name, fitting_data, partModelVars, participantFits,
                           fileNameGen=None, pickleData=False, saveFittingProgress=False, expData=None):
    """
    Record the data relevant to the participant fitting

    Parameters
    ----------
    participant : dict
        The participant data
    part_name : int or string
        The identifier for each participant
    model_data : dict
        The data from the model
    model_name : basestring
        The label given to the model
    fitting_data : dict
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters
    partModelVars : dict of string
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string.
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the participant fits
    fileNameGen : function or None
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string. Default ``None``
    pickleData : bool, optional
        If true the data for each model, task and participant is recorded.
        Default is ``False``
    saveFittingProgress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``
    expData : dict, optional
        The data from the task. Default ``None``

    Returns
    -------
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the previous and current participant fits

    See Also
    --------
    outputting.pickleLog : records the picked data
    """
    logger = logging.getLogger('Logging')
    partNameStr = str(part_name)

    message = "Recording participant " + partNameStr + " model fit"
    logger.info(message)

    label = "_Model-" + model_name + "_Part-" + partNameStr

    participantName = "Participant " + partNameStr

    participant.setdefault("Name", participantName)
    participant.setdefault("assigned_name", participantName)
    fitting_data.setdefault("Name", participantName)

    if fileNameGen:
        message = "Store data for " + participantName
        logger.info(message)

        participantFits = record_fitting(fitting_data, label, participant, partModelVars, participantFits, fileNameGen,
                                         save_fitting_progress=saveFittingProgress)

        if pickleData:
            if expData is not None:
                outputting.pickleLog(expData, fileNameGen, "_expData" + label)
            outputting.pickleLog(model_data, fileNameGen, "_modelData" + label)
            outputting.pickleLog(participant, fileNameGen, "_partData" + label)
            outputting.pickleLog(fitting_data, fileNameGen, "_fitData" + label)

    return participantFits


# %% Recording
def record_fitting(fitting_data, label, participant, participant_model_variables, participant_fits, file_name_generator,
                   save_fitting_progress=False):
    """
    Records formatted versions of the fitting data

    Parameters
    ----------
    fitting_data : dict, optional
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters.
    label : basestring
        The label used to identify the fit in the file names
    participant : dict
        The participant data
    participant_model_variables : dict of string
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string.
    participant_fits : defaultdict of lists
        A dictionary to be filled with the summary of the participant fits
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    save_fitting_progress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``

    Returns
    -------
    participant_fits : defaultdict of lists
        A dictionary to be filled with the summary of the previous and current participant fits

    """
    extendedLabel = "ParameterFits" + label

    participant_fits["Name"].append(participant["Name"])
    participant_fits["assigned_name"].append(participant["assigned_name"])
    for k in filter(lambda x: 'fit_quality' in x, fitting_data.keys()):
        participant_fits[k].append(fitting_data[k])
    for k, v in fitting_data["final_parameters"].iteritems():
        participant_fits[k].append(v)
    for k, v in participant_model_variables.iteritems():
        participant_fits[v] = participant[k]

    if save_fitting_progress:
        xlsx_fitting_data(fitting_data.copy(), extendedLabel, participant, file_name_generator)

    return participant_fits


#%% logging
def log_model_fitting_parameters(model, model_fit_variables, model_other_args):
    """
    Logs the model and task parameters that used as initial fitting conditions

    Parameters
    ----------
    model : string
        The name of the model
    model_fit_variables : dict
        The model parameters that will be fitted over and varied.
    model_other_args : dict
        The other parameters used in the model whose attributes have been
        modified by the user
    """
    model_args = copy.copy(model_fit_variables)
    model_args.update(copy.copy(model_other_args))
    model_instance = model(**model_args)
    model_properties = model_instance.params()

    message = "The fit will use the model ``{}``".format(model_properties['Name'])

    modelFitParams = [k + ' around ' + str(v) for k, v in model_fit_variables.iteritems()]
    message += " fitted with the parameters " + ", ".join(modelFitParams)

    model_parameters = [k + ' = ' + str(v).replace('\n', '').strip('[](){}<>') for k, v in model_other_args.iteritems()
                                                                               if k not in model_fit_variables.keys()]
    message += " and using the other user specified parameters " + ", ".join(model_parameters)

    logger_sim = logging.getLogger('Fitting')
    logger_sim.info(message)


def log_model_fitted_parameters(model_fit_variables, model_parameters, fit_quality, participant_name):
    """
    Logs the model and task parameters that used as initial fitting
    conditions

    Parameters
    ----------
    model_fit_variables : dict
        The model parameters that have been fitted over and varied.
    model_parameters : dict
        The model parameters for the fitted model
    fit_quality : float
        The value of goodness of fit
    participant_name : int or string
        The identifier for each participant
    """
    parameters = model_fit_variables.keys()

    model_fit_parameters = [k + ' = ' + str(v).strip('[]()') for k, v in model_parameters.iteritems() if k in parameters]
    message = "The fitted values for participant " + str(participant_name) + " are " + ", ".join(model_fit_parameters)

    message += " with a fit quality of " + str(fit_quality) + "."

    logger_sim = logging.getLogger('Fitting')
    logger_sim.info(message)


def log_fitting_parameters(fit_info):
    """
    Records and outputs to the log the parameters associated with the fitting algorithms

    Parameters
    ----------
    fit_info : dict
        The details of the fitting
    """

    log = logging.getLogger('Fitting')

    message = "Fitting information:"
    log.info(message)

    name = fit_info.pop('Name')
    message = "For " + name + ":"
    log.info(message)
    for k, v in fit_info.iteritems():
        message = k + ": " + repr(v)
        log.info(message)


#%% CSV
def fit_record(participant_fits, file_name_generator):
    """
    Returns the participant fits summary as a csv file

    Parameters
    ----------
    participant_fits : dict
        A summary of the recovered parameters
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    """
    participant_fit = pd.DataFrame.from_dict(participant_fits)
    output_file = file_name_generator("participantFits", 'csv')
    participant_fit.to_csv(output_file)


#%% Excel
def xlsx_fitting_data(fitting_data, label, participant, file_name_generator):
    """
    Saves the fitting data to an XLSX file

    Parameters
    ----------
    fitting_data : dict, optional
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters.
    label : basestring
        The label used to identify the fit in the file names
    participant : dict
        The participant data
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    """

    data = collections.OrderedDict()
    partData = outputting.newListDict(participant, 'part')
    data.update(partData)

    parameter_fitting_dict = copy.copy(fitting_data["tested_parameters"])
    parameter_fitting_dict['participant_fitting_name'] = fitting_data.pop("Name")
    #parameter_fitting_dict['fit_quality'] = fittingData.pop("fit_quality")
    #parameter_fitting_dict["fitQualities"] = fittingData.pop("fitQualities")
    for k, v in fitting_data.pop("final_parameters").iteritems():
        parameter_fitting_dict[k + "final"] = v
    parameter_fitting_dict.update(fitting_data)
    data.update(parameter_fitting_dict)
    record_data = outputting.newListDict(data, "")

    record = pd.DataFrame(record_data)

    name = "data/" + label
    output_file = file_name_generator(name, 'xlsx')
    xlsxT = pd.ExcelWriter(output_file)
    # TODO: Remove the engine specification when moving to Python 3
    record.to_excel(xlsxT, sheet_name='ParameterFits', engine='XlsxWriter')
    xlsxT.save()


#if __name__ == '__main__':
#    fire.Fire(data_fitting)
