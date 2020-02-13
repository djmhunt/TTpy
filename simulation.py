# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
#import fire

import pandas as pd

import outputting

from taskGenerator import TaskGeneration
from modelGenerator import ModelGen


def run(task_name='Basic',
        task_changing_properties=None,
        task_constant_properties=None,
        model_name='QLearn',
        model_changing_properties=None,
        model_constant_properties=None,
        label=None,
        config_file=None,
        output_path=None,
        pickle=False,
        min_log_level='INFO',
        numpy_error_level="log"):
    """
    A framework for letting models interact with tasks and record the data

    Parameters
    ----------
    task_name : string
        The name of the file where a tasks.taskTemplate.Task class can be found. Default ``Basic``
    task_changing_properties : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across task instances. When a parameter
        contains a list, an instance of the task will be created for every combination of this parameter with all
        the others. Default ``None``
    task_constant_properties : dictionary of float, string or binary valued elements
        These contain all the the task options that describe the task being studied but do not vary across
        task instances. Default ``None``
    model_name : string
        The name of the file where a model.modelTemplate.Model class can be found. Default ``QLearn``
    model_changing_properties : dictionary containing floats or lists of floats, optional
        Parameters are the options that you are or are likely to change across
        model instances. When a parameter contains a list, an instance of the
        model will be created for every combination of this parameter with
        all the others. Default ``None``
    model_constant_properties : dictionary of float, string or binary valued elements, optional
        These contain all the the model options that define the version
        of the model being studied. Default ``None``
    config_file : string, optional
        The file name and path of a ``.yaml`` configuration file. Overrides all other parameters if found.
        Default ``None``
    output_path : string, optional
        The path that will be used for the run output. Default ``None``
    pickle : bool, optional
        If true the data for each model, task and participant is recorded.
        Default is ``False``
    label : string, optional
        The label for the simulation. Default ``None``, which means nothing will be saved
    min_log_level : basestring, optional
        Defines the level of the log from (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``). Default ``INFO``
    numpy_error_level : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    See Also
    --------
    tasks.taskTemplate, model.modelTemplate
    """
    function_parameters = locals()

    tasks = TaskGeneration(task_name=task_name,
                           parameters=task_changing_properties,
                           other_options=task_constant_properties)

    models = ModelGen(model_name=model_name,
                      parameters=model_changing_properties,
                      other_options=model_constant_properties)

    output_folder, file_name_generator, close_loggers = outputting.saving(label=label,
                                                                          output_path=output_path,
                                                                          config_file=config_file,
                                                                          pickle=pickle,
                                                                          min_log_level=min_log_level,
                                                                          numpy_error_level=numpy_error_level)

    logger = logging.getLogger('Overview')

    simID = 0

    message = "Beginning the simulation set"
    logger.debug(message)

    for task_number in tasks.iter_task_ID():

        for model in models:

            tsk = tasks.new_task(task_number)

            log_simulation_parameters(tsk.params(), model.params(), simID=str(simID))

            message = "Beginning task"
            logger.debug(message)

            for state in tsk:
                model.observe(state)
                action = model.action()
                tsk.receiveAction(action)
                response = tsk.feedback()
                model.feedback(response)
                tsk.proceed()

            model.setsimID(str(simID))

            message = "Task completed"
            logger.debug(message)

            record_simulation(file_name_generator, tsk.returnTaskState(), model.returnTaskState(), str(simID), pickle=pickle)

            simID += 1

    close_loggers()


def record_simulation(file_name_generator, task_data, model_data, simID, pickle=False):
    """
    Records the data from an task-model run. Creates a pickled version

    Parameters
    ----------
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    task_data : dict
        The data from the task
    model_data : dict
        The data from the model
    simID : basestring
        The label identifying the simulation
    pickle : bool, optional
        If true the data for each model, task and participant is recorded.
        Default is ``False``

    See Also
    --------
    pickleLog : records the picked data
    """
    logger = logging.getLogger('Framework')

    message = "Beginning simulation output processing"
    logger.info(message)

    label = "_sim-" + simID

    message = "Store data for simulation " + simID
    logger.info(message)

    csv_model_simulation(model_data, simID, file_name_generator)

    if pickle:
        outputting.pickleLog(task_data, file_name_generator, "_taskData" + label)
        outputting.pickleLog(model_data, file_name_generator, "_modelData" + label)


def log_simulation_parameters(task_parameters, model_parameters, simID):
    """
    Writes to the log the description and the label of the task and model

    Parameters
    ----------
    task_parameters : dict
        The task parameters
    model_parameters : dict
        The model parameters
    simID : string
        The identifier for each simulation.

    See Also
    --------
    recordSimParams : Records these parameters for later use
    """

    task_description = task_parameters.pop('Name') + ": "
    task_descriptors = [k + ' = ' + str(v).strip('[]()') for k, v in task_parameters.iteritems()]
    task_description += ", ".join(task_descriptors)

    model_description = model_parameters.pop('Name') + ": "
    model_descriptors = [k + ' = ' + str(v).strip('[]()') for k, v in model_parameters.iteritems()]
    model_description += ", ".join(model_descriptors)

    message = "Simulation " + simID + " contains the task '" + task_description + "'."
    message += "The model used is '" + model_description + "'."

    logger_sim = logging.getLogger('Simulation')
    logger_sim.info(message)


def csv_model_simulation(modelData, simID, file_name_generator):
    # type: (dict, basestring, function) -> None
    """
    Saves the fitting data to a CSV file

    Parameters
    ----------
    modelData : dict
        The data from the model
    simID : string
        The identifier for the simulation
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    """

    data = outputting.newListDict(modelData)
    record = pd.DataFrame(data)
    name = "data/modelSim_" + simID
    outputFile = file_name_generator(name, 'csv')
    record.to_csv(outputFile)

#if __name__ == '__main__':
#    fire.Fire(simulation)
