# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
#import fire

import pandas as pd

import outputting

from experimentGenerator import ExperimentGen
from modelGenerator import ModelGen


def simulation(experiment_name='Basic',
               experiment_changing_properties=None,
               experiment_constant_properties=None,
               model_name='QLearn',
               model_changing_properties=None,
               model_constant_properties=None,
               sim_label=None,
               config_file=None,
               pickle=False,
               min_log_level='INFO',
               numpy_error_level="log"):
    """
    A framework for letting models interact with experiments and record the data

    Parameters
    ----------
    experiment_name : string
        The name of the file where a experiment.experimentTemplate.Experiment class can be found. Default ``Basic``
    experiment_changing_properties : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across experiment instances. When a parameter
        contains a list, an instance of the experiment will be created for every combination of this parameter with all
        the others. Default ``None``
    experiment_constant_properties : dictionary of float, string or binary valued elements
        These contain all the the experiment options that describe the experiment being studied but do not vary across
        experiment instances. Default ``None``
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
    pickle : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    sim_label : string, optional
        The label for the simulation. Default ``None``, which means nothing will be saved
    min_log_level : basestring, optional
        Defines the level of the log from (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``). Default ``INFO``
    numpy_error_level : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    See Also
    --------
    experiments.experiments, models.models
    """

    experiments = ExperimentGen(experiment_name=experiment_name,
                                parameters=experiment_changing_properties,
                                other_options=experiment_constant_properties)

    models = ModelGen(model_name=model_name,
                      parameters=model_changing_properties,
                      other_options=model_constant_properties)

    output_folder, file_name_generator, close_loggers = outputting.saving(label=sim_label,
                                                                          pickle=pickle,
                                                                          config_file=config_file,
                                                                          min_log_level=min_log_level,
                                                                          numpy_error_level=numpy_error_level)

    logger = logging.getLogger('Overview')

    simID = 0

    message = "Beginning the simulation set"
    logger.debug(message)

    for experiment_number in experiments.iter_experiment_ID():

        for model in models:

            exp = experiments.new_experiment(experiment_number)

            log_sim_parameters(exp.params(), model.params(), simID=str(simID))

            message = "Beginning experiment"
            logger.debug(message)

            for state in exp:
                model.observe(state)
                action = model.action()
                exp.receiveAction(action)
                response = exp.feedback()
                model.feedback(response)
                exp.proceed()

            model.setsimID(str(simID))

            message = "Experiment completed"
            logger.debug(message)

            record_sim(file_name_generator, exp.returnTaskState(), model.returnTaskState(), str(simID), pickle=pickle)

            simID += 1

    close_loggers()


def record_sim(file_name_generator, experiment_data, model_data, simID, pickle=False):
    """
    Records the data from an experiment-model run. Creates a pickled version

    Parameters
    ----------
    file_name_generator : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    experiment_data : dict
        The data from the experiment
    model_data : dict
        The data from the model
    simID : basestring
        The label identifying the simulation
    pickle : bool, optional
        If true the data for each model, experiment and participant is recorded.
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

    csv_model_sim(model_data, simID, file_name_generator)

    if pickle:
        outputting.pickleLog(experiment_data, file_name_generator, "_expData" + label)
        outputting.pickleLog(model_data, file_name_generator, "_modelData" + label)


def log_sim_parameters(experiment_parameters, model_parameters, simID):
    """
    Writes to the log the description and the label of the experiment and model

    Parameters
    ----------
    experiment_parameters : dict
        The experiment parameters
    model_parameters : dict
        The model parameters
    simID : string
        The identifier for each simulation.

    See Also
    --------
    recordSimParams : Records these parameters for later use
    """

    experiment_description = experiment_parameters.pop('Name') + ": "
    experiment_descriptors = [k + ' = ' + str(v).strip('[]()') for k, v in experiment_parameters.iteritems()]
    experiment_description += ", ".join(experiment_descriptors)

    model_description = model_parameters.pop('Name') + ": "
    model_descriptors = [k + ' = ' + str(v).strip('[]()') for k, v in model_parameters.iteritems()]
    model_description += ", ".join(model_descriptors)

    message = "Simulation " + simID + " contains the experiment '" + experiment_description + "'."
    message += "The model used is '" + model_description + "'."

    logger_sim = logging.getLogger('Simulation')
    logger_sim.info(message)


def csv_model_sim(modelData, simID, file_name_generator):
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
