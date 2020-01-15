# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import fire

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

    outputFolder, fileNameGen, closeLoggers = outputting.saving(label=sim_label,
                                                                pickle=pickle,
                                                                config_file=config_file,
                                                                min_log_level=min_log_level,
                                                                numpy_error_level=numpy_error_level)

    logger = logging.getLogger('Overview')

    simID = 0

    message = "Beginning the simulation set"
    logger.debug(message)

    for expNum in experiments.iter_experiment_ID():

        for model in models:

            exp = experiments.new_experiment(expNum)

            logSimParams(exp.params(), model.params(), simID=str(simID))

            message = "Beginning experiment"
            logger.debug(message)

            for state in exp:
                model.observe(state)
                act = model.action()
                exp.receiveAction(act)
                response = exp.feedback()
                model.feedback(response)
                exp.proceed()

            model.setsimID(str(simID))

            message = "Experiment completed"
            logger.debug(message)

            recordSim(fileNameGen, exp.returnTaskState(), model.returnTaskState(), str(simID), pickleData=pickle)

            simID += 1

    closeLoggers()


def recordSim(fileNameGen, expData, modelData, simID, pickleData=False):
    """
    Records the data from an experiment-model run. Creates a pickled version

    Parameters
    ----------
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    expData : dict
        The data from the experiment
    modelData : dict
        The data from the model
    simID : basestring
        The label identifying the simulation
    pickleData : bool, optional
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

    simModelCSV(modelData, simID, fileNameGen)

    if pickleData:
        outputting.pickleLog(expData, fileNameGen, "_expData" + label)
        outputting.pickleLog(modelData, fileNameGen, "_modelData" + label)


def logSimParams(expParams, modelParams, simID):
    """
    Writes to the log the description and the label of the experiment and model

    Parameters
    ----------
    expParams : dict
        The experiment parameters
    modelParams : dict
        The model parameters
    simID : string
        The identifier for each simulation.

    See Also
    --------
    recordSimParams : Records these parameters for later use
    """

    expDesc = expParams.pop('Name') + ": "
    expDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in expParams.iteritems()]
    expDesc += ", ".join(expDescriptors)

    modelDesc = modelParams.pop('Name') + ": "
    modelDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in modelParams.iteritems()]
    modelDesc += ", ".join(modelDescriptors)

    message = "Simulation " + simID + " contains the experiment '" + expDesc + "'."
    message += "The model used is '" + modelDesc + "'."

    loggerSim = logging.getLogger('Simulation')
    loggerSim.info(message)


def simModelCSV(modelData, simID, fileNameGen):
    # type: (dict, basestring, function) -> None
    """
    Saves the fitting data to a CSV file

    Parameters
    ----------
    modelData : dict
        The data from the model
    simID : string
        The identifier for the simulation
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    """

    data = outputting.newListDict(modelData)
    record = pd.DataFrame(data)
    name = "data/modelSim_" + simID
    outputFile = fileNameGen(name, 'csv')
    record.to_csv(outputFile)

if __name__ == '__main__':
  fire.Fire(simulation)
