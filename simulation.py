# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import pandas as pd

import outputting


def simulation(experiments, models, simLabel="Untitled", save=True, saveScript=True, pickleData=False, logLevel=logging.INFO, npSetErr="log"):
    """
    A framework for letting models interact with experiments and record the data

    Parameters
    ----------
    experiments : experimentGenerator.ExperimentGen
        An experiment factory generating each of the different experiments being considered
    models : modelGenerator.ModelGen
        A model factory generating each of the different models being considered
    save : bool, optional
        If true the data will be saved to files. Default ``True``
    saveScript : bool, optional
        If true a copy of the top level script running the current function
        will be copied to the log folder. Only works if save is set to ``True``
        Default ``True``
    pickleData : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    simLabel : string, optional
        The label for the simulation
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default ``logging.INFO``
    npSetErr : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    See Also
    --------
    experiments.experiments, models.models

    """

    outputFolder, fileNameGen, closeLoggers = outputting.saving(simLabel, save=save, pickleData=pickleData, saveScript=saveScript, logLevel=logLevel, npSetErr=npSetErr)

    logger = logging.getLogger('Overview')

    simID = 0

    message = "Beginning the simulation set"
    logger.debug(message)

    for expNum in experiments.iterExpID():

        for model in models:

            exp = experiments.newExp(expNum)

            logSimParams(exp.params(), model.params(), simID=str(simID))

            message = "Beginning experiment"
            logger.debug(message)

            for state in exp:
                model.observe(state)
                act = model.action()
                exp.receiveAction(act)
                response = exp.feedback()
                model.feedback(response)
                exp.procede()

            model.setsimID(str(simID))

            message = "Experiment completed"
            logger.debug(message)

            recordSim(fileNameGen, exp.outputEvolution(), model.outputEvolution(), str(simID), pickleData=pickleData)

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

    data = outputting.dictData2Lists(modelData)
    record = pd.DataFrame(data)
    name = "data/modelSim_" + simID
    outputFile = fileNameGen(name, 'csv')
    record.to_csv(outputFile)
