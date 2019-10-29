# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import collections
import copy

import pandas as pd

from typing import Any

import outputting
import utils


def dataFitting(models, data, fitter, partLabel="Name", partModelVars={}, simLabel="Untitled", save=True, saveFittingProgress=False, saveScript=True, pickleData=False, logLevel=logging.INFO, npSetErr="log"):
    """
    A framework for fitting models to data for tasks, along with
    recording the data associated with the fits.

    Parameters
    ----------
    models : modelGenerator.ModelGen
        A model factory generating each of the different models being considered
    data : list of dictionaries
        Each dictionary should all contain the keys associated with the fitting
    fitter : fitAlgs.fitAlg
        A fitAlg class instance
    partLabel : basestring, optional
        The key (label) used to identify each participant. Default ``Name``
    partModelVars : dict of string, optional
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string. Default ``{}``
    simLabel : string, optional
        The label for the simulation
    save : bool, optional
        If true the data will be saved to files. Default ``True``
    saveFittingProgress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``
    saveScript : bool, optional
        If true a copy of the top level script running the current function
        will be copied to the log folder. Only works if save is set to ``True``
        Default ``True``
    pickleData : bool, optional
        If true the data for each model, and participant is recorded.
        Default is ``False``
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default ``logging.INFO``
    npErrResp : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr


    See Also
    --------
    models.models : The model factory
    outputting.outputting : The outputting class
    fitAlgs.fitSims.fitSim : Abstract class for a method of fitting data
    data.data : Data import function
    """

    outputFolder, fileNameGen, closeLoggers = outputting.saving(simLabel, save=save, pickleData=pickleData, saveScript=saveScript, logLevel=logLevel, npSetErr=npSetErr)

    if not (isinstance(data, list)):

        logger = logging.getLogger('dataFitting')
        message = "Data not recognised. "
        logger.warning(message)
        closeLoggers()
        return

    else:
        logger = logging.getLogger('Overview')

    logFittingParams(fitter.info())

    message = "Beginning the data fitting"
    logger.info(message)

    modelID = 0
    # Initialise the stores of information
    participantFits = collections.defaultdict(list)  # type: defaultdict[Any, list]

    for model, modelInitParamVars, modelStaticArgs in models.iterInitDetails():

        for v in partModelVars.itervalues():
            modelStaticArgs[v] = "<Varies for each participant>"

        logSimFittingParams(model.Name, modelInitParamVars, modelStaticArgs)

        for participant in data:

            partName = participant[partLabel]
            if isinstance(partName, (list, tuple)):
                partName = partName[0]

            for k, v in partModelVars.iteritems():
                modelStaticArgs[v] = participant[k]

            # Find the best model values from those proposed
            message = "Beginning participant fit for participant %s"%(partName)
            logger.info(message)

            modelFitted, fitQuality, fittingData = fitter.participant(model, (modelInitParamVars, modelStaticArgs), participant)

            message = "Participant fitted"
            logger.debug(message)

            logModFittedParams(modelInitParamVars, modelFitted.params(), fitQuality, partName)

            participantFits = recordParticipantFit(participant,
                                                   partName,
                                                   modelFitted.outputEvolution(),
                                                   str(modelID),
                                                   fittingData,
                                                   partModelVars,
                                                   participantFits,
                                                   outputFolder=outputFolder,
                                                   fileNameGen=fileNameGen,
                                                   pickleData=pickleData,
                                                   saveFittingProgress=saveFittingProgress)

        modelID += 1

    fitRecord(participantFits, fileNameGen)
    closeLoggers()


# %% Data record functions  
def recordParticipantFit(participant, partName, modelData, modelName, fittingData, partModelVars, participantFits, outputFolder=None, fileNameGen=None, pickleData=False, saveFittingProgress=False, expData=None):
    """
    Record the data relevant to the participant fitting

    Parameters
    ----------
    participant : dict
        The participant data
    partName : int or string
        The identifier for each participant
    modelData : dict
        The data from the model
    modelName : basestring
        The label given to the model
    fittingData : dict
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters
    partModelVars : dict of string
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string.
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the participant fits
    outputFolder : basestring, optional
        The folder into which the data will be saved. Default ``None``
    fileNameGen : function or None
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string. Default ``None``
    pickleData : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    saveFittingProgress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``
    expData : dict, optional
        The data from the experiment. Default ``None``

    Returns
    -------
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the previous and current participant fits

    See Also
    --------
    outputting.pickleLog : records the picked data
    """
    logger = logging.getLogger('Framework')
    partNameStr = str(partName)

    message = "Recording participant " + partNameStr + " model fit"
    logger.info(message)

    label = "_Model-" + modelName + "_Part-" + partNameStr

    participantName = "Participant " + partNameStr

    participant.setdefault("Name", participantName)
    participant.setdefault("assignedName", participantName)
    fittingData.setdefault("Name", participantName)

    if fileNameGen:
        message = "Store data for " + participantName
        logger.info(message)

        participantFits = recordFitting(fittingData, label, participant, partModelVars, participantFits, outputFolder, fileNameGen, saveFittingProgress=saveFittingProgress)

        if pickleData:
            if expData is not None:
                outputting.pickleLog(expData, fileNameGen, "_expData" + label)
            outputting.pickleLog(modelData, fileNameGen, "_modelData" + label)
            outputting.pickleLog(participant, fileNameGen, "_partData" + label)
            outputting.pickleLog(fittingData, fileNameGen, "_fitData" + label)

    return participantFits


# %% Recording
def recordFitting(fittingData, label, participant, partModelVars, participantFits, outputFolder, fileNameGen, saveFittingProgress=False):
    """
    Records formatted versions of the fitting data

    Parameters
    ----------
    fittingData : dict, optional
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters.
    label : basestring
        The label used to identify the fit in the file names
    participant : dict
        The participant data
    partModelVars : dict of string
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string.
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the participant fits
    outputFolder : basestring
        The folder into which the data will be saved
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    saveFittingProgress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``

    Returns
    -------
    participantFits : defaultdict of lists
        A dictionary to be filled with the summary of the previous and current participant fits

    """
    extendedLabel = "ParameterFits" + label

    participantFits["Name"].append(participant["Name"])
    participantFits["assignedName"].append(participant["assignedName"])
    for k in filter(lambda x: 'fitQuality' in x, fittingData.keys()):
        participantFits[k].append(fittingData[k])
    for k, v in fittingData["finalParameters"].iteritems():
        participantFits[k].append(v)
    for k, v in partModelVars.iteritems():
        participantFits[v] = participant[k]

    if saveFittingProgress:
        fittingDataXLSX(fittingData.copy(), extendedLabel, participant, outputFolder, fileNameGen)

    return participantFits


#%% logging
def logSimFittingParams(modelName, modelFitVars, modelOtherArgs, expParams=None):
    """
    Logs the model and experiment parameters that used as initial fitting
    conditions

    Parameters
    ----------
    modelName : string
        The name of the model
    modelFitVars : dict
        The model parameters that will be fitted over and varied.
    modelOtherArgs : dict
        The other parameters used in the model whose attributes have been
        modified by the user
    expParams : dict, optional
        The experiment parameters. Default ``dict()``
    """
    if expParams is None:
        expParams = {}
    message = "The fit will use the model '" + modelName + "'"

    modelFitParams = [k + ' around ' + str(v).strip('[]()') for k, v in modelFitVars.iteritems()]
    message += " fitted with the parameters " + ", ".join(modelFitParams)

    modelParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelOtherArgs.iteritems() if not isinstance(v, collections.Callable)]
    modelFuncs = [k + ' = ' + utils.callableDetailsString(v) for k, v in modelOtherArgs.iteritems() if isinstance(v, collections.Callable)]
    message += " and using the other user specified parameters " + ", ".join(modelParams)
    message += " and the functions " + ", ".join(modelFuncs)

    if len(expParams) > 0:
        message += ". This is based on the experiment '" + expParams['Name'] + "' "

        expDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in expParams.iteritems() if k != 'Name']
        message += "with the parameters " + ", ".join(expDescriptors) + "."

    loggerSim = logging.getLogger('Simulation')
    loggerSim.info(message)


def logModFittedParams(modelFitVars, modelParams, fitQuality, partName):
    """
    Logs the model and experiment parameters that used as initial fitting
    conditions

    Parameters
    ----------
    modelFitVars : dict
        The model parameters that have been fitted over and varied.
    modelParams : dict
        The model parameters for the fitted model
    fitQuality : float
        The value of goodness of fit
    partName : int or string
        The identifier for each participant
    """
    params = modelFitVars.keys()

    modelFitParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelParams.iteritems() if k in params]
    message = "The fitted values for participant " + str(partName) + " are " + ", ".join(modelFitParams)

    message += " with a fit quality of " + str(fitQuality) + "."

    loggerSim = logging.getLogger('Simulation')
    loggerSim.info(message)


def logFittingParams(fitInfo):
    """
    Records and outputs to the log the parameters associated with the fitting algorithms

    Parameters
    ----------
    fitInfo : dict
        The details of the fitting
    """

    log = logging.getLogger('Framework')

    message = "Fitting information:"
    log.info(message)

    for f in fitInfo:
        message = "For " + f['Name'] + ":"
        log.info(message)

        for k, v in f.iteritems():
            if k == "Name":
                continue

            message = k + ": " + repr(v)
            log.info(message)


#%% CSV
def fitRecord(participantFits, fileNameGen):
    """
    Returns the participant fits summary as a csv file

    Parameters
    ----------
    participantFits :
        A summary of the recovered parameters
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    """
    # TODO: Update the function documentation here when the datatype is known
    participantFit = pd.DataFrame.from_dict(participantFits)
    outputFile = fileNameGen("participantFits", 'csv')
    participantFit.to_csv(outputFile)


#%% Excel
def fittingDataXLSX(fittingData, label, participant, outputFolder, fileNameGen):
    """
    Saves the fitting data to an XLSX file

    Parameters
    ----------
    fittingData : dict, optional
        Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
        tested, in the order they were tested, and a list of the fit qualities of these parameters.
    label : basestring
        The label used to identify the fit in the file names
    participant : dict
        The participant data
    outputFolder : basestring
        The path of the output folder
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    """

    data = collections.OrderedDict()
    data['folder'] = outputFolder
    partFittingKeys, partFittingMaxListLen = outputting.listDictKeySet(participant)
    partData = outputting.newListDict(partFittingKeys, partFittingMaxListLen, participant, 'part')
    data.update(partData)

    paramFittingDict = copy.copy(fittingData["testedParameters"])
    paramFittingDict['partFitName'] = fittingData.pop("Name")
    #paramFittingDict['fitQuality'] = fittingData.pop("fitQuality")
    #paramFittingDict["fitQualities"] = fittingData.pop("fitQualities")
    for k, v in fittingData.pop("finalParameters").iteritems():
        paramFittingDict[k + "final"] = v
    paramFittingDict.update(fittingData)
    data.update(paramFittingDict)
    recordFittingKeys, recordFittingMaxListLen = outputting.listDictKeySet(data)
    recordData = outputting.newListDict(recordFittingKeys, recordFittingMaxListLen, data, "")

    record = pd.DataFrame(recordData)

    name = "data/" + label
    outputFile = fileNameGen(name, 'xlsx')
    xlsxT = pd.ExcelWriter(outputFile)
    # TODO: Remove the engine specification when moving to Python 3
    record.to_excel(xlsxT, sheet_name='ParameterFits', engine='XlsxWriter')
    xlsxT.save()
