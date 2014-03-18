# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from itertools import izip
from importlib import import_module

from outputs import plots, pickleLog
from utils import fancyLogger, argProcess, saving, listMerge, folderSetup

# For analysing the state of the computer
# import psutil

### +++++ Basic simulation

def sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName):
    """This is the main run of the simulation

    sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)"""


    logger1 = logging.getLogger('Overview')

    message = "Begninning the core simulation "
    logger1.info(message)

    # Simulation

    ## Initialise the experiment
    expFullName = "experiment_" + expName
    expMod = import_module(expFullName)
    experimentClass = eval("expMod." + expFullName)
    exp = experimentClass(**expArgs)

    message = expFullName + " loaded"
    logger1.info(message)

    ## Initialise the model
    modelFullName = "model_" + modelName
    modelMod = import_module(modelFullName)
    modelClass = eval("modelMod." + modelFullName)
    model = modelClass(**modelArgs)

    message = modelFullName + " loaded"
    logger1.info(message)

    message = "Begining experiment"
    logger1.debug(message)

    for event in exp:
        model.observe(event)
        act = model.action()
        exp.receiveAction(act)
        response = exp.feedback()
        model.feedback(response)
        exp.procede()

    message = "Experiment completed"
    logger1.debug(message)

    return exp, model


### +++++ Simulations
def simpleSim(expName, modelName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Single")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName)

    logger1 = logging.getLogger('Framework')

    message = "Begining simple experiment"
    logger1.debug(message)

    experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)

    message = "Begining output processing"
    logger1.info(message)

    expData = experiment.outputEvolution()
    modelResult = model.outputEvolution()
    modelResults = {model.Name: modelResult}

    if folderName:
        pickleLog(expData,folderName)
        pickleLog(modelResult,folderName)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, **modelResults)

    message = "### Simulation complete"
    logger1.info(message)

def paramModSim(expName, modelName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Parameter")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName)

    logger1 = logging.getLogger('Framework')
    logger2 = logging.getLogger('Outputs')

    expDataSets = {}
    modelResults = {}

    params = []
    paramVals = []
    otherParams = []
    for a in args:
        if len(a)==2:
            params.append(a[0])
            paramVals.append(a[1])
        else:
            otherParams.append(a)

    paramCombs = listMerge(*paramVals)

    labelCount = 1
    for p in paramCombs:

        paramText = ""
        for param, val in izip(params,p):
            modelArgs[param] = val
            paramText += param + ' = ' + str(val) + ' '

        if len(paramText)>18:
            l = "Group " + str(labelCount)
            labelCount += 1
            message = "Outputting '" + paramText + "' with the label '" + l + "'"
            logger2.info(message)
            paramText = l

        message = "Begining experiment with" + paramText
        logger1.info(message)

        experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        modelResults[paramText] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, **modelResults)

    message = "### Simulation complete"
    logger1.info(message)

def multiModelSim(expName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Parameter")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName)

    logger1 = logging.getLogger('Framework')
    logger2 = logging.getLogger('Outputs')

    expDataSets = {}
    modelResults = {}

    for p in paramCombs:

        paramText = ""
        for param, val in izip(params,p):
            modelArgs[param] = val
            paramText += param + ' = ' + str(val) + ' '

        if len(paramText)>18:
            l = "Group " + str(labelCount)
            labelCount += 1
            message = "Outputting '" + paramText + "' with the label '" + l + "'"
            logger2.info(message)
            paramText = l

        message = "Begining experiment with" + paramText
        logger1.info(message)

        experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        modelResults[paramText] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, **modelResults)

    message = "### Simulation complete"
    logger1.info(message)

if __name__ == '__main__':
#    sim(sys.argv)

#    singleModel(sys.argv)

#    simpleSim("Beads", "RPE")
    paramModSim("Beads", "RPE", ('rateConst',[0.1,0.2,0.3,0.4,0.5]))