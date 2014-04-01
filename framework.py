# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from itertools import izip
from importlib import import_module

from outputs import plots, pickleLog
from utils import fancyLogger, argProcess, saving, listMerge
from plotting import varDynamics

# For analysing the state of the computer
# import psutil

### +++++ Basic simulation

def sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName):
    """This is the main run of the simulation

    sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)"""


    logger1 = logging.getLogger('Overview')

    message = "Beginning the core simulation "
    logger1.debug(message)

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
        plots(experiment, folderName, silent, save, label, modelResults)

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

    decisionTimes = []

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
            paramText += param + ' = ' + str(val).strip('[]()') + ', '

        if len(paramText)>18:
            l = "Group " + str(labelCount)
            labelCount += 1
            message = "Outputting '" + paramText + "' with the label '" + l + "'"
            logger2.info(message)
            paramText = l

        message = "Begining experiment with " + paramText
        logger1.info(message)

        experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs, folderName)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        decisionTimes.append(modelResult["firstDecision"])
        modelResults[paramText] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        pl = varDynamics(params, paramVals, decisionTimes)
        majFigureSets = (("firstDecision",pl),)
        plots(experiment, folderName, silent, save, label, modelResults, *majFigureSets)
        varDynamics(params, paramVals, decisionTimes)


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

    modelNames = []

    expDataSets = {}
    modelResults = {}

    for m in args:

        modelName = m["Name"]
        if modelName in modelNames:
            i = 1
            while modelName + "_"+ str(i) in modelName:
                i +=1
            modelName = modelName + "_"+ str(i)
        modelNames.append(modelName)

        for k, v in m.iteritems():
            modelArgs[k] = v

        message = "Begining experiment with " + modelName
        logger1.info(message)

        experiment, model = sim(expName, m["Name"], modelArgs, expArgs, otherArgs, folderName)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        modelResults[modelName] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= modelName)

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, modelResults)

    message = "### Simulation complete"
    logger1.info(message)

if __name__ == '__main__':

    from numpy import fromfunction

    # For RPE:
#    paramModSim("Beads", "RPE", ('rateConst',fromfunction(lambda i, j: i/500+0.1, (400, 1))), ('beta',fromfunction(lambda i, j: i/10+0.1, (9, 1))))
#    paramModSim("Beads", "RPE", ('rateConst',fromfunction(lambda i, j: i/5000+0.1, (4000, 1))), m_beta = 0.3)
#    paramModSim("Beads", "RPE", ('rateConst',fromfunction(lambda i, j: i/10+0.1, (7, 1))), m_beta = 0.3)

    # For MS:
#    paramModSim("Beads", "MS", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('actParam',fromfunction(lambda i, j: i/10, (9, 1))))
#    paramModSim("Beads", "MS", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('actParam',fromfunction(lambda i, j: i/10, (9, 1))), beta= 0.5)
#    paramModSim("Beads", "MS", ('theta',[1,2,4]), ('actParam',[0.3, 0.6]))

    # For MS_rev:
#    paramModSim("Beads", "MS_rev", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('actParam',fromfunction(lambda i, j: i/10, (9, 1))))
#    paramModSim("Beads", "MS_rev", ('theta',[1,2,4]), ('actParam',[0.3, 0.6]))
#    simpleSim("Beads", "MS_rev")

    # For BP
#    paramModSim("Beads", "BP", ('theta',fromfunction(lambda i, j: i/5, (80, 1))), ('beta',fromfunction(lambda i, j: i/10+0.1, (9, 1))))
#    paramModSim("Beads", "BP", ('theta',[]), ('beta',fromfunction(lambda i, j: i/10+0.1, (9, 1))))
#    simpleSim("Beads", "BP")


    multiModelSim("Beads", {'Name':'RPE'},{'Name':'MS'},{'Name':'MS_rev'},{'Name':'BP'})