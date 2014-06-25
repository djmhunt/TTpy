# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
import logging

from itertools import izip
from importlib import import_module
from numpy import array, sort

from outputs import plots, pickleLog, simSetLog, varCategoryDynamics
from utils import fancyLogger, argProcess, saving, listMergeNP
from plotting import varDynamics
from inputs import unpickleModels, unpickleSimDescription

# For analysing the state of the computer
# import psutil

### +++++ Basic simulation

def sim(expName, modelName, modelArgs, expArgs, otherArgs):
    """This is the main run of the simulation

    experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs)"""


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

    message = "Beginning experiment"
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
    expArgs, modelArgs, plotArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Single")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName, silent)

    logger1 = logging.getLogger('Framework')

    message = "Begining simple experiment"
    logger1.debug(message)

    experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs)

    message = "Begining output processing"
    logger1.info(message)

    expData = experiment.outputEvolution()
    modelResult = model.outputEvolution()
    modelResults = {model.Name: modelResult}

    if folderName:
        pickleLog(expData,folderName)
        pickleLog(modelResult,folderName)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, modelResults, **plotArgs)

    message = "### Simulation complete"
    logger1.info(message)

def paramModSim(expName, modelName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, plotArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Parameter")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName, silent)

    logger1 = logging.getLogger('Framework')
    logger2 = logging.getLogger('Outputs')

    expDataSets = {}
    modelResults = {}

    decisionTimes = []

    readableLog_paramText = []
    readableLog_paramVals = []
    readableLog_firstDecision = []

    params = []
    paramVals = []
    otherParams = []
    for a in args:
        if len(a)==2:
            params.append(a[0])
            paramVals.append(sort(a[1]))
        else:
            otherParams.append(a)

    paramCombs = listMergeNP(*paramVals)

    labelCount = 1
    for p in paramCombs:

        paramTexts = []
        for param, val in izip(params,p):
            modelArgs[param] = val
            paramTexts.append(param + ' = ' + str(val).strip('[]()'))
        paramText = ", ".join(paramTexts)

        if len(paramText)>18:
            l = "Group " + str(labelCount)
            labelCount += 1
            message = "Outputting '" + paramText + "' with the label '" + l + "'"
            logger2.info(message)
            paramText = l

        message = "Begining experiment with " + paramText
        logger1.info(message)

        experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        decisionTimes.append(modelResult["firstDecision"])
        modelResults[paramText] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

            # Store the added data for the record set
            readableLog_paramText.append(paramText)
            readableLog_paramVals.append(p)
            readableLog_firstDecision.append(modelResult["firstDecision"])


    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        pl = varDynamics(params, paramVals, array(decisionTimes), **plotArgs)
        majFigureSets = (("firstDecision",pl),)
        plots(experiment, folderName, silent, save, label, modelResults, *majFigureSets, **plotArgs)
#        varDynamics(params, paramVals, decisionTimes)

    if save:

        varCategoryDynamics(params, paramVals, array(decisionTimes), folderName)

        simSetLog(readableLog_paramText,
                  params,
                  array(readableLog_paramVals),
                  array(readableLog_firstDecision),
                  expName,
                  modelName,
                  kwargs,
                  label,
                  folderName)


    message = "### Simulation complete"
    logger1.info(message)

def multiModelSim(expName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, plotArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Models")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName, silent)

    logger1 = logging.getLogger('Framework')

    modelNames = []

    readableLog_paramText = []
    readableLog_paramVals = []
    readableLog_firstDecision = []

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

        mArgs = modelArgs.copy()
        for k, v in m.iteritems():
            mArgs[k] = v

        message = "Begining experiment with " + modelName
        logger1.info(message)

        experiment, model = sim(expName, m["Name"], mArgs, expArgs, otherArgs)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        modelResults[modelName] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= modelName)

            # Store the added data for the record set
            readableLog_paramText.append(modelName)
            readableLog_paramVals.append([])
            readableLog_firstDecision.append(modelResult["firstDecision"])

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, modelResults , **plotArgs)

    if save:
        simSetLog(readableLog_paramText,
                  readableLog_paramText,
                  array(readableLog_paramVals),
                  array(readableLog_firstDecision),
                  expName,
                  modelName,
                  kwargs,
                  label,
                  folderName)

    message = "### Simulation complete"
    logger1.info(message)

def modelDynSim(expName, modelNames, *args, **kwargs):
    """Comparing the dynamics of models across the same parameter range for a series of parameters

    Interface based on paramModSim, but taking a list of models rather than an individual one

    modelDynSim(expName, modelNames, *args, **kwargs)
    """

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, plotArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', True)
    label = otherArgs.pop("ivLabel","Parameter")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    folderName, fileName = saving(save, label)

    fancyLogger(logLevel, fileName, silent)

    logger1 = logging.getLogger('Framework')
    logger2 = logging.getLogger('Outputs')

    expDataSets = {}
    modelResults = {}

    decisionTimes = []

    readableLog_paramText = []
    readableLog_paramVals = []
    readableLog_firstDecision = []

    params = []
    paramVals = []
    otherParams = []
    for a in args:
        if len(a)==2:
            params.append(a[0])
            paramVals.append(a[1])
        else:
            otherParams.append(a)

    paramCombs = listMergeNP(*paramVals)

    labelCount = 1
    for p in paramCombs:

        paramTexts = []
        for param, val in izip(params,p):
            modelArgs[param] = val
            paramTexts.append(param + ' = ' + str(val).strip('[]()'))
        paramText = ", ".join(paramTexts)

        if len(paramText)>18:
            l = "Group " + str(labelCount)
            labelCount += 1
            message = "Outputting '" + paramText + "' with the label '" + l + "'"
            logger2.info(message)
            paramText = l

        message = "Begining experiment with " + paramText
        logger1.info(message)

        experiment, model = sim(expName, modelName, modelArgs, expArgs, otherArgs)
        expData = experiment.outputEvolution()
        modelResult = model.outputEvolution()
        decisionTimes.append(modelResult["firstDecision"])
        modelResults[paramText] = modelResult

        message = "Experiment ended. Recording data"
        logger1.debug(message)

        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

            # Store the added data for the record set
            readableLog_paramText.append(paramText)
            readableLog_paramVals.append(p)
            readableLog_firstDecision.append(modelResult["firstDecision"])


    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        pl = varDynamics(params, paramVals, array(decisionTimes), **plotArgs)
        majFigureSets = (("firstDecision",pl),)
        plots(experiment, folderName, silent, save, label, modelResults, *majFigureSets, **plotArgs)
#        varDynamics(params, paramVals, decisionTimes)

    if save:
        simSetLog(readableLog_paramText,
                  params,
                  array(readableLog_paramVals),
                  array(readableLog_firstDecision),
                  expName,
                  modelName,
                  kwargs,
                  label,
                  folderName)


    message = "### Simulation complete"
    logger1.info(message)

def plotLoadSim(folderName, *args, **kwargs):

    # Sift through kwargs to find those related to the experiment and those related to
    # the model
    expArgs, modelArgs, plotArgs, otherArgs = argProcess(**kwargs)

    silent = otherArgs.get('silent',False)
    save = otherArgs.get('save', False)
    label = otherArgs.pop("ivLabel","Reprocess")
    logLevel = otherArgs.pop("logLevel",logging.DEBUG)

    if save:
        fileName = folderName + "reloadLog.txt"
    else:
        fileName = ''

    fancyLogger(logLevel, fileName, silent)

    logger1 = logging.getLogger('Framework')

    modelResults = {}

    for simLabel, model in unpickleModels(folderName):

        modelResults[simLabel] = model

    simDetails = unpickleSimDescription(folderName)

    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        plots(experiment, folderName, silent, save, label, modelResults , **plotArgs)

    message = "### Simulation complete"
    logger1.info(message)

if __name__ == '__main__':

    from numpy import fromfunction

    saveSims = True
    silence = True
    alpha_central = 0.2
    beta_central = 0.15
    theta_central = 1.5
    alpha = fromfunction(lambda i, j: i/60.0 + alpha_central - 10/60.0 , (20, 1))
    beta = fromfunction(lambda i, j: i/400.0 + beta_central - 27.5/400.0, (55, 1))
    theta = fromfunction(lambda i, j: i/30.0 + theta_central - 0.5, (30, 1))

    altBeads = [1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1]

    # For EP:
#    paramModSim("Beads", "EP", ('alpha',fromfunction(lambda i, j: i/500+0.1, (400, 1))), ('beta',fromfunction(lambda i, j: i/10+0.1, (9, 1))),ivLabel = 'EP_broad', save = saveSims)
#    paramModSim("Beads", "EP", ('alpha',fromfunction(lambda i, j: i/5000+0.1, (4000, 1))), m_beta = 0.3, ivLabel = "EP_beta-0'3", save = saveSims)
#    paramModSim("Beads", "EP", ('alpha',fromfunction(lambda i, j: i/10+0.1, (7, 1))), m_beta = 0.3, ivLabel = 'EP_narrow', save = saveSims)
#    paramModSim("Beads", "EP", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('alpha',fromfunction(lambda i, j: i/10, (9, 1))),ivLabel = 'MS_broad_theta', save = saveSims)


    # For MS:
#    paramModSim("Beads", "MS", ('theta',fromfunction(lambda i, j: i/5, (100, 1))), ('beta',fromfunction(lambda i, j: i/10, (5, 1))),ivLabel = 'MS_theta_beta', save = saveSims)
#    paramModSim("Beads", "MS", ('theta',fromfunction(lambda i, j: i/20, (100, 1))), ('alpha',fromfunction(lambda i, j: i/30, (31, 1))), m_beta=0.1, ivLabel = 'MS_theta_beta', save = saveSims, p_contour = False)
#    paramModSim("Beads", "MS", ('beta',fromfunction(lambda i, j: i/10, (5, 1))), ('alpha',fromfunction(lambda i, j: i/30, (31, 1))), m_beta=0.1, ivLabel = 'MS_theta_beta', save = saveSims, p_contour = False)
#    paramModSim("Beads", "MS", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('alpha',fromfunction(lambda i, j: i/10, (9, 1))), m_beta= 0.5,ivLabel = "MS_beta-0'5", save = saveSims)
#    paramModSim("Beads", "MS", ('theta',[1,2,4]), ('alpha',[0.3, 0.6]), ivLabel = 'MS_narrow', save = saveSims)

#    # For MS_rev:
#    paramModSim("Beads", "MS_rev", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('alpha',fromfunction(lambda i, j: i/100, (99, 1))),ivLabel = 'MS_rev_broad', save = saveSims)
#    paramModSim("Beads", "MS_rev", ('theta',[1,2,4]), ('alpha',[0.3, 0.6]), ivLabel = 'MS_rev_narrow', save = saveSims)
#    paramModSim("Beads", "MS_rev", ('theta',[4,5,6,7]), ('alpha',[0.1]), ivLabel = 'MS_rev_narrow', save = saveSims)
#    simpleSim("Beads", "MS_rev", ivLabel = 'MS_rev_single')

    # For BP
#    paramModSim("Beads", "BP", ('theta',fromfunction(lambda i, j: i/5, (40, 1))), ('beta',fromfunction(lambda i, j: i/10+0.3, (4, 1))),ivLabel = 'BP_broad', save = saveSims)
#    paramModSim("Beads", "BP", ('theta',fromfunction(lambda i, j: i/5+2.4, (18, 1))), ('beta',fromfunction(lambda i, j: i/10+0.2, (4, 1))), ivLabel = 'BP_narrow', save = saveSims)
#    paramModSim("Beads", "BP", ('theta',array([2.8,3.8,4.8])), m_beta = 0.4, ivLabel = 'BP_narrow', save = saveSims)
#    simpleSim("Beads", "BP", ivLabel = 'BP_single', save = saveSims)

#    paramModSim("Beads", "MS", ('theta',theta), ('alpha',alpha), ('beta',beta),ivLabel = 'MS_all', save = saveSims, silent = silence, exp_beadSequence = altBeads)
#    paramModSim("Beads", "MS_rev", ('theta',theta), ('alpha',alpha), ('beta',beta),ivLabel = 'MS_rev_all', save = saveSims, silent = silence, exp_beadSequence = altBeads)
#    paramModSim("Beads", "EP", ('theta',theta), ('alpha',alpha), ('beta',beta),ivLabel = 'EP_all', save = saveSims, silent = silence, exp_beadSequence = altBeads)
    paramModSim("Beads", "BP", ('theta',theta), ('alpha',alpha), ('beta',beta),ivLabel = 'BP_all', save = saveSims, silent = silence, exp_beadSequence = altBeads)



#    multiModelSim("Beads", {'Name':'EP'},{'Name':'MS'},{'Name':'MS_rev'},{'Name':'BP'}, ivLabel = 'most_models', save = saveSims)

#    plotLoadSim(".\\Outputs\\2014-4-3_BP_broad\\")

