# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from itertools import izip

from simulation import simulation

from fitting.fit import fit
from fitting.fitters.qualityFunc import qualFuncIdent


def participantModelParamMatch(experiments, models, outputting, data, fitter, fitQualFunc, modelParam):
    """ 
    A framework for fitting models to data for experiments, along with 
    recording the plots and data associated with the best fits.

    Parameters
    ----------
    experiments : experiments.experiments
        An experiment factory generating each of the different experiments being considered
    models : models.models
        A model factory generating each of the different models being considered
    outputting : outputting.outputting
        An outputting class instance
    data : list of dictionaries
        Each dictionary should all contain the keys associated with the fitting
    fitter : fitting.fit.fit
        A fitting class instance
    fitQualFunc : string or function
        The function that collates the ``modelParam`` output and returns a model fit quality.
        This can describe the name as a string or pass a function directly
    modelParam : string
        The key label of the parameter used to calculate the fit quality
    
    See Also
    --------
    experiments.experiments : The experiments factory
    models.models : The model factory
    outputting.outputting : The outputting class
    fitting.fit.fit : Abstract class for fitting data
    data.data : Data import function
    fitting.fitters.qualityFunc : The fit quality functions
    """

    fitQualFunc = qualFuncIdent(fitQualFunc)

    logger = outputting.getLogger('Overview')

    # outputting.recordFittingParams(fitter.info())

    message = "Beginning the data fitting"
    logger.debug(message)

    exp = experiments.create(0)

    for modelInfo, participant in izip(models.iterFitting(), data):

        model = modelInfo[0]
        modelSetup = modelInfo[1:]

        outputting.logSimFittingParams(exp.params(), model.Name, modelSetup[0], modelSetup[1])

        message = "Beginning participant run"
        logger.debug(message)

        modelFitted = fitter.participantMatchResult(exp, model, modelSetup, participant)

        message = "Participant run"
        logger.debug(message)

        modelData = modelFitted.outputEvolution()
        modelVals = modelData[modelParam]
        fitQuality = fitQualFunc(modelVals)

        desc = outputting.recordSimParams(exp.params(), modelFitted.params())
        outputting.logModFittedParams(modelSetup[0], modelFitted.params(), fitQuality)

        outputting.recordParticipantFit(participant, exp.outputEvolution(), modelData, fitQuality)

        outputting.plotModel(modelFitted.plot())

    outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()

