# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

from simulation import simulation

from fitting.fit import fit

def dataFitting(experiments, models, outputting, data = None, fitter = None):
    """ 
    A framework for fitting models to data for experiments, along with 
    recording the plots and data associated with the best fits.

    Parameters
    ----------
    experiments : experiments.experiments
        An experiment factory generating each of the different experiments being considered
    models : models.models
        A model factory generating each of the different models being considered
    outputing : outputting.outputting
    data : list of dictionaries
    fitter : fitting.fit.fit
    """

#    if not (isinstance(data, pandas.DataFrame) and isinstance(fitter,fitters.fitter)):
    if not (isinstance(data, list) and isinstance(fitter,fit)):

        logger = outputting.getLogger('dataFitting')

        message = "Data or fitter missing. Starting a simple simulation"
        logger.warning(message)

        simulation(experiments, models, outputting)

        return


    logger = outputting.getLogger('Overview')

    outputting.recordFittingParams(fitter.info())

    message = "Beginning the data fitting"
    logger.debug(message)

    for modelInfo in models.iterFitting():

        model = modelInfo[0]
        modelSetup = modelInfo[1:]

        exp = experiments.create(0)

        for participant in data:

            # Find the best model values from those proposed

            message = "Beginning participant fit"
            logger.debug(message)

            modelFitted, fitQuality = fitter.participant(exp, model, modelSetup, participant)

            message = "Participant fitted"
            logger.debug(message)

            outputting.recordSimParams(exp.params(),modelFitted.params())

            outputting.recordParticipantFit(participant, exp.outputEvolution(),modelFitted.outputEvolution(), fitQuality)

            outputting.plotModel(modelFitted.plot())

        outputting.plotModelSet(modelFitted.plotSet())

    outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()

