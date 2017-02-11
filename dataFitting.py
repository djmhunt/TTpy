# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from simulation import simulation

from fitting.fit import fit

def dataFitting(experiments, models, outputting, data = None, fitter = None, partLabel = "Name"):
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
        An outputting class instance
    data : list of dictionaries
        Each dictionary should all contain the keys associated with the fitting
    fitter : fitting.fit.fit
        A fitting class instance
    partLabel : basestring, optional
        The key (label) used to identify each participant. Default ``Name``

    See Also
    --------
    experiments.experiments : The experiments factory
    models.models : The model factory
    outputting.outputting : The outputting class
    fitting.fit.fit : Abstract class for fitting data
    data.data : Data import function
    """

#    if not (isinstance(data, pandas.DataFrame) and isinstance(fitter,fitters.fitter)):
    if not (isinstance(data, list)): #and isinstance(fitter, fit)):

        logger = outputting.getLogger('dataFitting')

        message = "Data or fitter missing. Starting a simple simulation"
        logger.warning(message)

        simulation(experiments, models, outputting)

        return


    logger = outputting.getLogger('Overview')

    outputting.recordFittingParams(fitter.info())

    message = "Beginning the data fitting"
    logger.info(message)

    for modelInfo in models.iterFitting():

        model = modelInfo[0]
        modelSetup = modelInfo[1:]

        exp = experiments.create(0)

        outputting.logSimFittingParams(exp.params(), model.Name, modelSetup[0], modelSetup[1])

        for participant in data:

            partName = participant[partLabel]
            if isinstance(partName, (list, tuple)):
                partName = partName[0]

            # Find the best model values from those proposed

            message = "Beginning participant fit"
            logger.info(message)

            modelFitted, fitQuality, fittingData = fitter.participant(exp, model, modelSetup, participant)

            message = "Participant fitted"
            logger.debug(message)

            outputting.recordSimParams(exp.params(), modelFitted.params(), simID=partName)
            outputting.logModFittedParams(modelSetup[0], modelFitted.params(), fitQuality, partName)

            outputting.recordParticipantFit(participant,
                                            partName,
                                            exp.outputEvolution(),
                                            modelFitted.outputEvolution(),
                                            fitQuality,
                                            fittingData)

            outputting.plotModel(modelFitted.plot())

        outputting.plotModelSet(modelFitted.plotSet())

    outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()

