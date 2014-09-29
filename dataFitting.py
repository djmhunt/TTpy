# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

from simulation import simulation

import pandas

from fitting.fit import fit

def dataFitting(experiments, models, outputting, data = None, fitter = None):
    """ A framework for fitting models to data for experiments, along with recording the data

        Variables:
        experiments: An instance of the experiments factory
        models: An instance of the models factory
        outputing: An instance of the outputting class
        data: A list of dictionaries not a Pandas dataframe
        fitter: An instance of the fitter class

        dataFitting(experiments, models, outputting, data = None, fitter = None)
    """

#    if not (isinstance(data, pandas.DataFrame) and isinstance(fitter,fitters.fitter)):
    if not (isinstance(data, list) and isinstance(fitter,fit)):

        logger = outputting.getLogger('dataFitting')

        message = "Data or fitter missing. Starting a simple simulation"
        logger.warning(message)

        simulation(experiments, models, outputting)

        return


    logger = outputting.getLogger('Overview')

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

            modelFitted = fitter.participant(exp, model, modelSetup, participant)

            message = "Participant fitted"
            logger.debug(message)

            outputting.recordSimParams(exp.params(),modelFitted.params())

            outputting.recordParticipantFit(participant, exp.outputEvolution(),modelFitted.outputEvolution())

            outputting.plotModel(modelFitted.plot())

        outputting.plotModelSet(modelFitted.plotSet())

    outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()

