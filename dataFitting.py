# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

from simulation import simulation

def dataFitting(experiments, models, outputting, data = None, fitter = None, fittingParams = (None,None)):
    """ A framework for fitting models to data for experiments, along with recording the data

        Variables:
        experiments: An instance of the experiments factory
        models: An instance of the models factory
        outputing: An instance of the outputting class
        data: a Pandas dataframe
        fitter: An instance of the fitter class
        fittingParams : Tuple of strings of form (dataParam,modelParam)

        dataFitting(experiments, models, outputting, data = None, fitter = None)
    """

    if not data or not fitter:

        logger = outputting.getLogger('dataFitting')

        message = "Data or fitter missing. Starting a simple simulation"
        logger.warning(message)

        simulation(experiments, models, outputting)

        return


    logger = outputting.getLogger('Overview')

    message = "Beginning the data fitting"
    logger.debug(message)

    for expNum in experiments:

        for model in models.iterFitting():

            for participant in data.iterrows():

                # Find the best model values from those proposed

                exp = experiments.create(expNum)

                message = "Beginning run"
                logger.debug(message)

                model = fitter(exp, model, simRun, fittingParams)

                message = "Experiment run"
                logger.debug(message)

                outputting.recordSimParams(exp.params(),model.params())

                outputting.recordSim(exp.outputEvolution(),model.outputEvolution())

                outputting.plotModel(model.plot())

            outputting.plotModelSet(model.plotSet())

        outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()

def simRun(exp, model):

    for event in exp:
        model.observe(event)
        act = model.action()
        exp.receiveAction(act)
        response = exp.feedback()
        model.feedback(response)
        exp.procede()