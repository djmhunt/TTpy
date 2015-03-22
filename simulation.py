# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

def  simulation(experiments, models, outputting):
    """ 
    A framework for letting models interact with experiments and record the data

    Parameters
    ----------
    experiments : experiments.experiments
        An experiment factory generating each of the different experiments being considered
    models : models.models
        A model factory generating each of the different models being considered
    outputing : outputting.outputting
        
    See Also
    --------
    experiments.experiments, models.models, outputting.outputting
        
    """

    logger = outputting.getLogger('Overview')

    message = "Beginning the simulation set"
    logger.debug(message)

    for expNum in experiments:

        for modelSet in models:

            for model in modelSet:

                exp = experiments.create(expNum)

                desc = outputting.recordSimParams(exp.params(),model.params())
                outputting.logSimParams(*desc)

                message = "Beginning experiment"
                logger.debug(message)

                for event in exp:
                    model.observe(event)
                    act = model.action()
                    exp.receiveAction(act)
                    response = exp.feedback()
                    model.feedback(response)
                    exp.procede()

                message = "Experiment completed"
                logger.debug(message)


                outputting.recordSim(exp.outputEvolution(),model.outputEvolution())

                outputting.plotModel(model.plot())

            outputting.plotModelSet(model.plotSet())

        outputting.plotExperiment(exp.plot())

    outputting.simLog()

    outputting.end()