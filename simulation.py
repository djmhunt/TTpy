# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import


def simulation(experiments, models, outputting):
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

    simID = 0

    message = "Beginning the simulation set"
    logger.debug(message)

    for expNum in experiments:

        for modelSet in models:

            for model in modelSet:

                exp = experiments.create(expNum)

                outputting.logSimParams(exp.params(), model.params(), simID=str(simID))

                message = "Beginning experiment"
                logger.debug(message)

                for state in exp:
                    model.observe(state)
                    act = model.action()
                    exp.receiveAction(act)
                    response = exp.feedback()
                    model.feedback(response)
                    exp.procede()

                model.setsimID(str(simID))

                message = "Experiment completed"
                logger.debug(message)

                outputting.recordSim(exp.outputEvolution(), model.outputEvolution(), str(simID))

                #outputting.plotModel(model.plot())

                simID += 1

            #outputting.plotModelSet(model.plotSet())

        #outputting.plotExperiment(exp.plot())

    #outputting.plotExperimentSet(exp.plotSet())

    outputting.end()