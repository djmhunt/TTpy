# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

def  simulation(experiments, models, outputting):
    """ A framework for letting models interact with experiments and record the data

        Variables:
        experiments: An instance of the experiments factory
        models: An instance of the models factory
        outputing: An instance of the outputting class

        simulation(experiments, models, outputting)
    """

    logger = outputting.getLogger('Overview')

    message = "Beginning the simulation set"
    logger.debug(message)

    for exp in experiments:

        for model in models:

            exp.reset()

            outputting.recordSimParams(exp.params(),model.params())

            message = "Begining experiment"
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

        outputting.plotExperiment(exp.plot())