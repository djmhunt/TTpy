# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from fitAlgs.fitAlg import fitAlg


def dataFitting(models, outputting, data=None, fitter=None, partLabel="Name", partModelVars={}, experiments=None):
    """
    A framework for fitting models to data for experiments, along with
    recording the data associated with the fits.

    Parameters
    ----------
    models : models.models
        A model factory generating each of the different models being considered
    outputting : outputting.outputting
        An outputting class instance
    data : list of dictionaries
        Each dictionary should all contain the keys associated with the simMethods
    fitter : fitAlgs.fitAlg
        A fitAlg class instance
    partLabel : basestring, optional
        The key (label) used to identify each participant. Default ``Name``
    partModelVars : dict of string, optional
        A dictionary of model settings whose values should vary from participant to participant based on the
        values found in the imported participant data files. The key is the label given in the participant data file,
        as a string, and the value is the associated label in the model, also as a string. Default ``{}``
    experiments : experiments.experiments, optional
        An experiment factory generating each of the different experiments being considered. Default ``None``

    See Also
    --------
    experiments.experiments : The experiments factory
    models.models : The model factory
    outputting.outputting : The outputting class
    fitAlgs.simMethods.simMethod.simMethod : Abstract class for a method of fitting data
    data.data : Data import function
    """

    if not (isinstance(data, list)): #and isinstance(fitter, fitAlg)):

        logger = outputting.getLogger('dataFitting')
        message = "Data not recognised. "
        logger.warning(message)
        return

    else:
        logger = outputting.getLogger('Overview')

    outputting.logFittingParams(fitter.info())

    message = "Beginning the data fitting"
    logger.info(message)

    modelID = 0

    for modelInfo in models.iterFitting():

        model = modelInfo[0]
        modelInitParamVars = modelInfo[1]
        modelOtherArgs = modelInfo[2]

        if experiments is not None:
            exp = experiments.create(0)
        else:
            exp = None

        for v in partModelVars.itervalues():
            modelOtherArgs[v] = "<Varies for each participant>"

        outputting.logSimFittingParams(model.Name, modelInitParamVars, modelOtherArgs)

        for participant in data:

            partName = participant[partLabel]
            if isinstance(partName, (list, tuple)):
                partName = partName[0]

            for k, v in partModelVars.iteritems():
                modelOtherArgs[v] = participant[k]

            # Find the best model values from those proposed

            message = "Beginning participant fit for participant %s"%(partName)
            logger.info(message)

            modelFitted, fitQuality, fittingData = fitter.participant(model, (modelInitParamVars, modelOtherArgs), participant, exp=exp)

            message = "Participant fitted"
            logger.debug(message)

            if exp is not None:
                expEvolution = exp.outputEvolution()
            else:
                expEvolution = None

            outputting.logModFittedParams(modelInitParamVars, modelFitted.params(), fitQuality, partName)

            outputting.recordParticipantFit(participant,
                                            partName,
                                            modelFitted.outputEvolution(),
                                            str(modelID),
                                            fitQuality,
                                            fittingData,
                                            partModelVars,
                                            expData=expEvolution)

        modelID += 1

    outputting.end()

