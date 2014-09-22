# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from fit import fit

from itertools import izip
#from numpy import amax, amin
#
#from utils import listMerGen

from scipy import optimize

class fitter(fit):

    """A class for fitting data

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,partParam, modelParam, scaler):

        self.partParam = partParam
        self.modelparam = modelParam
        self.scaler = scaler

    def fitness(self, *modelParameters):

        #Run model with given parameters
        exp, model = self._simSetup(modelParameters)

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoices = modelData[self.modelparam]
        partChoices = self.partChoices

        #Check lengths
        if len(partChoices) != len(modelChoices):
            raise ValueError("The length of the model and participatiant data are different. %s:%s to %s:%s " % (self.partParam,len(partChoices),self.modelparam,len(modelChoices)))

        # Find the difference

        diff = modelChoices - partChoices

        return diff

    def participant(self, exp, model, modelSetup, partData):

        self.exp = exp
        self.model = model
        self.mInitialParams = modelSetup[0].values()
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scaler(partData[self.partParam])

        fitVals, success = optimize.leastsq(self.fitness, self.mInitialParams[:])

        return self._fittedModel(fitVals)

    def _fittedModel(self,fitVals):

        exp, model = self._simSetup(fitVals)

        return model

    def _getModInput(self, modelParameters):

        optional = self.mOtherParams

        inputs = {k : v for k,v in izip(self.mParamNames, modelParameters)}

        for k, v in optional.iteritems():
            inputs[k] = v

        return inputs

    def _simSetup(self, modelParameters):

        args = self._getModInput(modelParameters)

        model = self.model(**args)
        exp = self.exp.reset()

        self._simRun(exp,model)

        return exp, model

    def _simRun(self, exp, model):

        for event in exp:
            model.observe(event)
            act = model.action()
            exp.receiveAction(act)
            response = exp.feedback()
            model.feedback(response)
            exp.procede()


