# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from itertools import izip

class fit(object):

    """The abstact class for fitting data

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,partParam, modelParam, fitAlg, scaler):

        self.partParam = partParam
        self.modelparam = modelParam
        self.fitAlg = fitAlg
        self.scaler = scaler

    def fitness(self, *modelParameters):

        return 0

    def participant(self, exp, model, modelSetup, partData):

        self.model = model
        self.mInitialParams = modelSetup[0].values()
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        return self._fittedModel(self.mInitialParams)

    def _fittedModel(self,*fitVals):

        model = self._simSetup(*fitVals)

        return model

    def _getModInput(self, *modelParameters):

        optional = self.mOtherParams

        inputs = {k : v for k,v in izip(self.mParamNames, modelParameters)}

        for k, v in optional.iteritems():
            inputs[k] = v

        return inputs

    def _simSetup(self, *modelParameters):

        args = self._getModInput(*modelParameters)

        model = self.model(**args)

        self._simRun(model)

        return model

    def _simRun(self, model):

        pass

