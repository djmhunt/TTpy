# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from scipy import optimize

class fit(object):

    """A class for fitting data

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,partParam, modelParam, scaler):

        self.partParam = partParam
        self.modelparam = modelParam
        self.scaler = scaler

    def fitness(self, *modelParameters):

        return None

    def participant(self, exp, model, modelSetup, partData):

        return None

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





