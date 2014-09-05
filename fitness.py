# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from itertools import izip
from numpy import log, concatenate, array
from numpy import sum as asum
#
#from utils import listMerGen

from scipy import optimize

class fitter(object):

    """A class for fitting data

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,partParam, modelParam, scaler):

        self.partParam = partParam
        self.modelparam = modelParam
        self.scaler = scaler

    def fitness(self, *modelParameters):

        #Run model with given parameters
        model = self._simSetup(*modelParameters[0])

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoiceProbs = modelData["ActionProb"]

        logModCoiceprob = log(modelChoiceProbs)

        fit = -2*logModCoiceprob

        return fit

    def participant(self, exp, model, modelSetup, partData):

        self.model = model
        self.mInitialParams = modelSetup[0].values()
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scaler(partData[self.partParam])
        partCumRewards = partData["cumpts"]
        self.partRewards = concatenate((partCumRewards[0:1],partCumRewards[1:]-partCumRewards[:-1]))

        fitVals, success = optimize.leastsq(self.fitness, self.mInitialParams[:])

        return self._fittedModel(*fitVals)

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

        parAct = self.partChoices
        parReward = self.partRewards

        for action, reward in izip(parAct, parReward):

            model.currAction = action
            model._update(reward,'reac')
            model._storeState()




