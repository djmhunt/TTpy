# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import logging

from fit import fit

from itertools import izip
from numpy import log, concatenate, array
from numpy import sum as asum
#
#from utils import listMerGen

class fitter(fit):

    """A class for fitting data by passing the participant data through the model

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,partChoiceParam, partRewardParam, modelParam, fitAlg, scaler):

        self.partChoiceParam = partChoiceParam
        self.partRewardParam = partRewardParam
        self.modelparam = modelParam
        self.fitAlg = fitAlg
        self.scaler = scaler

        self.fitInfo = {'name':self.name,
                        'participantChoiceParam':partChoiceParam,
                        'participantRewardParam':partRewardParam,
                        'modelParam':modelParam,
                        'scalerEffect': self._scalerEffect()}

    def fitness(self, *modelParameters):
        """ Returns the value necessary for the fitting
        """

        #Run model with given parameters
        model = self._simSetup(*modelParameters[0])

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoiceProbs = modelData[self.modelparam]#"ActionProb"

        return modelChoiceProbs

    def participant(self, exp, model, modelSetup, partData):

        self.model = model
        self.mInitialParams = modelSetup[0].values()
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scaler(partData[self.partChoiceParam])

        self.partRewards = partData[self.partRewardParam]

        fitVals = self.fitAlg.fit(self.fitness, self.mInitialParams[:])

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
        """ Initialises the model for the running of the 'simulation'
        """

        args = self._getModInput(*modelParameters)

        model = self.model(**args)

        self._simRun(model)

        return model

    def _simRun(self, model):
        """ Simulates the events of a simulation from the perspective of a model
        """

        parAct = self.partChoices
        parReward = self.partRewards

        for action, reward in izip(parAct, parReward):

            model.currAction = action
            model._update(reward,'reac')
            model._storeState()




