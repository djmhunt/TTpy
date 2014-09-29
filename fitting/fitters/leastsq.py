# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from fitAlg import fitAlg

from scipy import optimize
from numpy import log

class leastsq(fitAlg):

    """Fits data based on the least squared

    fitters(partParam, modelParam, scaler)

    """


    def __init__(self,dataShaper = None):

        if dataShaper == "-2log":
            self.fitness = self.logprob
        else:
            self.fitness = self.null

    def null(self,*params):

        modVals = self.sim(*params)

        return modVals

    def logprob(self,*params):

        modVals = self.sim(*params)

        logModCoiceprob = log(modVals)

        fit = -2*logModCoiceprob

        return fit

    def fit(self, sim, mInitialParams):

        self.sim = sim

        fitParams, success = optimize.leastsq(self.fitness, mInitialParams[:])

        return fitParams

