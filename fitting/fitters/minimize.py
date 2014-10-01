# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from fitAlg import fitAlg

from scipy import optimize
from numpy import log

class minimize(fitAlg):

    """The abstact class for fitting data

    fitAlg()


    method:

    ‘Nelder-Mead’
    ‘Powell’
    ‘CG’
    ‘BFGS’
    Constrained methods:
    ‘L-BFGS-B’
    ‘TNC’
    ‘COBYLA’
    ‘SLSQP’
    custom

    """


    def __init__(self,dataShaper = None, method = None, bounds = None):

        unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
        constrained = ['L-BFGS-B','TNC','COBYLA','SLSQP']

        if dataShaper == "-2log":
            self.fitness = self.logprob
        else:
            self.fitness = self.null

        self.method = None
        self.methodSet = None
        self.bounds = None
        if isinstance(method,list):
            self.methodSet = method
            self.bounds = bounds
        elif method in unconstrained:
            self.method = method
        elif method in constrained:
            self.method = method
            self.bounds = bounds
        elif isinstance(method,('function','instancemethod')):
            self.method = method
            self.bounds = bounds
        elif method == 'constrained':
            self.methodSet = constrained
            self.bounds = bounds
        elif method == 'unconstrained':
            self.methodSet = unconstrained
        else:
            self.methodSet = unconstrained

    def null(self,*params):

        modVals = self.sim(*params)

        return sum(modVals)

    def logprob(self,*params):

        modVals = self.sim(*params)

        logModCoiceprob = log(modVals)

        fit = -2*logModCoiceprob

        return sum(fit)

    def fit(self, sim, mInitialParams):

        self.sim = sim

        method=self.method
        methodSet = self.methodSet
        bounds = self.bounds

        if method == None:

            fitParamSet = []
            fitValSet = []
            methodSuccessSet = []

            for method in methodSet:

                optimizeResult = optimize.minimize(self.fitness, mInitialParams[:], method=method, bounds=bounds)

                if optimizeResult.success == True:
                    fitParamSet.append(optimizeResult.x)
                    fitValSet.append(optimizeResult.fun)
                    methodSuccessSet.append(method)

            fitVal, fitid = min((v, idx) for (idx, v) in enumerate(fitValSet))

            return fitParamSet[fitid]

        else:
            optimizeResult = optimize.minimize(self.fitness, mInitialParams[:], method=method, bounds=bounds)

            fitParams = optimizeResult.x

            return fitParams
