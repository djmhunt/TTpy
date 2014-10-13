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


    Unconstrained method: ‘Nelder-Mead’, ‘Powell’, ‘CG’, ‘BFGS’,
    Constrained methods: ‘L-BFGS-B’, ‘TNC’, ‘SLSQP’
    custom

    """

    name = 'minimise'

    unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
    constrained = ['L-BFGS-B','TNC','SLSQP']


    def __init__(self,dataShaper = None, method = None, bounds = None):

        if dataShaper == "-2log":
            self.fitness = self.logprob
        else:
            self.fitness = self.null

        self._setType(method,bounds)

        self.fitInfo = {'name':self.name,
                        'shaper': dataShaper,
                        'bounds':self.bounds
                        }

        if self.methodSet == None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet

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

    def _setType(self,method,bounds):

        self.method = None
        self.methodSet = None
        self.bounds = None
        if isinstance(method,list):
            self.methodSet = method
            self.bounds = bounds
        elif method in self.unconstrained:
            self.method = method
        elif method in self.constrained:
            self.method = method
            self.bounds = bounds
        elif callable(method):
            self.method = method
            self.bounds = bounds
        elif method == 'constrained':
            self.methodSet = self.constrained
            self.bounds = bounds
        elif method == 'unconstrained':
            self.methodSet = self.unconstrained
        else:
            self.methodSet = self.unconstrained
