# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
from __future__ import division

from fitAlg import fitAlg

from scipy import optimize
from numpy import log2, linspace
from math import isinf
from itertools import izip
from utils import listMergeNP

import pytest

class minimize(fitAlg):

    """The class for fitting data using scipy.optimise.minimize

    minimize(dataShaper = None, method = None, bounds = None)


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
        elif dataShaper == "1-prob":
            self.fitness = self.maxprob
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
            
        self.count = 1

    def null(self,*params):

        modVals = self.sim(*params)

        return sum(modVals)

    def logprob(self,*params):

        modVals = self.sim(*params)

        logModCoiceprob = log2(modVals)

        probs = -2*logModCoiceprob
        
        fit = sum(probs)

        return fit
        
    def maxprob(self,*params):
        """Used to maximise the probability, so in this case minimise the 
        difference between 1 and the probability"""
        
        modVals = self.sim(*params)
        
        fit = sum(1-modVals)
        
        return fit
        
    def callback(self,Xi):
        
        print '{0:4d}: {1:s}'.format(self.count, Xi)
        
        self.count += 1

    def fit(self, sim, mInitialParams):

        self.sim = sim

        method=self.method
        methodSet = self.methodSet
        bounds = self.bounds
        
        initParamSets = self._setStartParams(mInitialParams, numPoints = 4)

        if method == None:

            fitParamSet = []
            fitValSet = []
            methodSuccessSet = []

            for method in methodSet:
                
                optimizeResult = self._methodFit(method, initParamSets, bounds)

                if optimizeResult != None:
                    fitParamSet.append(optimizeResult.x)
                    fitValSet.append(optimizeResult.fun)
                    methodSuccessSet.append(method)

            if len(fitValSet) == 0:
                return mInitialParams, float("inf")

            fitVal, fitid = min((v, idx) for (idx, v) in enumerate(fitValSet))

            return fitParamSet[fitid], fitVal

        else:
            optimizeResult = self._methodFit(method, initParamSets, bounds)

            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

            return fitParams, fitVal
            
    def _methodFit(self,method, initParamSets, bounds):
        
        fitValSet = []
        resultSet = []
#        data = {"initAlpha":[],
#                "initTheta":[],
#                "fitVal":[],
#                "nIter":[],
#                "alpha":[],
#                "theta":[],
#                "success":[], 
#                "nfev":[], 
#                "message":[],
#                "jacAlpha":[], 
#                "jacTheta":[]}
        for i in initParamSets:
        
            optimizeResult = optimize.minimize(self.fitness, i[:], 
                                               method=method, 
                                               bounds=bounds)#,  
#                                               callback= self.callback )
            self.count = 1
            
            if optimizeResult.success == True:
                fitValSet.append(optimizeResult.fun)
                resultSet.append(optimizeResult)
#            o = optimizeResult
#            data["initAlpha"].append(i[0])
#            data["initTheta"].append(i[1])
#            data["fitVal"].append(o.fun)
#            data['nIter'].append(o.nit)
#            data['alpha'].append(o.x[0]) 
#            data['theta'].append(o.x[1])
#            data['success'].append(o.success) 
#            data['nfev'].append(o.nfev)
#            data['message'].append(o.message) 
#            data['jacAlpha'].append(o.jac[0])
#            data['jacTheta'].append(o.jac[1])
#        pytest.set_trace()
                
        if len(resultSet) == 0:
            return None
        
        fitVal, fitid = min((v, idx) for (idx, v) in enumerate(fitValSet))
        
        return resultSet[fitid]


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
            
    def _setStartParams(self,initialParams, numPoints = 3):
        """ Defines a list of different starting parameters to run the minimization over"""

        if self.bounds == None:
            # We only have the values passed in as the starting parameters
            startLists = (self._startParamList(i, numPoints = numPoints) for i in initialParams)

        else: 
            startLists = (self._startParamList(i, bMax, numPoints) for i, (bMin, bMax) in izip(initialParams,self.bounds))
            
        startSets = listMergeNP(*startLists)
            
        return startSets
            
            
    def _startParamList(self,initial, bMax = float('Inf'), numPoints = 3):
        """Assumes that intial parameters are positive and all values will 
        be above zero"""
    
         #The number of initial points per parameter
        divVal = (numPoints+1)/2
        
        # We can assume that any initial parameter proposed has the 
        #correct order of magnitude. 
        vMin = initial / divVal
        
        
        if bMax == None or isinf(bMax):
            # We can also assume any number smaller than one should stay 
            #smaller than one.
            if initial < 1:
                valAbsMax = 1
            else:
                valAbsMax = float('inf')
        else:
            valAbsMax = bMax
            
        if numPoints*vMin > valAbsMax:
            inc = (valAbsMax - initial) / divVal
            vMin = valAbsMax - numPoints * inc 
            vMax = valAbsMax - inc
        else:
            vMax = vMin * numPoints
            
           
        points = linspace(vMin, vMax, numPoints)
        
        return points
