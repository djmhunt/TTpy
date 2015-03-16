# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from fitAlg import fitAlg

from scipy import optimize

from itertools import izip
from utils import listMergeNP

import pytest

class minimize(fitAlg):

    """The class for fitting data using scipy.optimise.minimize

    Parameters
    ----------
    fitQualFunc : function, optional
        The function used to calculate the quality of the fit. The value it 
        returns proivides the fitter with its fitting guide. Default ``fitAlg.null``
    method : string or list of strings, optional
        The name of the fitting method or list of names of fitting method or
        name of list of fitting methods. Valid names found in the notes.
        Default ``unconstrained``
    bounds : tuple of length two, optional
        The boundaries for methods that use bounds. If unbounded methods are
        specified then the bounds will be ignored. Default is ``(0,float('Inf'))``
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4
        
    Attributes
    ----------
    Name: string
        The name of the fitting method    
    unconstrained : list
        The list of valid unconstrained fitting methods
    constrained : list
        The list of valid constrained fitting methods


    Notes
    -----
    unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
    constrained = ['L-BFGS-B','TNC','SLSQP']
    Custom fitting algorithms are also allowed in theory, but it has yet to be 
    implemented.
    
    For each fitting function a set of different starting parameters will be 
    tried. These are the combinations of all the values of the different 
    parameters. For each starting parameter provided a set of numStartPoints 
    starting points will be chosen, surrounding the starting point provided. If
    the starting point provided is less than one it will be assumed that the 
    values cannot exceed 1, otherwise, unless otherwise told, it will be 
    assumed that they can take any value and will be chosen to be eavenly 
    spaced around the provided value.
    
    See Also
    --------
    fitting.fitters.fitAlg.fitAlg : The general fitting method class, from 
                                    which this one inherits
    fitting.fit.fit : The general fitting framework class
    scipy.optimise.minimize : The fitting class this wraps around

    """

    Name = 'minimise'

    unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
    constrained = ['L-BFGS-B','TNC','SLSQP']


    def __init__(self,fitQualFunc = None, method = None, bounds = (0,float('Inf')), numStartPoints = 4):
        
        self.numStartPoints = numStartPoints

        if fitQualFunc == "-2log":
            self.fitness = self.logprob
        elif fitQualFunc == "1-prob":
            self.fitness = self.maxprob
        else:
            self.fitness = self.null

        self._setType(method,bounds)

        self.fitInfo = {'Name':self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'bounds':self.bounds,
                        'numStartPoints' : self.numStartPoints
                        }

        if self.methodSet == None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet
            
        self.count = 1
        
#    def callback(self,Xi):
#        """
#        Used for printing state after each stage of fitting
#        """
#        
#        print '{0:4d}: {1:s}'.format(self.count, Xi)
#        
#        self.count += 1

    def fit(self, sim, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters 
        and returns the best one.
        
        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for 
            given model parameters. One example is fit.fitness
        mInitialParams : list of floats
            The list of the intial parameters
            
        Returns
        -------
        fitParams : list of floats
            The best fitting parameters
        fitQuality : float
            The quality of the fit as defined by the quality function chosen.
            
        See Also
        --------
        fit.fitness
        
        """

        self.sim = sim

        method=self.method
        methodSet = self.methodSet
        bounds = self.bounds
        numStartPoints = self.numStartPoints
        
        initParamSets = self._setStartParams(mInitialParams, numPoints = numStartPoints)

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
            startLists = (self.startParamList(i, numPoints = numPoints) for i in initialParams)

        else: 
            startLists = (self.startParamList(i, bMax, numPoints) for i, (bMin, bMax) in izip(initialParams,self.bounds))
            
        startSets = listMergeNP(*startLists)
            
        return startSets
            
            
    
