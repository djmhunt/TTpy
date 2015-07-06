# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from fitAlg import fitAlg

from numpy import array, around
from scipy.optimize import basinhopping
from itertools import izip

from utils import callableDetailsString
from qualityFunc import qualFuncIdent
from boundFunc import scalarBound

import pytest

class basinhopping(fitAlg):

    """The class for fitting data using scipy.optimise.basinhopping

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns proivides the fitter with its fitting guide.
        Default ``fitAlg.null``
    method : string or list of strings, optional
        The name of the fitting method or list of names of fitting method or
        name of list of fitting methods. Valid names found in the notes.
        Default ``unconstrained``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for methods that use bounds. If unbounded methods are
        specified then the bounds will be ignored. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    boundCostFunc : function, optional
        A function used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound``
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4
    boundFit : bool, optional
        Defines if fits that reach a boundary should be considered the same way
        as those that do not. Default is True
    boundSensitivity : int, optional
        Defines the smallest number of decimal places difference (so the
        minimal difference) between a fit value and its related boundaries
        before a fit value is considered different from a boundary. The default
        is `5`. This is only valid if ``boundFit`` is ``False``

    Attributes
    ----------
    Name : string
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
    scipy.optimise.basinhopping : The fitting class this wraps around
    scipy.optimise.minimize : The fitting class basinhopping wraps around

    """

    Name = 'basinhopping'

    unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
    constrained = ['L-BFGS-B','TNC','SLSQP']


    def __init__(self,fitQualFunc = None, method = None, bounds = None, boundCostFunc = scalarBound(), numStartPoints = 4, boundFit = True, boundSensitivity = 5):

        self.numStartPoints = numStartPoints
        self.boundFit = boundFit
        self.boundSensitivity = boundSensitivity

        self.fitness = qualFuncIdent(fitQualFunc)
        self.boundCostFunc = boundCostFunc

        self._setType(method,bounds)

        self.fitInfo = {'Name':self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'bounds':self.bounds,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'numStartPoints' : self.numStartPoints,
                        'boundFit' : self.boundFit,
                        'boundSensitivity' : self.boundSensitivity
                        }

        if self.methodSet == None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet

        self.count = 1

        self.boundVals = None

#    def callback(self,Xi):
#        """
#        Used for printing state after each stage of fitting
#        """
#
#        print '{0:4d}: {1:s}'.format(self.count, Xi)
#
#        self.count += 1

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters
        and returns the best one.

        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for
            given model parameters. One example is fit.fitness
        mParamNames : list of strings
            The list of initial parameter names
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
        boundVals = self.boundVals
        boundFit = self.boundFit
        boundSensitivity = self.boundSensitivity
        numStartPoints = self.numStartPoints

        if bounds == None:
            boundVals = [(0,float('Inf')) for i in mInitialParams]
            bounds = {k : v for k, v in izip(mParamNames, boundVals)}
            self.bounds = bounds
            self.boundVals = boundVals

        if boundVals == None:
            boundVals = [ bounds[k] for k in mParamNames]
            self.boundVals = boundVals

        initParamSets = self.startParams(mInitialParams, bounds = boundVals, numPoints = numStartPoints)

        if method == None:

            resultSet = []
            methodSuccessSet = []

            for method in methodSet:

                optimizeResult = self._methodFit(method, initParamSets, boundVals, boundFit = boundFit)

                if optimizeResult != None:
                    resultSet.append(optimizeResult)
                    methodSuccessSet.append(method)

            bestResult = self._bestfit(resultSet, boundVals, boundFit = boundFit, boundSensitivity = boundSensitivity)

            if bestResult == None:
                return mInitialParams, float("inf")
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

                return fitParams, fitVal

        else:
            optimizeResult = self._methodFit(method, initParamSets, boundVals, boundFit = boundFit)

            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

            return fitParams, fitVal

    def _methodFit(self,method, initParamSets, bounds, boundFit = True, boundSensitivity = 5):

        resultSet = []

        for i in initParamSets:

            optimizeResult = basinhopping(self.fitness, i[:],
                                          minimizer_kwargs = {method: method,
                                                              bounds: bounds})#,
#                                                             callback: self.callback})
            self.count = 1

            if optimizeResult.success == True:
                resultSet.append(optimizeResult)

        bestResult = self._bestfit(resultSet, bounds, boundFit = boundFit, boundSensitivity = boundSensitivity)

        return bestResult

    def _bestfit(self, resultSet, bounds, boundFit = True, boundSensitivity = 5):

        # Check that there are fits
        if len(resultSet) == 0:
            return None

        genFitVal, genFitid = min((r.fun, idx) for (idx, r) in enumerate(resultSet))

        # Debug code
#        data = {}
#        data["fitVal"] = array([o.fun for o in resultSet])
#        data['nIter'] = array([o.nit for o in resultSet])
#        data['parameters'] = array([o.x for o in resultSet])
#        data['success'] = array([o.success for o in resultSet])
#        data['nfev'] = array([o.nfev for o in resultSet])
#        data['message'] = array([o.message for o in resultSet])
#        data['jac'] = array([o.jac for o in resultSet])
#        print array([data['parameters'].T[0], data['parameters'].T[1], data["fitVal"]]).T
#        pytest.set_trace()

        # If boundary fits are acceptable
        if boundFit:
            return resultSet[genFitid]

        else:
            reducedResults = []
            for r in resultSet:
                invalid = [1 for fitVal, boundVals in izip(r.x,bounds) if any(around(fitVal-boundVals,boundSensitivity)==0)]

                if 1 not in invalid:
                    reducedResults.append(r)

            if len(reducedResults) == 0:
                return resultSet[genFitid]

            else:
                fitVal, fitid = min((r.fun, idx) for (idx, r) in enumerate(reducedResults))

                return reducedResults[fitid]


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



