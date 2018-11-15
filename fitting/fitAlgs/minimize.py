# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from fitting.fitAlgs.fitAlg import fitAlg

from numpy import array, around, nanargmin
from scipy import optimize
from itertools import izip

from utils import callableDetailsString
from fitting.fitAlgs.qualityFunc import qualFuncIdent
from fitting.fitAlgs.boundFunc import scalarBound

import pytest


class minimize(fitAlg):

    """The class for fitting data using scipy.optimise.minimize

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns proivides the fitter with its fitting guide.
        Default ``fitAlg.null``
    qualFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
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
    extraFitMeasures : dict of dict, optional
        Dictionary of fit measures not used to fit the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

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
    fitting.fitAlgs.fitAlg.fitAlg : The general fitting method class, from
                                    which this one inherits
    fitting.fit.fit : The general fitting framework class
    scipy.optimise.minimize : The fitting class this wraps around

    """

    Name = 'minimise'

    unconstrained = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
    constrained = ['L-BFGS-B', 'TNC', 'SLSQP']

    def __init__(self, modFit, fitQualFunc=None, qualFuncArgs={}, boundCostFunc=scalarBound(), bounds=None, **kwargs):

        self.modFit = modFit

        method = kwargs.pop("method", None)

        self.boundCostFunc = boundCostFunc
        self.allBounds = bounds
        self.numStartPoints = kwargs.pop("numStartPoints", 4)
        self.fitQualFunc = qualFuncIdent(fitQualFunc, **qualFuncArgs)
        self.boundFit = kwargs.pop("boundFit", True)
        self.boundSensitivity = kwargs.pop("boundSensitivity", 5)

        measureDict = kwargs.pop("extraFitMeasures", {})
        self.measures = {fitQualFunc: qualFuncIdent(fitQualFunc, **qualFuncArgs) for fitQualFunc, qualFuncArgs in measureDict.iteritems()}

        self._setType(method, bounds)

        self.fitInfo = {'Name': self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'bounds': self.allBounds,
                        'numStartPoints': self.numStartPoints,
                        'boundFit': self.boundFit,
                        'boundSensitivity': self.boundSensitivity
                        }

        if self.methodSet is None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet

        self.count = 1

        self.boundVals = None
        self.boundNames = None

        self.testedParams = []
        self.testedParamQualities = []

        self.logger = logging.getLogger('Fitting.fitAlgs.minimize')

#    def callback(self,Xi):
#        """
#        Used for printing state after each stage of fitting
#        """
#
#        print('{0:4d}: {1:s}'.format(self.count, Xi))
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
            The list of the initial parameters

        Returns
        -------
        fitParams : list of floats
            The best fitting parameters
        fitQuality : float
            The quality of the fit as defined by the quality function chosen.
        testedParams : tuple of two lists
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.

        See Also
        --------
        fit.fitness
        """

        self.sim = sim
        self.testedParams = []
        self.testedParamQualities = []

        method = self.method
        methodSet = self.methodSet
        boundFit = self.boundFit
        boundSensitivity = self.boundSensitivity
        numStartPoints = self.numStartPoints

        self.setBounds(mParamNames)
        boundVals = self.boundVals

        initParamSets = self.startParams(mInitialParams, bounds=boundVals, numPoints=numStartPoints)

        if method is None:

            resultSet = []
            methodSuccessSet = []

            for method in methodSet:

                optimizeResult = self._methodFit(method, initParamSets, boundVals, boundFit=boundFit)

                if optimizeResult is not None:
                    resultSet.append(optimizeResult)
                    methodSuccessSet.append(method)

            bestResult = self._bestfit(resultSet, boundVals, boundFit=boundFit, boundSensitivity=boundSensitivity)

            if bestResult is None:
                return mInitialParams, float("inf"), (self.testedParams, self.testedParamQualities)
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

                return fitParams, fitVal, (self.testedParams, self.testedParamQualities)

        else:
            optimizeResult = self._methodFit(method, initParamSets, boundVals, boundFit=boundFit)

            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

            return fitParams, fitVal, (self.testedParams, self.testedParamQualities)

    def _methodFit(self, method, initParamSets, bounds, boundFit=True, boundSensitivity=5):

        resultSet = []

        for i in initParamSets:

            optimizeResult = optimize.minimize(self.fitness, i[:],
                                               method=method,
                                               bounds=bounds)  # ,
        #                                      callback= self.callback )
            self.count = 1

            if optimizeResult.success is True:
                resultSet.append(optimizeResult)

        bestResult = self._bestfit(resultSet, bounds, boundFit=boundFit, boundSensitivity=boundSensitivity)

        return bestResult

    def _bestfit(self, resultSet, bounds, boundFit=True, boundSensitivity=5):

        # Check that there are fits
        if len(resultSet) == 0:
            return None

        genFitid = nanargmin([r.fun for r in resultSet])

        # Debug code
#        data = {}
#        data["fitVal"] = array([o.fun for o in resultSet])
#        data['nIter'] = array([o.nit for o in resultSet])
#        data['parameters'] = array([o.x for o in resultSet])
#        data['success'] = array([o.success for o in resultSet])
#        data['nfev'] = array([o.nfev for o in resultSet])
#        data['message'] = array([o.message for o in resultSet])
#        data['jac'] = array([o.jac for o in resultSet])
#        print(array([data['parameters'].T[0], data['parameters'].T[1], data["fitVal"]]).T)
#        print(array([array([o.x[0] for o in resultSet]), array([o.x[1] for o in resultSet]), array([o.fun for o in resultSet])]).T)
#        pytest.set_trace()

        # If boundary fits are acceptable
        if boundFit:
            return resultSet[genFitid]

        else:
            reducedResults = []
            for r in resultSet:
                invalid = [1 for fitVal, boundVals in izip(r.x, bounds) if any(around(fitVal-boundVals, boundSensitivity) == 0)]

                if 1 not in invalid:
                    reducedResults.append(r)

            if len(reducedResults) == 0:
                return resultSet[genFitid]

            else:
                fitid = nanargmin([r.fun for r in reducedResults])

                return reducedResults[fitid]

    def _setType(self, method, bounds):

        self.method = None
        self.methodSet = None
        self.allBounds = None
        if isinstance(method, list):
            self.methodSet = method
            self.allBounds = bounds
        elif method in self.unconstrained:
            self.method = method
        elif method in self.constrained:
            self.method = method
            self.allBounds = bounds
        elif callable(method):
            self.method = method
            self.allBounds = bounds
        elif method == 'constrained':
            self.methodSet = self.constrained
            self.allBounds = bounds
        elif method == 'unconstrained':
            self.methodSet = self.unconstrained
        else:
            self.methodSet = self.unconstrained
