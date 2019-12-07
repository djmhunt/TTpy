# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np
import scipy as sp

import itertools

import fitAlgs.fitAlg.FitAlg as FitAlg


class Basinhopping(FitAlg):

    """The class for fitting data using scipy.optimise.basinhopping

    Parameters
    ----------
    fitQualityFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fit guide.
        Default ``fitAlg.null``
    qualityFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    method : string or list of strings, optional
        The name of the fitting method or list of names of fitting methods or
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
    calcCov : bool, optional
        Is the covariance calculated. Default ``True``
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
    fitAlgs.fitAlg.fitAlg : The general fitting method class, from which this one inherits
    filtAlgs.fitSims.fitSim : The general fitting class
    scipy.optimise.basinhopping : The fitting class this wraps around
    """

    unconstrained = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
    constrained = ['L-BFGS-B', 'TNC', 'SLSQP']

    def __init__(self, method=None, numStartPoints=4, boundFit=True, boundSensitivity=5, **kwargs):

        super(Basinhopping, self).__init__(**kwargs)

        self.numStartPoints = numStartPoints
        self.boundFit = boundFit
        self.boundSensitivity = boundSensitivity

        self._setType(method, self.allBounds)

        self.fitInfo['numStartPoints'] = self.numStartPoints
        self.fitInfo['boundFit'] = self.boundFit
        self.fitInfo['boundSensitivity'] = self.boundSensitivity

        if self.methodSet is None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet

        self.iterBestParams = []
        self.iterFuncValueMin = []
        self.iterParameterAccept = []

    def callback(self, x, f, accept):
        """
        Used for storing the state after each stage of fitter

        Parameters
        ----------
        x : coordinates of the trial minimum
        f : function value of the trial minimum
        accept : whether or not that minimum was accepted
        """

        self.iterBestParams.append(x)
        self.iterFuncValueMin.append(f)
        self.iterParameterAccept.append(accept)

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters
        and returns the best one.

        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for
            given model parameters. One example is fitAlgs.fitAlg.fitness
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
        testedParams : tuple of two lists and a dictionary
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters. The dictionary contains the coordinates of the trial minimum, the
            function value of the trial minimum and whether or not that minimum was accepted. Each is stored in a list.

        See Also
        --------
        fitAlgs.fitAlg.fitness
        """

        self.sim = sim
        self.testedParams = []
        self.testedParamQualities = []
        self.iterBestParams = []
        self.iterFuncValueMin = []
        self.iterParameterAccept = []

        method = self.method
        methodSet = self.methodSet
        bounds = self.bounds
        boundVals = self.boundVals
        boundFit = self.boundFit
        boundSensitivity = self.boundSensitivity
        numStartPoints = self.numStartPoints

        if bounds is None:
            boundVals = [(0, float('Inf')) for i in mInitialParams]
            bounds = {k: v for k, v in itertools.izip(mParamNames, boundVals)}
            self.bounds = bounds
            self.boundVals = np.array(boundVals)

        if boundVals is None:
            boundVals = np.array([bounds[k] for k in mParamNames])
            self.boundVals = boundVals

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
                return mInitialParams, float("inf")
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

                return fitParams, fitVal

        else:
            optimizeResult = self._methodFit(method, initParamSets, boundVals, boundFit=boundFit)

            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

            iterDetails = dict(bestParams=self.iterBestParams, funcVal=self.iterFuncValueMin, paramAccept=self.iterParameterAccept)

            return fitParams, fitVal, (self.testedParams, self.testedParamQualities, iterDetails)

    def _methodFit(self, method, initParamSets, bounds, boundFit=True, boundSensitivity=5):

        resultSet = []

        boundFunc = self._bounds

        for i in initParamSets:

            optimizeResult = sp.optimize.basinhopping(self.fitQualFunc, i[:],
                                                   accept_test=boundFunc,
                                                   callback=self.callback
 #                                                  minimizer_kwargs={'method': method,
 #                                                                    'bounds': bounds}
 #                                                                    }
                                                  )

            resultSet.append(optimizeResult)

        bestResult = self._bestfit(resultSet, bounds, boundFit=boundFit, boundSensitivity=boundSensitivity)

        return bestResult

    def _bestfit(self, resultSet, bounds, boundFit=True, boundSensitivity=5):

        # Check that there are fits
        if len(resultSet) == 0:
            return None

        genFitVal, genFitid = min((r.fun, idx) for (idx, r) in enumerate(resultSet))

        # Debug code
#        data = {}
#        data["fitVal"] = np.array([o.fun for o in resultSet])
#        data['nIter'] = np.array([o.nit for o in resultSet])
#        data['parameters'] = np.array([o.x for o in resultSet])
#        data['nfev'] = np.array([o.nfev for o in resultSet])
#        data['message'] = np.array([o.message for o in resultSet])
#        data['minimization_failures'] = np.array([o.minimization_failures for o in resultSet])
#        print(np.array([data['parameters'].T[0], data['parameters'].T[1], data["fitVal"]]).T)
#        pytest.set_trace()

        # If boundary fits are acceptable
        if boundFit:
            return resultSet[genFitid]

        else:
            reducedResults = []
            for r in resultSet:
                invalid = [1 for fitVal, boundVals in itertools.izip(r.x, bounds) if any(np.around(fitVal-boundVals, boundSensitivity) == 0)]

                if 1 not in invalid:
                    reducedResults.append(r)

            if len(reducedResults) == 0:
                return resultSet[genFitid]

            else:
                fitVal, fitid = min((r.fun, idx) for (idx, r) in enumerate(reducedResults))

                return reducedResults[fitid]

    def _setType(self, method, bounds):

        self.method = None
        self.methodSet = None
        self.bounds = None
        if isinstance(method, list):
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

    def _bounds(self, **kwargs):
        """
        Based on http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.optimize.basinhopping.html
        """
        boundArr = self.boundVals
        x = kwargs["x_new"]
        tmax = bool(np.all(x < boundArr[:, 1]))
        tmin = bool(np.all(x > boundArr[:, 0]))

        return tmax and tmin



