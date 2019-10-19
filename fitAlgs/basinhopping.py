# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from fitAlgs.fitAlg import fitAlg

from numpy import array, around, all
from scipy import optimize
from itertools import izip

from utils import callableDetailsString
from fitAlgs.qualityFunc import qualFuncIdent
from fitAlgs.boundFunc import scalarBound


class basinhopping(fitAlg):

    """The class for simMethods data using scipy.optimise.basinhopping

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the simMethod.
        The value it returns proivides the fitter with its simMethods guide.
        Default ``fitAlg.null``
    qualFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    method : string or list of strings, optional
        The name of the simMethods method or list of names of simMethods method or
        name of list of simMethods methods. Valid names found in the notes.
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
        minimal difference) between a simMethod value and its related boundaries
        before a simMethod value is considered different from a boundary. The default
        is `5`. This is only valid if ``boundFit`` is ``False``
    calcCov : bool, optional
        Is the covariance calculated. Default ``True``
    extraFitMeasures : dict of dict, optional
        Dictionary of simMethod measures not used to simMethod the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the simMethods method
    unconstrained : list
        The list of valid unconstrained simMethods methods
    constrained : list
        The list of valid constrained simMethods methods


    Notes
    -----
    unconstrained = ['Nelder-Mead','Powell','CG','BFGS']
    constrained = ['L-BFGS-B','TNC','SLSQP']
    Custom simMethods algorithms are also allowed in theory, but it has yet to be
    implemented.

    For each simMethods function a set of different starting parameters will be
    tried. These are the combinations of all the values of the different
    parameters. For each starting parameter provided a set of numStartPoints
    starting points will be chosen, surrounding the starting point provided. If
    the starting point provided is less than one it will be assumed that the
    values cannot exceed 1, otherwise, unless otherwise told, it will be
    assumed that they can take any value and will be chosen to be eavenly
    spaced around the provided value.

    See Also
    --------
    simMethods.fitAlgs.fitAlg.fitAlg : The general simMethods method class, from
                                    which this one inherits
    simMethods.simMethod.simMethod : The general simMethods framework class
    scipy.optimise.basinhopping : The simMethods class this wraps around
    scipy.optimise.minimize : The simMethods class basinhopping wraps around

    """

    Name = 'basinhopping'

    unconstrained = ['Nelder-Mead', 'Powell', 'CG', 'BFGS']
    constrained = ['L-BFGS-B', 'TNC', 'SLSQP']

    def __init__(self, simMethod, fitQualFunc=None, qualFuncArgs={}, boundCostFunc=scalarBound(), bounds=None, **kwargs):

        self.simMethod = simMethod

        method = kwargs.pop("method", None)

        self.boundCostFunc = boundCostFunc
        self.allBounds = bounds
        self.numStartPoints = kwargs.pop("numStartPoints", 4)
        self.fitQualFunc = qualFuncIdent(fitQualFunc, **qualFuncArgs)
        self.boundFit = kwargs.pop("boundFit", True)
        self.boundSensitivity = kwargs.pop("boundSensitivity", 5)
        self.calcCovariance = kwargs.pop('calcCov', True)
        if self.calcCovariance:
            br = kwargs.pop('boundRatio', 0.000001)
            self.hessInc = {k: br * (u - l) for k, (l, u) in self.allBounds.iteritems()}

        measureDict = kwargs.pop("extraFitMeasures", {})
        self.measures = {fitQualFunc: qualFuncIdent(fitQualFunc, **qualFuncArgs) for fitQualFunc, qualFuncArgs in measureDict.iteritems()}

        self._setType(method, bounds)

        self.fitInfo = {'Name': self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'bounds': self.bounds,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'numStartPoints': self.numStartPoints,
                        'boundFit': self.boundFit,
                        'boundSensitivity': self.boundSensitivity
                        }

        if self.methodSet is None:
            self.fitInfo['method'] = self.method
        else:
            self.fitInfo['method'] = self.methodSet

        self.boundVals = None

        self.testedParams = []
        self.testedParamQualities = []
        self.iterbestParams = []
        self.iterfuncValMin = []
        self.iterparamAccept = []

        self.logger = logging.getLogger('Fitting.fitAlgs.basinhopping')

    def callback(self, x, f, accept):
        """
        Used for storing the state after each stage of simMethods

        Parameters
        ----------
        x : coordinates of the trial minimum
        f : function value of the trial minimum
        accept : whether or not that minimum was accepted
        """

        self.iterbestParams.append(x)
        self.iterfuncValMin.append(f)
        self.iterparamAccept.append(accept)

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the simMethods algorithms and starting parameters
        and returns the best one.

        Parameters
        ----------
        sim : function
            The function used by a simMethods algorithm to generate a simMethod for
            given model parameters. One example is simMethod.fitness
        mParamNames : list of strings
            The list of initial parameter names
        mInitialParams : list of floats
            The list of the intial parameters

        Returns
        -------
        fitParams : list of floats
            The best simMethods parameters
        fitQuality : float
            The quality of the simMethod as defined by the quality function chosen.
        testedParams : tuple of two lists and a dictionary
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            simMethod qualities of these parameters. The dictionary contains the coordinates of the trial minimum, the
            function value of the trial minimum and whether or not that minimum was accepted. Each is stored in a list.

        See Also
        --------
        simMethod.fitness
        """

        self.sim = sim
        self.testedParams = []
        self.testedParamQualities = []
        self.iterbestParams = []
        self.iterfuncValMin = []
        self.iterparamAccept = []

        method = self.method
        methodSet = self.methodSet
        bounds = self.bounds
        boundVals = self.boundVals
        boundFit = self.boundFit
        boundSensitivity = self.boundSensitivity
        numStartPoints = self.numStartPoints

        if bounds is None:
            boundVals = [(0, float('Inf')) for i in mInitialParams]
            bounds = {k: v for k, v in izip(mParamNames, boundVals)}
            self.bounds = bounds
            self.boundVals = array(boundVals)

        if boundVals is None:
            boundVals = array([bounds[k] for k in mParamNames])
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

            iterDetails = dict(bestParams=self.iterbestParams, funcVal=self.iterfuncValMin, paramAccept=self.iterparamAccept)

            return fitParams, fitVal, (self.testedParams, self.testedParamQualities, iterDetails)

    def _methodFit(self, method, initParamSets, bounds, boundFit=True, boundSensitivity=5):

        resultSet = []

        boundFunc = self._bounds

        for i in initParamSets:

            optimizeResult = optimize.basinhopping(self.fitQualFunc, i[:],
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
#        data["fitVal"] = array([o.fun for o in resultSet])
#        data['nIter'] = array([o.nit for o in resultSet])
#        data['parameters'] = array([o.x for o in resultSet])
#        data['nfev'] = array([o.nfev for o in resultSet])
#        data['message'] = array([o.message for o in resultSet])
#        data['minimization_failures'] = array([o.minimization_failures for o in resultSet])
#        print(array([data['parameters'].T[0], data['parameters'].T[1], data["fitVal"]]).T)
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
        tmax = bool(all(x < boundArr[:,1]))
        tmin = bool(all(x > boundArr[:,0]))

        return tmax and tmin



