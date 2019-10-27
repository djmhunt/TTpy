# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from scipy import optimize
from itertools import izip
from numpy import array

from utils import callableDetailsString
from fitAlgs.fitAlg import fitAlg
from fitAlgs.qualityFunc import qualFuncIdent
from fitAlgs.boundFunc import scalarBound


class leastsq(fitAlg):
    """
    Fits data based on the least squared optimizer scipy.optimize.least_squares

    Not properly developed and will not be documented until upgrade

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fitting guide.
        Default ``fitAlg.null``
    qualFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for fitting. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    method : {‘trf’, ‘dogbox’, ‘lm’}, optional
        Algorithm to perform minimization. Default ``dogbox``
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4
    calcCov : bool, optional
        Is the covariance calculated. Default ``True``
    extraFitMeasures : dict of dict, optional
        Dictionary of fit measures not used to fit the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitAlgs.fitAlg.fitAlg : The general fitting method class, from which this one inherits
    fitAlgs.fitSims.fitSim : The general fitting class
    scipy.optimize.least_squares : The fitting class this wraps around

    """

    Name = 'leastsq'

    def __init__(self, fitSim, fitQualFunc=None, qualFuncArgs={}, boundCostFunc=scalarBound(), bounds=None, **kwargs):

        self.fitSim = fitSim

        self.numStartPoints = kwargs.pop("numStartPoints", 4)
        self.fitQualFunc = qualFuncIdent(fitQualFunc, **qualFuncArgs)
        self.method = kwargs.pop('method', "dogbox")
        self.jacmethod = kwargs.pop('jac', '3-point')
        self.boundCostFunc = boundCostFunc
        self.allBounds = bounds
        self.calcCovariance = kwargs.pop('calcCov', True)
        if self.calcCovariance:
            br = kwargs.pop('boundRatio', 0.000001)
            self.hessInc = {k: br * (u - l) for k, (l, u) in self.allBounds.iteritems()}

        measureDict = kwargs.pop("extraFitMeasures", {})
        self.measures = {fitQualFunc: qualFuncIdent(fitQualFunc, **qualFuncArgs) for fitQualFunc, qualFuncArgs in measureDict.iteritems()}

        self.fitInfo = {'Name': self.Name,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'method': self.method,
                        'jac': self.jacmethod,
                        'bounds': self.allBounds,
                        'fitQualityFunction': fitQualFunc}

        self.testedParams = []
        self.testedParamQualities = []

        self.logger = logging.getLogger('Fitting.fitAlgs.leastsq')

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters and returns the best one.

        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for given model parameters. One example is
            fitAlg.fitness
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
        fitAlgs.fitAlg.fitness
        """

        self.sim = sim
        self.testedParams = []
        self.testedParamQualities = []

        bounds = [i for i in izip(*self.boundVals)]

        optimizeResult = optimize.least_squares(self.fitness,
                                                mInitialParams[:],
                                                method=self.method,
                                                jac=self.jacmethod,
                                                bounds=bounds)

        if optimizeResult.success is False and optimizeResult.status == -1:
            fitParams = mInitialParams
            fitVal = float("inf")
        else:
            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

        if optimizeResult.status == 0:
            message = "Maximum number of fitting evaluations has been exceeded. " \
                      "Returning the best results found so far: "
            message += "Params " + str(fitParams)
            message += " Fit quality " + str(fitVal)
            self.logger.info(message)

        fitDetails = dict(optimizeResult)
        fitDetails['bestParams'] = array(self.iterbestParams).T
        fitDetails['convergence'] = self.iterConvergence

        return fitParams, fitVal, (self.testedParams, self.testedParamQualities, fitDetails)
