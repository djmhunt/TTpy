# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from fitting.fitAlgs.fitAlg import fitAlg

from scipy import optimize
from numpy import log

from utils import callableDetailsString
from fitting.fitAlgs.qualityFunc import qualFuncIdent
from fitting.fitAlgs.boundFunc import scalarBound


class leastsq(fitAlg):
    """
    Fits data based on the least squared optimizer

    Not properly developed and will not be documented until upgrade

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns proivides the fitter with its fitting guide.
        Default ``fitAlg.null``
    qualFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4
    extraFitMeasures : dict of dict, optional
        Dictionary of fit measures not used to fit the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitting.fitAlgs.fitAlg.fitAlg : The general fitting method class, from
                                    which this one inherits
    fitting.fit.fit : The general fitting framework class
    scipy.optimise.leastsq : The fitting class this wraps around

    """

    Name = 'leastsq'

    def __init__(self, fitQualFunc=None, qualFuncArgs={}, boundCostFunc=scalarBound(), bounds=None, **kwargs):

        self.numStartPoints = kwargs.pop("numStartPoints", 4)
        self.fitQualFunc = qualFuncIdent(fitQualFunc, **qualFuncArgs)
        self.boundCostFunc = boundCostFunc
        self.allBounds = bounds

        measureDict = kwargs.pop("extraFitMeasures", {})
        self.measures = {fitQualFunc: qualFuncIdent(fitQualFunc, **qualFuncArgs) for fitQualFunc, qualFuncArgs in measureDict.iteritems()}

        self.fitInfo = {'Name':self.Name,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'bounds': self.allBounds,
                        'fitQualityFunction': fitQualFunc}

        self.testedParams = []
        self.testedParamQualities = []

        self.logger = logging.getLogger('Fitting.fitAlgs.leastsq')

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

        fitParams, success = optimize.leastsq(self.fitness, mInitialParams[:])

        return fitParams, 0, (self.testedParams, self.testedParamQualities)

