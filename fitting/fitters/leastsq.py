# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from fitAlg import fitAlg

from scipy import optimize
from numpy import log

from utils import callableDetailsString
from qualityFunc import qualFuncIdent

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
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4

    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitting.fitters.fitAlg.fitAlg : The general fitting method class, from
                                    which this one inherits
    fitting.fit.fit : The general fitting framework class
    scipy.optimise.leastsq : The fitting class this wraps around

    """

    Name = 'leastsq'

    def __init__(self,fitQualFunc = None, numStartPoints = 4):

        self.numStartPoints = numStartPoints

        self.fitQualFunc = qualFuncIdent(fitQualFunc)

        self.fitInfo = {'Name':self.Name,
                        'fitQualityFunction': fitQualFunc}


    def fit(self, sim, mParamNames, mInitialParams):

        self.sim = sim

        fitParams, success = optimize.leastsq(self.fitness, mInitialParams[:])

        return fitParams

