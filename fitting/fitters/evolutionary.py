# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from fitAlg import fitAlg

from numpy import array, around
from scipy import optimize
from itertools import izip

from utils import callableDetailsString
from qualityFunc import qualFuncIdent
from boundFunc import scalarBound

import pytest

class evolutionary(fitAlg):

    """The class for fitting data using scipy.optimise.differential_evolution

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns proivides the fitter with its fitting guide.
        Default ``fitAlg.null``
    strategy : string or list of strings, optional
        The name of the fitting strategy or list of names of fitting strategy or
        name of list of fitting strategies. Valid names found in the notes.
        Default ``unconstrained``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for fitting. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    polish : bool, optional
        If True (default), then scipy.optimize.minimize with the ``L-BFGS-B``
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.

    Attributes
    ----------
    Name : string
        The name of the fitting strategies
    strategySet : list
        The list of valid fitting strategies

    See Also
    --------
    fitting.fitters.fitAlg.fitAlg : The general fitting strategy class, from
                                    which this one inherits
    fitting.fit.fit : The general fitting framework class
    scipy.optimise.differential_evolution : The fitting class this wraps around

    """

    Name = 'evolutionary'

    validStrategySet = ['best1bin',
                        'best1exp',
                        'rand1exp',
                        'randtobest1exp',
                        'best2exp',
                        'rand2exp',
                        'randtobest1bin',
                        'best2bin',
                        'rand2bin',
                        'rand1bin']


    def __init__(self,fitQualFunc = None, strategy = None, bounds = None, polish = True):

        self.allBounds = bounds
        self.polish = polish

        self.fitQualFunc = qualFuncIdent(fitQualFunc)

        self._setType(strategy)

        self.fitInfo = {'Name':self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'bounds':self.allBounds,
                        'polish': polish
                        }

        if self.strategySet == None:
            self.fitInfo['strategy'] = self.strategy
        else:
            self.fitInfo['strategy'] = self.strategySet

        self.count = 1

        self.boundVals = None
        self.boundNames = None

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

        strategy=self.strategy
        strategySet = self.strategySet

        self.setBounds(mParamNames)
        boundVals = self.boundVals


        if strategy == None:

            resultSet = []
            strategySuccessSet = []

            for strategy in strategySet:

                optimizeResult = self._strategyFit(strategy, boundVals)

                if optimizeResult != None:
                    resultSet.append(optimizeResult)
                    strategySuccessSet.append(strategy)

            bestResult = self._bestfit(resultSet, boundVals)

            if bestResult == None:
                return mInitialParams, float("inf")
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

                return fitParams, fitVal

        else:
            optimizeResult = self._strategyFit(strategy, boundVals)

            fitParams = optimizeResult.x
            fitVal = optimizeResult.fun

            return fitParams, fitVal

    def _strategyFit(self, strategy, bounds):

        optimizeResult = optimize.differential_evolution(self.fitness,
                                                         bounds,
                                                         strategy = strategy,
                                                         polish = self.polish,
                                                         init = 'latinhypercube' # 'random'
                                                         )

        if optimizeResult.success == True:
            return optimizeResult
        else:
            return None

    def _bestfit(self, resultSet):

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
#        print array([array([o.x[0] for o in resultSet]), array([o.x[1] for o in resultSet]), array([o.fun for o in resultSet])]).T
#        pytest.set_trace()

        return resultSet[genFitid]

    def _setType(self,strategy):

        self.strategy = None
        self.strategySet = None
        if isinstance(strategy,list):
            self.strategySet = strategy
        elif strategy in self.validStrategySet:
            self.strategy = strategy
        elif strategy == "all":
            self.strategySet = self.validStrategySet
        else:
            self.strategy = 'best1bin'



