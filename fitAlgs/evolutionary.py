# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np
import scipy as sp

from fitAlgs.fitAlg import FitAlg


class Evolutionary(FitAlg):

    """The class for fitting data using scipy.optimise.differential_evolution

    Parameters
    ----------
    fitQualityFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fitting guide.
        Default ``fitAlg.null``
    qualityFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    strategy : string or list of strings, optional
        The name of the fitting strategy or list of names of fitting strategies or
        name of a list of fitting strategies. Valid names found in the notes.
        Default ``best1bin``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for fitting. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    boundCostFunc : function, optional
        A function used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound``
    polish : bool, optional
        If True (default), then scipy.optimize.minimize with the ``L-BFGS-B``
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. Default ``False``
    popSize : int, optional
        A multiplier for setting the total population size. The population has
        popsize * len(x) individuals. Default 20
    tolerance : float, optional
        When the mean of the population energies, multiplied by tol, divided by
        the standard deviation of the population energies is greater than 1 the
        solving process terminates: convergence = mean(pop) * tol / stdev(pop) > 1
        Default 0.01
    calcCov : bool, optional
        Is the covariance calculated. Default ``False``
    extraFitMeasures : dict of dict, optional
        Dictionary of fitting measures not used to fitting the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the fitting strategies
    strategySet : list
        The list of valid fitting strategies.
        Currently these are: 'best1bin', 'best1exp', 'rand1exp',
        'randtobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin',
        'best2bin', 'rand2bin', 'rand1bin'
        For all strategies, use 'all'

    See Also
    --------
    fitAlgs.fitAlg.FitAlg : The general fitting strategy class, from which this one inherits
    fitAlgs.fitSims.FitSim : The general class for seeing how a parameter combination perform
    scipy.optimise.differential_evolution : The fitting method this wraps around

    """

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

    def __init__(self, strategy=None, polish=False, popSize=20, tolerance=0.01, **kwargs):

        super(Evolutionary, self).__init__(**kwargs)

        self.polish = polish
        self.populationSize = popSize
        self.tolerance = tolerance

        self._setType(strategy)

        self.fitInfo['polish'] = self.polish
        self.fitInfo['popSize'] = self.populationSize
        self.fitInfo['tolerance'] = self.tolerance

        if self.strategySet is None:
            self.fitInfo['strategy'] = self.strategy
        else:
            self.fitInfo['strategy'] = self.strategySet

        self.iterbestParams = []
        self.iterConvergence = []

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
            The list of the initial parameters

        Returns
        -------
        fitParams : list of floats
            The best fitting parameters
        fitQuality : float
            The quality of the fit as defined by the quality function chosen.
        testedParams : tuple of two lists and a dictionary
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters. The dictionary contains the parameters and convergence values from each
            iteration, stored in two lists.

        See Also
        --------
        fitAlgs.fitAlg.fitness

        """

        self.sim = sim
        self.testedParams = []
        self.testedParamQualities = []
        self.iterbestParams = []
        self.iterConvergence = []

        strategy = self.strategy
        strategySet = self.strategySet

        self.setBounds(mParamNames)
        boundVals = self.boundVals

        if strategy is None:

            resultSet = []
            strategySuccessSet = []

            for strategy in strategySet:
                optimizeResult = self._strategyFit(strategy, boundVals)
                if optimizeResult is not None:
                    resultSet.append(optimizeResult)
                    strategySuccessSet.append(strategy)
            bestResult = self._bestfit(resultSet)

            if bestResult is None:
                fitParams = mInitialParams
                fitVal = float("inf")
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

        else:
            optimizeResult = self._strategyFit(strategy, boundVals)

            if optimizeResult is None:
                fitParams = mInitialParams
                fitVal = float("inf")
            else:
                fitParams = optimizeResult.x
                fitVal = optimizeResult.fun

        fitDetails = dict(optimizeResult)
        fitDetails['bestParams'] = np.array(self.iterbestParams).T
        fitDetails['convergence'] = self.iterConvergence

        return fitParams, fitVal, (self.testedParams, self.testedParamQualities, fitDetails)

    def callback(self, xk, convergence):
        """
        Used for storing the state after each stage of fitting

        Parameters
        ----------
        xk : coordinates of best fit
        convergence : the proportion of the points from the iteration that have converged
        """

        self.iterbestParams.append(xk)
        self.iterConvergence.append(convergence)

    def _strategyFit(self, strategy, bounds):
        """

        Parameters
        ----------
        strategy : str
            The name of the chosen strategy
        bounds : list of length 2 tuples containing floats
            The bounds for each parameter being looked at

        Returns
        -------
        optimizeResult : None or scipy.optimize.optimize.OptimizeResult instance

        See Also
        --------
        fitAlgs.fitAlg.fitAlg.fitness : The function called to provide the fitness of parameter sets
        """

        try:
            optimizeResult = sp.optimize.differential_evolution(self.fitness,
                                                                bounds,
                                                                strategy=strategy,
                                                                popsize=self.populationSize,
                                                                tol=self.tolerance,
                                                                polish=self.polish,
                                                                callback=self.callback,
                                                                init='latinhypercube'  # 'random'
                                                                )
        except RuntimeError as e:
            self.logger.warn("{} in evolutionary fitting. Retrying to run it: {} - {}".format(type(e), str(e), e.args))

            #Try it one last time
            optimizeResult = sp.optimize.differential_evolution(self.fitness,
                                                                bounds,
                                                                strategy=strategy,
                                                                popsize=self.populationSize,
                                                                tol=self.tolerance,
                                                                polish=self.polish,
                                                                callback=self.callback,
                                                                init='latinhypercube'  # 'random'
                                                                )

        if optimizeResult.success is True:
            return optimizeResult
        else:
            if optimizeResult.message == 'Maximum number of iterations has been exceeded.':
                message = "Maximum number of fitting iterations has been exceeded. " \
                          "Returning the best results found so far: "
                message += "Params " + str(optimizeResult.x)
                message += " Fit quality " + str(optimizeResult.fun)
                self.logger.info(message)
                return optimizeResult
            else:
                return None

    def _bestfit(self, resultSet):

        # Check that there are fits
        if len(resultSet) == 0:
            return None

        genFitid = np.nanargmin([r.fun for r in resultSet])

        # Debug code
#        data = {}
#        data["fitVal"] = np.array([o.fun for o in resultSet])
#        data['nIter'] = np.array([o.nit for o in resultSet])
#        data['parameters'] = np.array([o.x for o in resultSet])
#        data['success'] = np.array([o.success for o in resultSet])
#        data['nfev'] = np.array([o.nfev for o in resultSet])
#        data['message'] = np.array([o.message for o in resultSet])
#        data['jac'] = np.array([o.jac for o in resultSet])
#        print(np.array([data['parameters'].T[0], data['parameters'].T[1], data["fitVal"]]).T)
#        print(np.array([np.array([o.x[0] for o in resultSet]), np.array([o.x[1] for o in resultSet]),
        #      np.array([o.fun for o in resultSet])]).T)
#        pytest.set_trace()

        return resultSet[genFitid]

    def _setType(self, strategy):

        self.strategy = None
        self.strategySet = None
        if isinstance(strategy, list):
            self.strategySet = strategy
        elif strategy in self.validStrategySet:
            self.strategy = strategy
        elif strategy == "all":
            self.strategySet = self.validStrategySet
        else:
            self.strategy = 'best1bin'
