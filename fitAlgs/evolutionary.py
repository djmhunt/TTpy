# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from numpy import array, nanargmin
from scipy import optimize
from types import NoneType

from fitAlgs.fitAlg import fitAlg
from utils import callableDetailsString
from fitAlgs.qualityFunc import qualFuncIdent
from fitAlgs.boundFunc import scalarBound


class evolutionary(fitAlg):

    """The class for simMethods data using scipy.optimise.differential_evolution

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the simMethod.
        The value it returns provides the fitter with its simMethods guide.
        Default ``fitAlg.null``
    qualFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    strategy : string or list of strings, optional
        The name of the simMethods strategy or list of names of simMethods strategy or
        name of list of simMethods strategies. Valid names found in the notes.
        Default ``best1bin``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for simMethods. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    boundCostFunc : function, optional
        A function used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound``
    polish : bool, optional
        If True (default), then scipy.optimize.minimize with the ``L-BFGS-B``
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    popSize : int, optional
        A multiplier for setting the total population size. The population has
        popsize * len(x) individuals. Default 15
    tolerance : float, optional
        When the mean of the population energies, multiplied by tol, divided by
        the standard deviation of the population energies is greater than 1 the
        solving process terminates: convergence = mean(pop) * tol / stdev(pop) > 1
        Default 0.01
    extraFitMeasures : dict of dict, optional
        Dictionary of simMethod measures not used to simMethod the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the simMethods strategies
    strategySet : list
        The list of valid simMethods strategies.
        Currently these are: 'best1bin', 'best1exp', 'rand1exp',
        'randtobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin',
        'best2bin', 'rand2bin', 'rand1bin'
        For all strategies, use 'all'

    See Also
    --------
    simMethods.fitAlgs.fitAlg.fitAlg : The general simMethods strategy class, from
                                    which this one inherits
    simMethods.simMethod.simMethod : The general simMethods framework class
    scipy.optimise.differential_evolution : The simMethods class this wraps around

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

    def __init__(self, simMethod, fitQualFunc=None, qualFuncArgs={}, boundCostFunc=scalarBound(), bounds=None, **kwargs):

        self.simMethod = simMethod
        strategy = kwargs.pop("strategy", None)

        self.boundCostFunc = boundCostFunc
        self.allBounds = bounds
        self.fitQualFunc = qualFuncIdent(fitQualFunc, **qualFuncArgs)
        self.calcCovariance = kwargs.pop('calcCov', True)
        self.polish = kwargs.pop("polish", False)
        self.popsize = kwargs.pop("popSize", 15)
        self.tolerence = kwargs.pop("tolerance", 0.01)
        if self.calcCovariance:
            br = kwargs.pop('boundRatio', 0.000001)
            self.hessInc = {k: br * (u - l) for k, (l, u) in self.allBounds.iteritems()}

        measureDict = kwargs.pop("extraFitMeasures", {})
        self.measures = {fitQualFunc: qualFuncIdent(fitQualFunc, **qualFuncArgs) for fitQualFunc, qualFuncArgs in measureDict.iteritems()}

        self._setType(strategy)

        self.fitInfo = {'Name': self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'bounds': self.allBounds,
                        'polish': self.polish,
                        "popSize": self.popsize,
                        "tolerance": self.tolerence
                        }

        if type(self.strategySet) is NoneType:
            self.fitInfo['strategy'] = self.strategy
        else:
            self.fitInfo['strategy'] = self.strategySet

        self.count = 1

        self.boundVals = None
        self.boundNames = None

        self.testedParams = []
        self.testedParamQualities = []
        self.iterbestParams = []
        self.iterConvergence = []

        self.logger = logging.getLogger('Fitting.fitAlgs.evolutionary')

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
            The list of the initial parameters

        Returns
        -------
        fitParams : list of floats
            The best simMethods parameters
        fitQuality : float
            The quality of the simMethod as defined by the quality function chosen.
        testedParams : tuple of two lists and a dictionary
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            simMethod qualities of these parameters. The dictionary contains the parameters and convergence values from each
            iteration, stored in two lists.

        See Also
        --------
        simMethod.fitness

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

        if type(strategy) is NoneType:

            resultSet = []
            strategySuccessSet = []

            for strategy in strategySet:
                optimizeResult = self._strategyFit(strategy, boundVals)
                if type(optimizeResult) is not NoneType:
                    resultSet.append(optimizeResult)
                    strategySuccessSet.append(strategy)
            bestResult = self._bestfit(resultSet)

            if type(bestResult) is NoneType:
                fitParams = mInitialParams
                fitVal = float("inf")
            else:
                fitParams = bestResult.x
                fitVal = bestResult.fun

        else:
            optimizeResult = self._strategyFit(strategy, boundVals)

            if type(optimizeResult) is NoneType:
                fitParams = mInitialParams
                fitVal = float("inf")
            else:
                fitParams = optimizeResult.x
                fitVal = optimizeResult.fun

        fitDetails = dict(optimizeResult)
        fitDetails['bestParams'] = array(self.iterbestParams).T
        fitDetails['convergence'] = self.iterConvergence

        return fitParams, fitVal, (self.testedParams, self.testedParamQualities, fitDetails)

    def callback(self, xk, convergence):
        """
        Used for storing the state after each stage of simMethods

        Parameters
        ----------
        xk : coordinates of best simMethod
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
        simMethods.fitAlgs.fitAlg.fitAlg.fitness : The function called to provide the fitness of parameter sets
        """

        optimizeResult = optimize.differential_evolution(self.fitness,
                                                         bounds,
                                                         strategy=strategy,
                                                         popsize=self.popsize,
                                                         tol=self.tolerence,
                                                         polish=self.polish,
                                                         callback=self.callback,
                                                         init='latinhypercube'  # 'random'
                                                         )

        if optimizeResult.success is True:
            return optimizeResult
        else:
            if optimizeResult.message == 'Maximum number of iterations has been exceeded.':
                message = "Maximum number of simMethods iterations has been exceeded. " \
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
#        print(array([array([o.x[0] for o in resultSet]), array([o.x[1] for o in resultSet]),
        #      array([o.fun for o in resultSet])]).T)
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
