# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import logging

import numpy as np
import scipy as sp

import itertools

from fitAlgs.fitAlg import FitAlg


class Leastsq(FitAlg):
    """
    Fits data based on the least squared optimizer scipy.optimize.least_squares

    Not properly developed and will not be documented until upgrade

    Parameters
    ----------
    fit_sim : fitAlgs.fitSims.FitSim instance, optional
        An instance of one of the fitting simulation methods. Default ``fitAlgs.fitSims.FitSim``
    fit_measure : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fitting guide. Default ``-loge``
    fit_measure_args : dict, optional
        The parameters used to initialise fit_measure and extra_fit_measures. Default ``None``
    extra_fit_measures : list of strings, optional
        List of fit measures not used to fit the model, but to provide more information. Any arguments needed for these
        measures should be placed in fit_measure_args. Default ``None``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for methods that use bounds. If unbounded methods are
        specified then the bounds will be ignored. Default is ``None``, which
        translates to boundaries of (0, np.inf) for each parameter.
    boundary_excess_cost : str or callable returning a function, optional
        The function is used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound()``
    boundary_excess_cost_properties : dict, optional
        The parameters for the boundary_excess_cost function. Default {}
    method : {‘trf’, ‘dogbox’, ‘lm’}, optional
        Algorithm to perform minimization. Default ``dogbox``

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

    def __init__(self, method="dogbox", jacobian_method='3-point', **kwargs):

        super(Leastsq, self).__init__(**kwargs)

        self.method = method
        self.jacobian_method = jacobian_method

        self.fit_info['method'] = self.method
        self.fit_info['jacobian_method'] = self.jacobian_method

    def fit(self, simulator, model_parameter_names, model_initial_parameters):
        """
        Runs the model through the fitting algorithms and starting parameters and returns the best one.

        Parameters
        ----------
        simulator : function
            The function used by a fitting algorithm to generate a fit for given model parameters. One example is
            fitAlg.fitness
        model_parameter_names : list of strings
            The list of initial parameter names
        model_initial_parameters : list of floats
            The list of the initial parameters

        Returns
        -------
        fitParams : list of floats
            The best fitting parameters
        fit_quality : float
            The quality of the fit as defined by the quality function chosen.
        testedParams : tuple of two lists
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.

        See Also
        --------
        fitAlgs.fitAlg.fitness
        """

        self.simulator = simulator
        self.tested_parameters = []
        self.tested_parameter_qualities = []

        bounds = [i for i in zip(*self.boundary_values)]

        optimizeResult = sp.optimize.least_squares(self.fitness,
                                                   model_initial_parameters[:],
                                                   method=self.method,
                                                   jac=self.jacobian_method,
                                                   bounds=bounds)

        if optimizeResult.success is False and optimizeResult.status == -1:
            best_fit_parameters = model_initial_parameters
            fit_quality = float("inf")
        else:
            best_fit_parameters = optimizeResult.x
            fit_quality = optimizeResult.fun

        if optimizeResult.status == 0:
            message = "Maximum number of fitting evaluations has been exceeded. " \
                      "Returning the best results found so far: "
            message += "Params " + str(best_fit_parameters)
            message += " Fit quality " + str(fit_quality)
            self.logger.info(message)

        fitDetails = dict(optimizeResult)
        fitDetails['bestParams'] = np.array(self.iterbestParams).T
        fitDetails['convergence'] = self.iterConvergence

        return best_fit_parameters, fit_quality, (self.tested_parameters, self.tested_parameter_qualities, fitDetails)
