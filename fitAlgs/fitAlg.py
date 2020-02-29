# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import math
import collections
import itertools

import numpy as np
import scipy as sp
#import numdifftools as nd

from fitAlgs.fitSims import FitSim
from fitAlgs import qualityFunc
from fitAlgs import boundFunc

import utils


class FitAlg(object):
    """
    The abstract class for fitting data

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
    boundary_excess_cost : basestring or callable returning a function, optional
        The function is used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound()``
    boundary_excess_cost_properties : dict, optional
        The parameters for the boundary_excess_cost function. Default {}
    calculate_covariance : bool, optional
        Is the covariance calculated. Default ``False``


    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitAlgs.fitSims.fitSim : The general fitting class

    """

    def __init__(self,
                 fit_sim=None, fit_measure='-loge', fit_measure_args=None, extra_fit_measures=None,
                 bounds=None, boundary_excess_cost=None, boundary_excess_cost_properties=None, bound_ratio=0.000001,
                 calculate_covariance=False, **kwargs):

        if fit_sim is None:
            self.fit_sim = FitSim()
        elif isinstance(fit_sim, FitSim):
            self.fit_sim = fit_sim
        else:
            raise NameError("fitSim type is incorrect: {}".format(type(fit_sim)))

        if bounds is None:
            raise NameError("Please specify bounds for your parameters")
        else:
            self.boundaries = bounds

        if fit_measure_args is None:
            fit_measure_args = {}

        if extra_fit_measures is None:
            extra_fit_measures = []

        self.Name = self.find_name()

        if callable(boundary_excess_cost):
            if boundary_excess_cost_properties is not None:
                boundary_excess_cost_kwargs = {k: v for k, v in kwargs.iteritems()
                                                    if k in boundary_excess_cost_properties}
            else:
                boundary_excess_cost_kwargs = kwargs.copy()
            self.boundary_excess_cost = boundary_excess_cost(**boundary_excess_cost_kwargs)
        elif isinstance(boundary_excess_cost, basestring):
            boundary_excess_cost_function = utils.find_function(boundary_excess_cost, 'fitAlgs', excluded_files=['fit'])
            boundary_excess_cost_kwargs = {k: v for k, v in kwargs.iteritems()
                                                if k in utils.getFuncArgs(boundary_excess_cost_function)}
            self.boundary_excess_cost = boundary_excess_cost_function(**boundary_excess_cost_kwargs)
        else:
            self.boundary_excess_cost = boundFunc.scalarBound()

        self.fit_quality_function = qualityFunc.qualFuncIdent(fit_measure, **fit_measure_args.copy())
        self.calculate_covariance = calculate_covariance
        if self.calculate_covariance:
            self.hessInc = {k: bound_ratio * (u - l) for k, (l, u) in self.boundaries.iteritems()}

        self.measures = {m: qualityFunc.qualFuncIdent(m, **fit_measure_args.copy()) for m in extra_fit_measures}

        self.fit_info = {'Name': self.Name,
                         'fit_measure_function': fit_measure,
                         'fit_measure_arguments': fit_measure_args,
                         'boundary_cost_function': utils.callableDetailsString(boundary_excess_cost),
                         'bounds': self.boundaries,
                         'extra_fit_measures': extra_fit_measures,
                         'calculate_covariance': calculate_covariance,
                         'bound_ratio': bound_ratio,
                         'FitSim': self.fit_sim.info()}

        self.boundary_values = None
        self.boundary_names = None

        self.tested_parameters = []
        self.tested_parameter_qualities = []

        self.logger = logging.getLogger(self.Name)

    def __repr__(self):

        return repr(self.info())

    def find_name(self):
        """
        Returns the name of the class
        """

        return self.__class__.__name__

    def participant(self, model, model_parameters, model_properties, participant_data):
        """
        Fit participant data to a model for a given task

        Parameters
        ----------
        model : model.modelTemplate.Model inherited class
            The model you wish to try and fit values to
        model_parameters : dict
            The model initial parameters
        model_properties : dict
            The model static properties
        participant_data : dict
            The participant data

        Returns
        -------
        model : model.modelTemplate.Model inherited class instance
            The model with the best fit parameters
        fit_quality : float
            Specifies the fit quality for this participant to the model
        fitting_data : tuple of OrderedDict and list
            They are an ordered dictionary containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.
        """

        sim = self.fit_sim.prepare_sim(model, model_parameters, model_properties, participant_data)

        model_initial_parameters = model_parameters.values()
        model_parameter_names = model_parameters.keys()

        best_fit_parameters, fit_quality, fit_info = self.fit(sim, model_parameter_names, model_initial_parameters[:])

        model_run = self.fit_sim.fitted_model(*best_fit_parameters)

        fit_measures = self.extra_measures(*best_fit_parameters)

        testedParamDict = collections.OrderedDict([(key, val[0]) for key, val in itertools.izip(model_parameter_names, np.array(fit_info[0]).T)])

        fitting_data = {"tested_parameters": testedParamDict,
                        "fit_qualities": fit_info[1],
                        "fit_quality": fit_quality,
                        "final_parameters": collections.OrderedDict([(key, val) for key, val in itertools.izip(model_parameter_names, best_fit_parameters)])}

        fitting_data.update({"fit_quality_" + k: v for k, v in fit_measures.iteritems()})

        if self.calculate_covariance:
            covariance = self.covariance(model_parameter_names, best_fit_parameters, fit_info[2])
            covdict = ({"fit_quality_cov_{}_{}".format(p1, p2): c for p1, cr in itertools.izip(model_parameter_names, covariance)
                                                                 for p2, c in itertools.izip(model_parameter_names, cr)})
            fitting_data.update(covdict)

        try:
            fitting_data.update(fit_info[2])
        finally:
            return model_run, fit_quality, fitting_data

    def fit(self, simulator, model_parameter_names, model_initial_parameters):
        """
        Runs the model through the fitting algorithms and starting parameters
        and returns the best one. This is the abstract version that always
        returns ``(0,0)``

        Parameters
        ----------
        simulator : function
            The function used by a fitting algorithm to generate a fit for
            given model parameters. One example is ``fitAlgs.fitAlg.fitness``
        model_parameter_names : list of strings
            The list of initial parameter names
        model_initial_parameters : list of floats
            The list of the initial parameters

        Returns
        -------
        best_fit_parameters : list of floats
            The best fitting parameters
        fit_quality : float
            The quality of the fit as defined by the quality function chosen.
        tested_parameters : tuple of two lists
            The two lists are a list containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.

        See Also
        --------
        fitAlgs.fitAlg.fitness

        """
        # TODO : consider passing model_parameter_names and model_initial_parameters as one parameter

        self.simulator = simulator
        self.tested_parameters = []
        self.tested_parameter_qualities = []
        best_fit_parameters = 0
        fit_quality = np.inf

        return best_fit_parameters, fit_quality, (self.tested_parameters, self.tested_parameter_qualities)

    def fitness(self, *params):
        """
        Generates a fit quality value used by the fitting function. This is the function passed to the fitting function.

        Parameters
        ----------
        *params : array of floats
            The parameters proposed by the fitting algorithm

        Returns
        -------
        fit_quality : float
            The fit quality value calculated using the fitQualFunc function

        See Also
        --------
        fitAlgs.qualityFunc : the module of fitQualFunc functions
        fitAlg.invalidParams : Checks if the parameters are valid and if not returns ``inf``
        fitAlgs.fitSims.fitSim.fitness : Runs the model simulation and returns the values used to calculate the fit quality

        """
        # This is because the fitting functions return an array and we want a list
        pms = list(*params)

        # Start by checking that the parameters are valid
        if self.invalid_parameters(*pms):
            pseudo_fit_quality = self.boundary_excess_cost(pms, self.boundary_values, self.fit_quality_function)
            return pseudo_fit_quality

        # Run the simulation with these parameters
        modVals = self.simulator(*pms)

        fit_quality = self.fit_quality_function(modVals)

        self.tested_parameters.append(params)
        self.tested_parameter_qualities.append(fit_quality)

        return fit_quality

    def extra_measures(self, *model_parameter_values):
        """

        Parameters
        ----------
        *model_parameter_values : array of floats
            The parameters proposed by the fitting algorithm

        Returns
        -------
        fit_quality : dict of float
            The fit quality value calculated using the fit quality functions described in extraMeasures

        """

        modVals = self.simulator(*model_parameter_values)

        measureVals = {}
        for m, f in self.measures.iteritems():

            fit_quality = f(modVals)
            measureVals[m] = fit_quality

        return measureVals

    def covariance(self, model_parameter_names, paramvals, fitinfo):
        """
        The covariance at a point

        Parameters
        ----------
        paramvals : array or list
            The parameters at which the
        fitinfo : dict
            The

        Returns
        -------
        covariance : float
            The covariance at the point paramvals

        """

        if 'hess_inv' in fitinfo:
            cov = fitinfo['hess_inv']
        elif 'hess' in fitinfo:
            hess = fitinfo['hess']
            cov = np.linalg.inv(hess)
        #elif 'jac' in fitinfo:
        #    jac = fitinfo['jac']
        #    cov = covariance(jac)
        else:
            inc = [self.hessInc[p] for p in model_parameter_names]
            # TODO : Check if this is correct or replace it with other methods
            jac = np.expand_dims(sp.optimise.approx_fprime(paramvals, self.fitness, inc), axis=0)
            #cov = covariance(jac)
            cov = np.linalg.inv(np.dot(jac.T, jac))

        return cov

    def info(self):
        """
        The information relating to the fitting method used

        Includes information on the fitting algorithm used

        Returns
        -------
        info : (dict,dict)
            The fitSims info and the fitAlgs.fitAlg info

        See Also
        --------
        fitAlg.fitSims.fitSim.info
        """

        return self.fit_info

    def set_bounds(self, model_parameter_names):
        """
        Checks if the bounds have changed

        Parameters
        ----------
        model_parameter_names : list of strings
            An ordered list of the names of the parameters to be fitted

        Examples
        --------
        >>> a = FitAlg(bounds={1: (0, 5), 2: (0, 2), 3: (-1, 1)})
        >>> a.allBounds
        {1: (0, 5), 2: (0, 2), 3: (-1, 1)}
        >>> a.set_bounds([])
        >>> a.allBounds
        {1: (0, 5), 2: (0, 2), 3: (-1, 1)}
        >>> a.boundNames
        []
        >>> a.set_bounds([3,1])
        >>> a.boundVals
        [(-1, 1), (0, 5)]
        >>> a.set_bounds([2,1])
        >>> a.boundVals
        [(0, 2), (0, 5)]
        """

        bounds = self.boundaries
        boundNames = self.boundary_names
        boundVals = self.boundary_values

        # Check if the bounds have changed or should be added
        if boundNames:
            changed = False
            for m, b in itertools.izip(model_parameter_names, boundNames):
                if m != b:
                    changed = True
                    break
        else:
            changed = True

        # If they have not, then we can leave
        if not changed:
            if len(model_parameter_names) == len(boundNames):
                return

        # If no bounds were defined
        if not bounds:
            boundVals = [(0, float('Inf')) for i in model_parameter_names]
            self.boundaries = {k: v for k, v in itertools.izip(model_parameter_names, boundVals)}
            self.boundary_names = model_parameter_names
            self.boundary_values = boundVals
        else:
            self.boundary_values = [bounds[k] for k in model_parameter_names]
            self.boundary_names = model_parameter_names

        return

    @classmethod
    def startParams(cls, initial_parameters, bounds=None, number_starting_points=3):
        """
        Defines a list of different starting parameters to run the minimization
        over

        Parameters
        ----------
        initial_parameters : list of floats
            The initial starting values proposed
        bounds : list of tuples of length two with floats, optional
            The boundaries for methods that use bounds. If unbounded methods are
            specified then the bounds will be ignored. Default is ``None``, which
            translates to boundaries of (0,float('Inf')) for each parameter.
        number_starting_points : int
            The number of starting parameter values to be calculated around
            each initial point

        Returns
        -------
        startParamSet : list of list of floats
            The generated starting parameter combinations

        See Also
        --------
        FitAlg.start_parameter_values : Used in this function

        Examples
        --------
        >>> FitAlg.startParams([0.5,0.5], number_starting_points=2)
        array([[0.33333333, 0.33333333],
               [0.66666667, 0.33333333],
               [0.33333333, 0.66666667],
               [0.66666667, 0.66666667]])
        """

        if bounds is None:
            # We only have the values passed in as the starting parameters
            startLists = (cls.start_parameter_values(i, number_starting_points=number_starting_points) for i in initial_parameters)

        else:
            if len(bounds) != len(initial_parameters):
                raise ValueError('Bounds do not fit the number of initial parameters', str(len(bounds)), str(len(initial_parameters)))

            startLists = (cls.start_parameter_values(i, boundary_min=bMin, boundary_max=bMax, number_starting_points=number_starting_points)
                          for i, (bMin, bMax) in itertools.izip(initial_parameters, bounds))

        startSets = utils.listMergeNP(*startLists)

        return startSets

    @staticmethod
    def start_parameter_values(initial, boundary_min=float('-Inf'), boundary_max=float('Inf'), number_starting_points=3):
        """
        Provides a set of starting points

        Parameters
        ----------
        initial : float
            The initial starting value proposed
        boundary_min : float, optional
            The minimum value of the parameter. Default is ``float('-Inf')``
        boundary_max : float, optional
            The maximum value of the parameter. Default is ``float('Inf')``
        number_starting_points : int
            The number of starting parameter values to be calculated around the inital
            point

        Returns
        -------
        startParams : list of floats
            The generated starting parameters

        Notes
        -----
        For each starting parameter provided a set of numStartPoints
        starting points will be chosen, surrounding the starting point provided. If
        the starting point provided is less than one but greater than zero it
        will be assumed that the values cannot leave those bounds, otherwise,
        unless otherwise told, it will be assumed that they can take any
        positive value and will be chosen to be eavenly spaced around the
        provided value.

        Examples
        --------
        >>> FitAlg.start_parameter_values(0.5)
        array([0.25, 0.5 , 0.75])
        >>> FitAlg.start_parameter_values(5)
        array([2.5, 5. , 7.5])
        >>> FitAlg.start_parameter_values(-5)
        array([2.5, 5. , 7.5])
        >>> FitAlg.start_parameter_values(5, boundary_min = 0, boundary_max = 7)
        array([4., 5., 6.])
        >>> FitAlg.start_parameter_values(5, boundary_min = -3, boundary_max = 30)
        array([1., 5., 9.])
        >>> FitAlg.start_parameter_values(5, boundary_min = 0, boundary_max = 30)
        array([2.5, 5. , 7.5])
        >>> FitAlg.start_parameter_values(5, boundary_min = 3, boundary_max = 30, number_starting_points = 7)
        array([3.5, 4., 4.5, 5., 5.5, 6., 6.5])
        """

#        initialAbs = abs(initial)

        # The number of initial points per parameter
        divVal = (number_starting_points + 1) / 2

        if boundary_max is None or math.isinf(boundary_max):
            # We can also assume any number smaller than one should stay
            # smaller than one.
            if initial < 1 and initial > 0:
                valMax = 1
            else:
                valMax = float('inf')
        else:
            valMax = boundary_max

        if boundary_min is None or math.isinf(boundary_min):
            # We can also assume any number larger than one should stay
            # bigger than zero.
            if initial > 0:
                valMin = 0

                initialAbs = initial
                valAbsMax = valMax

            else:
                # this should never happen, but regardless
                valMin = 0

                initialAbs = abs(initial)
                valAbsMax = abs(valMax) + initialAbs

        else:
            valMin = boundary_min

            initialAbs = initial - valMin
            valAbsMax = valMax - valMin

        # Now that the bounds have been set we have shifted the space to
        # calculate the points in the space and then shift them back.

        # We can assume that any initial parameter proposed has the
        # correct order of magnitude.
        vMin = initialAbs / divVal

        if number_starting_points*vMin > valAbsMax:
            inc = (valAbsMax - initialAbs) / divVal
            vMin = valAbsMax - number_starting_points * inc
            vMax = valAbsMax - inc
        else:
            vMax = vMin * number_starting_points

        points = np.linspace(vMin, vMax, number_starting_points) + valMin

        return points

    def invalid_parameters(self, *params):
        """
        Identifies if the parameters passed are within the bounds provided

        If they are not returns ``inf``

        Parameters
        ----------
        params : list of floats
            Parameters to be passed to the sim

        Returns
        -------
        validity : Bool
            If the parameters are valid or not

        Notes
        -----
        No note

        Examples
        --------
        >>> a = FitAlg(bounds={1:(0,5), 2:(0,2), 3:(-1,1)})
        >>> a.set_bounds([3, 1])
        >>> a.invalid_parameters(0, 0)
        False
        >>> a.invalid_parameters(2, 0)
        True
        >>> a.invalid_parameters(0, -1)
        True
        >>> a.invalid_parameters(6, 6)
        True
        """

        for p, (mi, ma) in itertools.izip(params, self.boundary_values):

            if p < mi or p > ma:
                return True

        return False


def covariance(jac):
    """ Calculates the covariance based on the estimated jacobian

    Inspired by how this is calculated in scipy.optimise.curve_fit, as found at
    https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739

    """

    # Do Moore-Penrose inverse discarding zero singular values.
    U, s, VT = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    cov = np.dot(VT.T / s ** 2, VT)

    # Alternative method found, but assumes the residuals are small
    #cov = np.linalg.inv(np.dot(jac.T, jac))

    return cov