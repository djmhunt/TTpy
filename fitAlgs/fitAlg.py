# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

import numpy as np
import scipy as sp

import math
import collections
import itertools

#import numdifftools as nd

from fitAlgs import qualityFunc
from fitAlgs import boundFunc

import utils


class FitAlg(object):
    """
    The abstract class for fitting data

    Parameters
    ----------
    fitSim : fitAlgs.fitSims.fitSim instance
        An instance of one of the fitting algorithms
    fitQualityFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns provides the fitter with its fitting guide.
        Default ``fitAlg.null``
    qualityFuncArgs : dict, optional
        The parameters used to initialise fitQualFunc. Default ``{}``
    bounds : dictionary of tuples of length two with floats, optional
        The boundaries for methods that use bounds. If unbounded methods are
        specified then the bounds will be ignored. Default is ``None``, which
        translates to boundaries of (0,float('Inf')) for each parameter.
    boundCostFunc : function, optional
        A function used to calculate the penalty for exceeding the boundaries.
        Default is ``boundFunc.scalarBound``
    calcCov : bool, optional
        Is the covariance calculated. Default ``False``
    extraFitMeasures : dict of dict, optional
        Dictionary of fit measures not used to fit the model, but to provide more information. The keys are the
        fitQUalFunc used names and the values are the qualFuncArgs. Default ``{}``

    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitAlgs.fitSims.fitSim : The general fitting class

    """

    def __init__(self, fitSim=None, fitQualityFunc=None, qualityFuncArgs={}, bounds=None, boundCostFunc=boundFunc.scalarBound(), extraFitMeasures={}, calcCov=False, boundRatio=0.000001, **kwargs):

        if fitSim is None:
            self.fitSim = FitSim()
        elif isinstance(fitSim, FitSim):
            self.fitSim = fitSim
        else:
            raise NameError("fitSim type is incorrect: {}".format(type(fitSim)))

        if bounds is None:
            raise NameError("Please specify bounds for your parameters")
        else:
            self.allBounds = bounds

        self.Name = self.findName()

        self.boundCostFunc = boundCostFunc

        self.fitQualityFunc = qualityFunc.qualFuncIdent(fitQualityFunc, **qualityFuncArgs)
        self.calcCovariance = calcCov
        if self.calcCovariance:
            self.hessInc = {k: boundRatio * (u - l) for k, (l, u) in self.allBounds.iteritems()}

        self.measures = {k: qualityFunc.qualFuncIdent(k, **v) for k, v in extraFitMeasures.iteritems()}

        self.fitInfo = {'Name': self.Name,
                        'fitQualityFunction': fitQualityFunc,
                        'fitQualityArguments': qualityFuncArgs,
                        'boundaryCostFunction': utils.callableDetailsString(boundCostFunc),
                        'bounds': self.allBounds,
                        'extraFitMeasures': extraFitMeasures,
                        'calculateCovariance': calcCov,
                        'boundRatio': boundRatio}

        self.boundVals = None
        self.boundNames = None

        self.testedParams = []
        self.testedParamQualities = []

        self.logger = logging.getLogger(self.Name)

    def __repr__(self):

        return repr(self.info())

    def findName(self):
        """
        Returns the name of the class
        """

        return self.__class__.__name__

    def participant(self, model, modelSetup, partData):
        """
        Fit participant data to a model for a given task

        Parameters
        ----------
        model : model.model.model inherited class
            The model you wish to try and fit values to
        modelSetup : (dict,dict)
            The first dictionary is the model initial parameters. The second
            are the other model parameters
        partData : dict
            The participant data

        Returns
        -------
        model : model.model.model inherited class instance
            The model with the best fit parameters
        fitQuality : float
            Specifies the fit quality for this participant to the model
        testedParams : tuple of OrderedDict and list
            They are an ordered dictionary containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.
        """

        sim = self.fitSim.getSim(model, modelSetup, partData)

        mInitialParams = modelSetup[0].values() # These are passed separately to define at this point the order of the parameters
        mParamNames = modelSetup[0].keys()

        fitVals, fitQuality, fitInfo = self.fit(sim, mParamNames, mInitialParams[:])

        modelRun = self.fitSim.fittedModel(*fitVals)

        fitMeasures = self.extraMeasures(*fitVals)

        testedParamDict = collections.OrderedDict([(key, val[0]) for key, val in itertools.izip(mParamNames, np.array(fitInfo[0]).T)])

        fittingData = {"testedParameters": testedParamDict,
                       "fitQualities": fitInfo[1],
                       "fitQuality": fitQuality,
                       "finalParameters": collections.OrderedDict([(key, val) for key, val in itertools.izip(mParamNames, fitVals)])}

        fittingData.update({"fitQuality_" + k: v for k, v in fitMeasures.iteritems()})

        if self.calcCovariance:
            covariance = self.covariance(mParamNames, fitVals, fitInfo[2])
            covdict = ({"fitQuality_cov_{}_{}".format(p1, p2): c for p1, cr in itertools.izip(mParamNames, covariance)
                                                                 for p2, c in itertools.izip(mParamNames, cr)})
            fittingData.update(covdict)

        try:
            fittingData.update(fitInfo[2])
        finally:
            return modelRun, fitQuality, fittingData

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters
        and returns the best one. This is the abstract version that always
        returns ``(0,0)``

        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for
            given model parameters. One example is ``fitAlgs.fitAlg.fitness``
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

        return 0, 0, (self.testedParams, self.testedParamQualities)

    def fitness(self, *params):
        """
        Generates a fit quality value used by the fitting function. This is the function passed to the fitting function.

        Parameters
        ----------
        *params : array of floats
            The parameters proposed by the fitting algorithm

        Returns
        -------
        fitQuality : float
            The fit quality value calculated using the fitQualFunc function

        See Also
        --------
        fitAlgs.qualityFunc : the module of fitQualFunc functions
        fitAlg.invalidParams : Checks if the parameters are valid and if not returns ``inf``
        fitAlgs.fitSims.fitSim.fitness : Runs the model simulation and returns the values used to calculate the fitQuality

        """
        # This is because the fitting functions return an array and we want a list
        pms = list(*params)

        # Start by checking that the parameters are valid
        if self.invalidParams(*pms):
            pseudofitQuality = self.boundCostFunc(pms, self.boundVals, self.fitQualityFunc)
            return pseudofitQuality

        # Run the simulation with these parameters
        modVals = self.sim(*pms)

        fitQuality = self.fitQualityFunc(modVals)

        self.testedParams.append(params)
        self.testedParamQualities.append(fitQuality)

        return fitQuality

    def extraMeasures(self, *params):
        """

        Parameters
        ----------
        *params : array of floats
            The parameters proposed by the fitting algorithm

        Returns
        -------
        fitQuality : dict of float
            The fit quality value calculated using the fit quality functions described in extraMeasures

        """

        modVals = self.sim(*params)

        measureVals = {}
        for m, f in self.measures.iteritems():

            fitQuality = f(modVals)
            measureVals[m] = fitQuality

        return measureVals

    def covariance(self, mParamNames, paramvals, fitinfo):
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
            inc = [self.hessInc[p] for p in mParamNames]
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

        fitSimInfo = self.fitSim.info()

        return self.fitInfo, fitSimInfo

    def setBounds(self, mParamNames):
        """
        Checks if the bounds have changed

        Parameters
        ----------
        mParamNames : list of strings
            An ordered list of the names of the parameters to be fitted

        Examples
        --------
        >>> a = FitAlg()
        >>> a.setBounds([])
        >>> a.setBounds(['string','two'])
        >>> a.allBounds
        {'string': (0, inf), 'two': (0, inf)}
        >>> a.boundNames
        ['string', 'two']
        >>> a.boundVals
        [(0, inf), (0, inf)]

        >>> a = FitAlg(bounds = {1:(0,5),2:(0,2),3:(-1,1)})
        >>> a.allBounds
        {1:(0,5),2:(0,2),3:(-1,1)}
        >>> a.setBounds([])
        >>> a.allBounds
        {1:(0,5),2:(0,2),3:(-1,1)}
        >>> a.boundNames
        []
        >>> a.setBounds([3,1])
        >>> a.boundVals
        [(-1, 1), (0, 5)]
        >>> a.setBounds([2,1])
        >>> a.boundVals
        [(0, 2), (0, 5)]
        """

        bounds = self.allBounds
        boundNames = self.boundNames
        boundVals = self.boundVals

        # Check if the bounds have changed or should be added
        if boundNames:
            changed = False
            for m, b in itertools.izip(mParamNames, boundNames):
                if m != b:
                    changed = True
                    break
        else:
            changed = True

        # If they have not, then we can leave
        if not changed:
            if len(mParamNames) == len(boundNames):
                return

        # If no bounds were defined
        if not bounds:
            boundVals = [(0, float('Inf')) for i in mParamNames]
            self.allBounds = {k: v for k, v in itertools.izip(mParamNames, boundVals)}
            self.boundNames = mParamNames
            self.boundVals = boundVals
        else:
            self.boundVals = [bounds[k] for k in mParamNames]
            self.boundNames = mParamNames

        return

    def startParams(self, initialParams, bounds=None, numPoints=3):
        """
        Defines a list of different starting parameters to run the minimization
        over

        Parameters
        ----------
        initialParams : list of floats
            The inital starting values proposed
        bounds : list of tuples of length two with floats, optional
            The boundaries for methods that use bounds. If unbounded methods are
            specified then the bounds will be ignored. Default is ``None``, which
            translates to boundaries of (0,float('Inf')) for each parameter.
        numPoints : int
            The number of starting parameter values to be calculated around
            each inital point

        Returns
        -------
        startParamSet : list of list of floats
            The generated starting parameter combinations

        See Also
        --------
        fitAlg.startParamVals : Used in this function

        Examples
        --------
        >>> a = FitAlg()
        >>> self.startParams([0.5,0.5], numPoints=2)
        np.array([[ 0.33333333,  0.33333333],
               [ 0.66666667,  0.33333333],
               [ 0.33333333,  0.66666667],
               [ 0.66666667,  0.66666667]])
        """

        if bounds is None:
            # We only have the values passed in as the starting parameters
            startLists = (self.startParamVals(i, numPoints = numPoints) for i in initialParams)

        else:
            if len(bounds) != len(initialParams):
                raise ValueError('Bounds do not fit the number of initial parameters', str(len(bounds)), str(len(initialParams)))

            startLists = (self.startParamVals(i, bMin=bMin, bMax=bMax, numPoints=numPoints) for i, (bMin, bMax) in itertools.izip(initialParams, bounds))

        startSets = utils.listMergeNP(*startLists)

        return startSets

    def startParamVals(self, initial, bMin=float('-Inf'), bMax=float('Inf'), numPoints=3):
        """
        Provides a set of starting points

        Parameters
        ----------
        initial : float
            The inital starting value proposed
        bMin : float, optional
            The minimum value of the parameter. Default is ``float('-Inf')``
        bMax : float, optional
            The maximum value of the parameter. Default is ``float('Inf')``
        numPoints : int
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
        >>> a = FitAlg()
        >>> a.startParamVals(0.5)
        np.array([ 0.25,  0.5 ,  0.75])

        >>> a.startParamVals(5)
        np.array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(-5)
        np.array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(5, bMin = 0, bMax = 7)
        np.array([ 4.,  5.,  6.])

        >>> a.startParamVals(5, bMin = -3, bMax = 30)
        np.array([ 1.,  5.,  9.])

        >>> a.startParamVals(5, bMin = 0, bMax = 30)
        np.array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(5, bMin = 3, bMax = 30, numPoints = 7)
        np.array([ 3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5])
        """

#        initialAbs = abs(initial)

        # The number of initial points per parameter
        divVal = (numPoints+1)/2

        if bMax is None or math.isinf(bMax):
            # We can also assume any number smaller than one should stay
            # smaller than one.
            if initial < 1 and initial > 0:
                valMax = 1
            else:
                valMax = float('inf')
        else:
            valMax = bMax

        if bMin is None or math.isinf(bMin):
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
            valMin = bMin

            initialAbs = initial - valMin
            valAbsMax = valMax - valMin

        # Now that the bounds have been set we have shifted the space to
        # calculate the points in the space and then shift them back.

        # We can assume that any initial parameter proposed has the
        # correct order of magnitude.
        vMin = initialAbs / divVal

        if numPoints*vMin > valAbsMax:
            inc = (valAbsMax - initialAbs) / divVal
            vMin = valAbsMax - numPoints * inc
            vMax = valAbsMax - inc
        else:
            vMax = vMin * numPoints

        points = np.linspace(vMin, vMax, numPoints) + valMin

        return points

    def invalidParams(self, *params):
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
        >>> a = FitAlg(bounds = {1:(0,5), 2:(0,2), 3:(-1,1)})
        >>> a.setBounds([3, 1])
        >>> a.invalidParams(0, 0)
        False
        >>> a.invalidParams(2, 0)
        True
        >>> a.invalidParams(0, -1)
        True
        >>> a.invalidParams(6, 6)
        True
        """

        for p, (mi, ma) in itertools.izip(params, self.boundVals):

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