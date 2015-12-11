# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

import logging

from math import isinf
from numpy import linspace
from itertools import izip

from utils import listMergeNP, callableDetailsString
from qualityFunc import qualFuncIdent
from boundFunc import scalarBound

class fitAlg(object):
    """
    The abstact class for fitting data

    Parameters
    ----------
    fitQualFunc : string, optional
        The name of the function used to calculate the quality of the fit.
        The value it returns proivides the fitter with its fitting guide.
        Default ``fitAlg.null``
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

    Attributes
    ----------
    Name : string
        The name of the fitting method

    See Also
    --------
    fitting.fit.fit : The general fitting framework class

    """

    Name = 'none'


    def __init__(self,fitQualFunc = None, bounds = None, boundCostFunc = scalarBound(), numStartPoints = 4):

        self.numStartPoints = numStartPoints
        self.allBounds = bounds

        self.fitQualFunc = qualFuncIdent(fitQualFunc)
        self.boundCostFunc = boundCostFunc

        self.fitInfo = {'Name':self.Name,
                        'fitQualityFunction': fitQualFunc,
                        'boundaryCostFunction': callableDetailsString(boundCostFunc),
                        'bounds':self.allBounds,
                        'numStartPoints' : self.numStartPoints}

        self.boundVals = None
        self.boundNames = None

        self.logger = logging.getLogger('Fitting.fitters.fitAlg')

    def fit(self, sim, mParamNames, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters
        and returns the best one. This is the abstract version that always
        returns ``(0,0)``

        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for
            given model parameters. One example is ``fit.fitness``
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

        return 0, 0

    def fitness(self,*params):
        """
        Generates a fit quality value

        Returns
        -------
        fit : float
            The fit quality value calculated using the fitQualFunc function

        See Also
        --------
        fitting.fitters.qualityFunc : the module of fitQualFunc functions
        fitAlg.invalidParams : Checks if the parameters are valid and if not returns ``inf``
        """
        # This is because the fitting functions return an array and we want a list
        pms = list(*params)

        # Start by checking that the parameters are valid
        if self.allBounds and self.invalidParams(*pms):
            return self.boundCostFunc(pms,self.boundVals)

        modVals = self.sim(*pms)

        fit = self.fitQualFunc(modVals)

        return fit

    def info(self):
        """
        The dictionary describing the fitting algorithm chosen

        Returns
        -------
        fitInfo : dict
            The dictionary of fitting class information
        """

        return self.fitInfo

    def setBounds(self,mParamNames):
        """
        Checks if the bounds have changed

        Parameters
        ----------
        mParamNames : list of strings
            An ordered list of the names of the parameters to be fitted

        Examples
        --------
        >>> from fitting.fitters.fitAlg import fitAlg
        >>> a = fitAlg()
        >>> a.setBounds([])
        >>> a.setBounds(['string','two'])
        >>> a.allBounds
        {'string': (0, inf), 'two': (0, inf)}
        >>> a.boundNames
        ['string', 'two']
        >>> a.boundVals
        ￼[(0, inf), (0, inf)]

        >>> a = fitAlg(bounds = {1:(0,5),2:(0,2),3:(-1,1)})
        >>> a.allBounds
        {1:(0,5),2:(0,2),3:(-1,1)}
        >>> a.setBounds([])
        >>> a.allBounds
        {1:(0,5),2:(0,2),3:(-1,1)}
        >>> a.boundNames
        []
        >>> a.setBounds([3,1])
        >>> a.boundVals
        ￼[(-1, 1), (0, 5)]
        >>> a.setBounds([2,1])
        >>> a.boundVals
        ￼[(0, 2), (0, 5)]
        """

        bounds = self.allBounds
        boundNames = self.boundNames
        boundVals = self.boundVals

        # Check if the bounds have changed or should be added
        if boundNames:
            changed = False
            for m,b in izip(mParamNames,boundNames):
                if m != b :
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
            boundVals = [(0,float('Inf')) for i in mParamNames]
            self.allBounds = {k : v for k, v in izip(mParamNames, boundVals)}
            self.boundNames = mParamNames
            self.boundVals = boundVals
        else:
            self.boundVals = [ bounds[k] for k in mParamNames]
            self.boundNames = mParamNames

        return


    def startParams(self,initialParams, bounds = None, numPoints = 3):
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
        >>> from fitting.fitters.fitAlg import fitAlg
        >>> a = fitAlg()
        >>> self.startParams([0.5,0.5], numPoints=2)
        array([[ 0.33333333,  0.33333333],
               [ 0.66666667,  0.33333333],
               [ 0.33333333,  0.66666667],
               [ 0.66666667,  0.66666667]])
        """

        if bounds == None:
            # We only have the values passed in as the starting parameters
            startLists = (self.startParamVals(i, numPoints = numPoints) for i in initialParams)

        else:
            if len(bounds) != len(initialParams):
                raise ValueError('Bounds do not fit number of intial parameters', str(len(bounds)), str(len(initialParams)))

            startLists = (self.startParamVals(i, bMin = bMin, bMax = bMax, numPoints = numPoints) for i, (bMin, bMax) in izip(initialParams,bounds))

        startSets = listMergeNP(*startLists)

        return startSets

    def startParamVals(self,initial, bMin = float('-Inf'), bMax = float('Inf'), numPoints = 3):
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
        >>> from fitting.fitters.fitAlg import fitAlg
        >>> a = fitAlg()
        >>> a.startParamVals(0.5)
        array([ 0.25,  0.5 ,  0.75])

        >>> a.startParamVals(5)
        array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(-5)
        array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(5, bMin = 0, bMax = 7)
        array([ 4.,  5.,  6.])

        >>> a.startParamVals(5, bMin = -3, bMax = 30)
        array([ 1.,  5.,  9.])

        >>> a.startParamVals(5, bMin = 0, bMax = 30)
        array([ 2.5,  5. ,  7.5])

        >>> a.startParamVals(5, bMin = 3, bMax = 30, numPoints = 7)
        array([ 3.5,  4. ,  4.5,  5. ,  5.5,  6. ,  6.5])
        """

#        initialAbs = abs(initial)

        #The number of initial points per parameter
        divVal = (numPoints+1)/2

        if bMax == None or isinf(bMax):
            # We can also assume any number smaller than one should stay
            #smaller than one.
            if initial < 1 and initial > 0:
                valMax = 1
            else:
                valMax = float('inf')
        else:
            valMax = bMax

        if bMin == None or isinf(bMin):
            # We can also assume any number larger than one should stay
            #bigger than zero.
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


        points = linspace(vMin, vMax, numPoints) + valMin

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
        >>> from fitting.fitters.fitAlg import fitAlg
        >>> a = fitAlg(bounds = {1:(0,5),2:(0,2),3:(-1,1)})
        >>> a.setBounds([3,1])
        >>> a.invalidParams(0,0)
        False
        >>> a.invalidParams(2,0)
        True
        >>> a.invalidParams(0,-1)
        True
        >>> a.invalidParams(6,6)
        True
        """

        for p, (mi, ma) in izip(params,self.boundVals):

            if p < mi or p > ma:
                return True

        return False
