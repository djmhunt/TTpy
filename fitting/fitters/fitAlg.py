# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from math import isinf
from numpy import log2, linspace
from itertools import izip
from utils import listMergeNP

class fitAlg(object):
    """
    The abstact class for fitting data

    Parameters
    ----------
    fitQualFunc : function, optional
        The function used to calculate the quality of the fit. The value it 
        returns proivides the fitter with its fitting guide. Default ``fitAlg.null``
    numStartPoints : int, optional
        The number of starting points generated for each parameter.
        Default 4
        
    Attributes
    ----------
    Name: string
        The name of the fitting method
        
    See Also
    --------
    fitting.fit.fit : The general fitting framework class

    """

    Name = 'none'


    def __init__(self,fitQualFunc = None, numStartPoints = 4):
        
        self.numStartPoints = numStartPoints

        self.fitness = self.null

        self.fitInfo = {'Name':self.Name}

    def null(self,*params):
        """
        Generates a fit quality value
        
        Returns
        -------
        fit : float
            The sum of the model valaues returned
        """

        modVals = self.sim(*params)

        return sum(modVals)

    def logprob(self,*params):
        """
        Generates a fit quality value based on :math:`\sum -2\mathrm{log}_2({\\vec x}^2)`
        
        Returns
        -------
        fit : float
            The sum of the model valaues returned
        """

        modVals = self.sim(*params)

        logModCoiceprob = log2(modVals)

        probs = -2*logModCoiceprob
        
        fit = sum(probs)

        return fit
        
    def maxprob(self,*params):
        """
        Generates a fit quality value based on :math:`\sum 1-{\\vec x}`
        
        Returns
        -------
        fit : float
            The sum of the model valaues returned
        """
        
        modVals = self.sim(*params)
        
        fit = sum(1-modVals)
        
        return fit

    def fit(self, sim, mInitialParams):
        """
        Runs the model through the fitting algorithms and starting parameters 
        and returns the best one. This is the abstract version that always 
        returns ``(0,0)``
        
        Parameters
        ----------
        sim : function
            The function used by a fitting algorithm to generate a fit for 
            given model parameters. One example is ``fit.fitness``
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

    def info(self):
        """
        The dictionary describing the fitting algorithm chosen
        
        Returns
        -------
        fitInfo : dict
            The dictionary of fitting class information
        """

        return self.fitInfo
        
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
        """

        if bounds == None:
            # We only have the values passed in as the starting parameters
            startLists = (self.startParamVals(i, numPoints = numPoints) for i in initialParams)

        else: 
            startLists = (self.startParamVals(i, bMax = bMax, numPoints = numPoints) for i, (bMin, bMax) in izip(initialParams,bounds))
            
        startSets = listMergeNP(*startLists)
            
        return startSets
        
    def startParamVals(self,initial, bMax = float('Inf'), numPoints = 3):
        """
        Assumes that intial parameters are positive and provides all starting 
        values above zero
        
        Parameters
        ----------
        initial : float
            The inital starting value proposed
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
        the starting point provided is less than one it will be assumed that the 
        values cannot exceed 1, otherwise, unless otherwise told, it will be 
        assumed that they can take any value and will be chosen to be eavenly 
        spaced around the provided value.
        
        """
        
        initialAbs = abs(initial)
    
         #The number of initial points per parameter
        divVal = (numPoints+1)/2
        
        # We can assume that any initial parameter proposed has the 
        #correct order of magnitude. 
        vMin = initialAbs / divVal
        
        
        if bMax == None or isinf(bMax):
            # We can also assume any number smaller than one should stay 
            #smaller than one.
            if initialAbs < 1:
                valAbsMax = 1
            else:
                valAbsMax = float('inf')
        else:
            valAbsMax = bMax
            
        if numPoints*vMin > valAbsMax:
            inc = (valAbsMax - initialAbs) / divVal
            vMin = valAbsMax - numPoints * inc 
            vMax = valAbsMax - inc
        else:
            vMax = vMin * numPoints
            
           
        points = linspace(vMin, vMax, numPoints)
        
        return points

