# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

from __future__ import division, print_function

from numpy import log2, sum
from collections import Callable

def qualFuncIdent(value):
    
    if isinstance(value, Callable):
        fitness = value
    elif value == "-2log":
        fitness = logprob
    elif value == "1-prob":
        fitness = maxprob
    else:
        fitness = simpleSum
        
    return fitness

                
def simpleSum(modVals):
    """
    Generates a fit quality value based on :math:`\sum {\\vec x}`
    
    Returns
    -------
    fit : float
        The sum of the model valaues returned
    """

    return sum(modVals)


def logprob(modVals):
    """
    Generates a fit quality value based on :math:`\sum -2\mathrm{log}_2(\\vec x)`
    
    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    logModCoiceprob = log2(modVals)

    probs = -2*logModCoiceprob
    
    fit = sum(probs)

    return fit


def selectLogprob(modVals):
    """
    Generates a fit quality value based on :math:`\sum -2\mathrm{log}_2(\\vec x)`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    if (modVals >= 0.5).all():
        logModCoiceprob = log2(modVals)
    else:
        modValsShaped = (modVals < 0.5) *modVals + (modVals >= 0.5) * 0.5
        logModCoiceprob = log2(modValsShaped)

    probs = -2 * logModCoiceprob

    fit = sum(probs)

    return fit


def maxprob(modVals):
    """
    Generates a fit quality value based on :math:`\sum 1-{\\vec x}`
    
    Returns
    -------
    fit : float
        The sum of the model valaues returned
    """
    
    fit = sum(1-modVals)
    
    return fit