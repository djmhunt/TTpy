# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

from __future__ import division, print_function, unicode_literals, absolute_import

from numpy import log2, sum, ones, exp, shape, amax
from collections import Callable

from utils import movingaverage


def qualFuncIdent(value, **kwargs):

    if isinstance(value, Callable):
        fitness = value
    elif value == "BIC":
        fitness = BIC(**kwargs)
    elif value == "-2log":
        fitness = logprob
    elif value == "-2AvLog":
        fitness = logAverageProb
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
        The sum of the model values returned
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


def logAverageProb(modVals):
    """
    Generates a fit quality value based on :math:`\sum -2\mathrm{log}_2(\\vec x)`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    correctedVals = movingaverage(modVals, 3, edgeCorrection=True)

    logModCoiceprob = log2(correctedVals)

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


def BIC(**kwargs):
    # type : (int, float) -> Callable[[Union[ndarray, list]], float]

    numParams = kwargs.pop("numParams", 2)
    qualityThreshold = kwargs.pop("qualityThreshold", 20)

    def BICfunc(modVals):
        # type: (Union[ndarray, list]) -> float
        """
        Generates a fit quality value based on :math:`\mathrm{exp}^{\frac{\mathrm{numParams}\mathrm{log2}\left(\mathrm{numSamples}\right) + \mathrm{BICval}}{\mathrm{BICrandom}} - 1}`
        The function is a modified version of the Bayesian Informaiton Criterion

        It provides a fit such that when a value is less than one it is a valid fit

        Returns
        -------
        fit : float
            The sum of the model valaues returned
        """

        numSamples = shape(modVals)

        # We define the Bayesian Information Criteria for the probability of the model given the data, relative a
        # guessing model

        BICval = numParams * log2(amax(numSamples)) + logprob(modVals)
        BICrandom = logprob(ones(numSamples) * 0.5)
        qualityConvertor = qualityThreshold ** (2/BICrandom)

        fit = qualityConvertor * exp(BICval/BICrandom - 1)

        return fit

    BICfunc.Name = "BIC"
    BICfunc.Params = {"numParams": numParams}
    return BICfunc