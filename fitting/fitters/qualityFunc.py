# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

from __future__ import division, print_function, unicode_literals, absolute_import

from numpy import log2, sum, ones, exp, shape, amax, array, linspace, concatenate
from numpy.random import choice
from collections import Callable

from utils import movingaverage


def qualFuncIdent(value, **kwargs):

    if isinstance(value, Callable):
        fitness = value
    elif value == "BIC2":
        fitness = BIC2(**kwargs)
    elif value == "BIC2Boot":
        fitness = BIC2Boot(**kwargs)
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


def BIC2(**kwargs):
    # type : (int, float) -> Callable[[Union[ndarray, list]], float]
    """

    Parameters
    ----------
    numParams : int, optional
        The number of parameters used by the model used for the fitting process. Default 2
    qualityThreshold : float, optional
        The BIC minimum fit quality criterion used for determining if a fit is valid. Default 20.0
    numActions: int or list of ints the length of the number of trials being fitted, optional
        The number of actions the participant can choose between for each timestep of the experiment. May need to be
        specified for each trial if the number of action choices varies between trials. Default 2
    randActProb: float or list of floats the length of the number of trials being fitted. Optional
        The prior probability of an action being randomly chosen. May need to be specified for each trial if the number
        of action choices varies between trials. Default ``1/numActions``

    Returns
    -------

    """

    # Set the values that will be fixed for the whole fitting process
    numParams = kwargs.pop("numParams", 2)
    qualityThreshold = kwargs.pop("qualityThreshold", 20)
    numActions = kwargs.pop("numActions", 2)
    randActProb = kwargs.pop("randActProb", 1/numActions)

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
        BICrandom = logprob(ones(numSamples) * randActProb)
        qualityConvertor = qualityThreshold**(2/BICrandom)

        fit = qualityConvertor * 2**(BICval/BICrandom - 1)

        return fit

    BICfunc.Name = "BIC2"
    BICfunc.Params = {"numParams": numParams,
                      "qualityThreshold": qualityThreshold,
                      "numActions": numActions,
                      "randActProb": randActProb}
    return BICfunc


def BIC2Boot(**kwargs):
    # type : (int, float) -> Callable[[Union[ndarray, list]], float]
    """

    Parameters
    ----------
    numParams : int, optional
        The number of parameters used by the model used for the fitting process. Default 2
    qualityThreshold : float, optional
        The BIC minimum fit quality criterion used for determining if a fit is valid. Default 20.0
    numActions: int or list of ints the length of the number of trials being fitted, optional
        The number of actions the participant can choose between for each timestep of the experiment. May need to be
        specified for each trial if the number of action choices varies between trials. Default 2
    randActProb: float or list of floats the length of the number of trials being fitted. Optional
        The prior probability of an action being randomly chosen. May need to be specified for each trial if the number
        of action choices varies between trials. Default ``1/numActions``
    numSamples: int, optional
        The number of samples that will be randomly resampled from ``modVals``. Default 100
    sampleLen: int, optional
        The length of the random sample. Default 1

    Returns
    -------

    """

    # Set the values that will be fixed for the whole fitting process
    numParams = kwargs.pop("numParams", 2)
    qualityThreshold = kwargs.pop("qualityThreshold", 20)
    numActions = kwargs.pop("numActions", 2)
    randActProb = kwargs.pop("randActProb", 1/numActions)
    numSamples = kwargs.pop("numSamples", 100)
    sampleLen = kwargs.pop("sampleLen", 1)

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

        sample = array(modVals).squeeze()

        # Calculate the resampled modVals
        numVals = sample.shape
        T = max(numVals)
        probdist = linspace(2 / (T * (T + 1)), 2 / (T + 1), T)
        choices = choice(range(T), size=numSamples, p=probdist)
        modValsExtra = array([sample[i: i + sampleLen] for i in choices]).squeeze()

        extendedSample = concatenate((sample, modValsExtra))

        numTrials = shape(extendedSample)

        # We define the Bayesian Information Criteria for the probability of the model given the data, relative a
        # guessing model

        BICval = numParams * log2(amax(numTrials)) + logprob(extendedSample)
        BICrandom = logprob(ones(numTrials) * randActProb)
        qualityConvertor = qualityThreshold**(2/BICrandom)

        fit = qualityConvertor * 2**(BICval/BICrandom - 1)

        return fit

    BICfunc.Name = "BIC2Boot"
    BICfunc.Params = {"numParams": numParams,
                      "qualityThreshold": qualityThreshold,
                      "numActions": numActions,
                      "randActProb": randActProb,
                      "numSamples": numSamples,
                      "sampleLen": sampleLen}
    return BICfunc