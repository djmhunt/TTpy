# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

import collections

import numpy as np

import utils


def qualFuncIdent(value, **kwargs):

    if isinstance(value, collections.Callable):
        fitness = value
    elif value == "BIC":
        fitness = BIC2(**kwargs)
    elif value == "r2":
        fitness = r2(**kwargs)
    elif value == "bayesFactor":
        fitness = bayesFactor(**kwargs)
    elif value == "BIC2norm":
        fitness = BIC2norm(**kwargs)
    elif value == "BIC2normBoot":
        fitness = BIC2normBoot(**kwargs)
    elif value == "WBIC2":
        fitness = WBIC2(**kwargs)
    elif value == "-2log":
        fitness = logprob
    elif value == "-loge":
        fitness = logeprob
    elif value == "-2AvLog":
        fitness = logAverageProb
    elif value == "1-prob":
        fitness = maxprob
    else:
        fitness = simpleSum

    return fitness


def simpleSum(modVals):
    r"""
    Generates a fit quality value based on :math:`\sum {\\vec x}`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    return np.sum(modVals)


def logprob(modVals):
    # type: (Union[ndarray, list]) -> float
    r"""
    Generates a fit quality value based on :math:`f_{\mathrm{mod}}\left(\\vec x\right) = \sum -2\mathrm{log}_2(\\vec x)`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    logModCoiceprob = np.log2(modVals)

    probs = -2*logModCoiceprob

    fit = np.sum(probs)

    return fit


def logeprob(modVals):
    # type: (Union[ndarray, list]) -> float
    r"""
    Generates a fit quality value based on :math:`f_{\mathrm{mod}}\left(\\vec x\right) = \sum -\mathrm{log}_e(\\vec x)`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    logModCoiceprob = np.log(modVals)

    probs = -logModCoiceprob

    fit = np.sum(probs)

    return fit


def logAverageProb(modVals):
    # type: (Union[ndarray, list]) -> float
    r"""
    Generates a fit quality value based on :math:`\sum -2\mathrm{log}_2(\\vec x)`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    correctedVals = utils.movingaverage(modVals, 3, edgeCorrection=True)

    logModCoiceprob = np.log2(correctedVals)

    probs = -2 * logModCoiceprob

    fit = np.sum(probs)

    return fit


def maxprob(modVals):
    # type: (Union[ndarray, list]) -> float
    r"""
    Generates a fit quality value based on :math:`\sum 1-{\\vec x}`

    Returns
    -------
    fit : float
        The sum of the model values returned
    """

    fit = np.sum(1-modVals)

    return fit


def BIC2(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    r"""
    Generates a function that calculates the Bayesian Information Criterion (BIC)

    :math:`\lambda \mathrm{log}_2(T)+ f_{\mathrm{mod}}\left(\\vec x\right)`

    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    numParams = kwargs.get("numParams", 2)

    def BICfunc(modVals, **kwargs):
        # type: (Union[ndarray, list]) -> float
        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))
        BICmod = numParams * np.log2(numSamples) + logprob(modVals)
        return BICmod

    BICfunc.Name = "BIC2"
    BICfunc.Params = {"numParams": numParams}
    return BICfunc


def bayesRand(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    randActProb = kwargs.get("randActProb", 1 / 2)

    def BICfunc(modVals, **kwargs):

        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))
        BICrand = logprob(np.ones(numSamples) * randActProb)
        return BICrand

    BICfunc.Name = "bayesRand"
    BICfunc.Params = {"randActProb": randActProb}
    return BICfunc


def bayesFactor(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    r"""

    :math:`2^{\frac{x}{2}}`

    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    numParams = kwargs.get("numParams", 2)
    randActProb = kwargs.get("randActProb", 1 / 2)
    BICmodfunc = BIC2(numParams=numParams)
    BICrandfunc = bayesRand(randActProb=randActProb)

    def bayesFunc(modVals, **kwargs):
        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))
        BICmod = kwargs.get('BICmod', BICmodfunc(modVals, numSamples=numSamples))
        BICrandom = kwargs.get('BICrand', BICrandfunc(modVals, numSamples=numSamples))

        bayesF = 2 ** ((BICrandom - BICmod) / 2)

        return bayesF

    bayesFunc.Name = "bayesFactor"
    bayesFunc.Params = {"randActProb": randActProb,
                        "numParams": numParams}
    return bayesFunc


def bayesInv(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    r"""

    Parameters
    ----------
    numParams : int, optional
        The number of parameters used by the model used for the fitters process. Default 2
    qualityThreshold : float, optional
        The BIC minimum fit quality criterion used for determining if a fit is valid. Default 20.0
    number_actions: int or list of ints the length of the number of trials being fitted, optional
        The number of actions the participant can choose between for each trialstep of the task. May need to be
        specified for each trial if the number of action choices varies between trials. Default 2
    randActProb: float or list of floats the length of the number of trials being fitted. Optional
        The prior probability of an action being randomly chosen. May need to be specified for each trial if the number
        of action choices varies between trials. Default ``1/number_actions``

    Returns
    -------

    """

    # Set the values that will be fixed for the whole fitters process
    numParams = kwargs.get("numParams", 2)
    qualityThreshold = kwargs.get("qualityThreshold", 20)
    number_actions = kwargs.get("number_actions", 2)
    randActProb = kwargs.get("randActProb", 1/number_actions)

    BICmodfunc = BIC2(numParams=numParams)
    BICrandfunc = bayesRand(randActProb=randActProb)

    def BICfunc(modVals, **kwargs):
        # type: (Union[ndarray, list]) -> float
        r"""
        Generates a fit quality value based on :math:`\mathrm{exp}^{\frac{\mathrm{numParams}\mathrm{log2}\left(\mathrm{numSamples}\right) + \mathrm{BICval}}{\mathrm{BICrandom}} - 1}`
        The function is a modified version of the Bayesian Information Criterion

        It provides a fit such that when a value is less than one it is a valid fit

        Returns
        -------
        fit : float
            The sum of the model values returned
        """

        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))

        # We define the Bayesian Information Criteria for the probability of the model given the data, relative a
        # guessing model

        BICmod = kwargs.get('BICmod', BICmodfunc(modVals, numSamples=numSamples))
        BICrandom = kwargs.get('BICrand', BICrandfunc(modVals, numSamples=numSamples))

        fit = qualityThreshold * 2**((BICmod-BICrandom) / 2)

        return fit

    BICfunc.Name = "bayesInv"
    BICfunc.Params = {"numParams": numParams,
                      "qualityThreshold": qualityThreshold,
                      "number_actions": number_actions,
                      "randActProb": randActProb}
    return BICfunc


def r2(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    numParams = kwargs.get("numParams", 2)
    randActProb = kwargs.get("randActProb", 1 / 2)
    BICmodfunc = BIC2(numParams=numParams)
    BICrandfunc = bayesRand(randActProb=randActProb)

    def r2func(modVals, **kwargs):
        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))
        BICmod = kwargs.get('BICmod', BICmodfunc(modVals, numSamples=numSamples))
        BICrandom = kwargs.get('BICrand', BICrandfunc(modVals, numSamples=numSamples))

        r = BICmod/BICrandom - 1
        return r

    r2func.Name = "r2"
    r2func.Params = {"randActProb": randActProb,
                     "numParams": numParams}
    return r2func


def BIC2norm(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    """

    Parameters
    ----------
    numParams : int, optional
        The number of parameters used by the model used for the fits process. Default 2
    qualityThreshold : float, optional
        The BIC minimum fit quality criterion used for determining if a fit is valid. Default 20.0
    number_actions: int or list of ints the length of the number of trials being fitted, optional
        The number of actions the participant can choose between for each trialstep of the task. May need to be
        specified for each trial if the number of action choices varies between trials. Default 2
    randActProb: float or list of floats the length of the number of trials being fitted. Optional
        The prior probability of an action being randomly chosen. May need to be specified for each trial if the number
        of action choices varies between trials. Default ``1/number_actions``

    Returns
    -------

    """

    # Set the values that will be fixed for the whole fits process
    numParams = kwargs.get("numParams", 2)
    qualityThreshold = kwargs.get("qualityThreshold", 20)
    number_actions = kwargs.get("number_actions", 2)
    randActProb = kwargs.get("randActProb", 1/number_actions)

    BICmodfunc = BIC2(numParams=numParams)
    BICrandfunc = bayesRand(randActProb=randActProb)

    def BICfunc(modVals, **kwargs):
        # type: (Union[ndarray, list]) -> float
        r"""
        Generates a fit quality value based on :math:`\mathrm{exp}^{\frac{\mathrm{numParams}\mathrm{log2}\left(\mathrm{numSamples}\right) + \mathrm{BICval}}{\mathrm{BICrandom}} - 1}`
        The function is a modified version of the Bayesian Information Criterion

        It provides a fit such that when a value is less than one it is a valid fit

        Returns
        -------
        fit : float
            The sum of the model values returned
        """

        numSamples = kwargs.get('numSamples', np.amax(np.shape(modVals)))

        # We define the Bayesian Information Criteria for the probability of the model given the data, relative a
        # guessing model

        BICmod = kwargs.get('BICmod', BICmodfunc(modVals, numSamples=numSamples))
        BICrandom = kwargs.get('BICrand', BICrandfunc(modVals, numSamples=numSamples))
        qualityConvertor = qualityThreshold**(2/BICrandom)

        fit = qualityConvertor * 2**(BICmod/BICrandom - 1)

        return fit

    BICfunc.Name = "BIC2norm"
    BICfunc.Params = {"numParams": numParams,
                      "qualityThreshold": qualityThreshold,
                      "number_actions": number_actions,
                      "randActProb": randActProb}
    return BICfunc


def BIC2normBoot(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    """
    An attempt at looking what would happen if the samples were resampled. It was hoped that by doing this, the
    difference between different sample distributions would become more pronounced. This was not found to be true.

    Parameters
    ----------
    numParams : int, optional
        The number of parameters used by the model used for the fits process. Default 2
    qualityThreshold : float, optional
        The BIC minimum fit quality criterion used for determining if a fit is valid. Default 20.0
    number_actions: int or list of ints the length of the number of trials being fitted, optional
        The number of actions the participant can choose between for each trialstep of the task. May need to be
        specified for each trial if the number of action choices varies between trials. Default 2
    randActProb: float or list of floats the length of the number of trials being fitted. Optional
        The prior probability of an action being randomly chosen. May need to be specified for each trial if the number
        of action choices varies between trials. Default ``1/number_actions``
    numSamples: int, optional
        The number of samples that will be randomly resampled from ``modVals``. Default 100
    sampleLen: int, optional
        The length of the random sample. Default 1

    Returns
    -------

    """

    # Set the values that will be fixed for the whole fits process
    numParams = kwargs.pop("numParams", 2)
    qualityThreshold = kwargs.pop("qualityThreshold", 20)
    number_actions = kwargs.pop("number_actions", 2)
    randActProb = kwargs.pop("randActProb", 1/number_actions)
    numSamples = kwargs.pop("numSamples", 100)
    sampleLen = kwargs.pop("sampleLen", 1)

    def BICfunc(modVals):
        # type: (Union[ndarray, list]) -> float
        r"""
        Generates a fit quality value based on :math:`\mathrm{exp}^{\frac{\mathrm{numParams}\mathrm{log2}\left(\mathrm{numSamples}\right) + \mathrm{BICval}}{\mathrm{BICrandom}} - 1}`
        The function is a modified version of the Bayesian Informaiton Criterion

        It provides a fit such that when a value is less than one it is a valid fit

        Returns
        -------
        fit : float
            The sum of the model valaues returned
        """

        sample = np.array(modVals).squeeze()

        # Calculate the resampled modVals
        numVals = sample.shape
        T = max(numVals)
        probdist = np.linspace(2 / (T * (T + 1)), 2 / (T + 1), T)
        choices = np.random.choice(list(range(T)), size=numSamples, p=probdist)
        modValsExtra = np.array([sample[i: i + sampleLen] for i in choices]).squeeze()

        extendedSample = np.concatenate((sample, modValsExtra))

        numTrials = np.shape(extendedSample)

        # We define the Bayesian Information Criteria for the probability of the model given the data, relative a
        # guessing model

        BICval = numParams * np.log2(np.amax(numTrials)) + logprob(extendedSample)
        BICrandom = logprob(np.ones(numTrials) * randActProb)
        qualityConvertor = qualityThreshold**(2/BICrandom)

        fit = qualityConvertor * 2**(BICval/BICrandom - 1)

        return fit

    BICfunc.Name = "BIC2normBoot"
    BICfunc.Params = {"numParams": numParams,
                      "qualityThreshold": qualityThreshold,
                      "number_actions": number_actions,
                      "randActProb": randActProb,
                      "numSamples": numSamples,
                      "sampleLen": sampleLen}
    return BICfunc


def WBIC2(**kwargs):
    # type : (**Union[int, float]) -> Callable[[Union[ndarray, list]], float]
    """
    Unfinished WBIC implementation

    Parameters
    ----------


    Returns
    -------

    """

    def WBICfunc(modVals):
        # type: (Union[ndarray, list]) -> float
        r"""
        Generates a fit quality value based on :math:`\mathrm{exp}^{\frac{\mathrm{numParams}\mathrm{log2}\left(\mathrm{numSamples}\right) + \mathrm{BICval}}{\mathrm{BICrandom}} - 1}`
        The function is a modified version of the Bayesian Information Criterion

        It provides a fit such that when a value is less than one it is a valid fit

        Returns
        -------
        fit : float
            The sum of the model values returned
        """

        numSamples = np.shape(modVals)

        temp = 1 / (np.log2(np.amax(numSamples)))
        fit = 1 + logprob(modVals) / np.prod(modVals**temp)

        return fit

    WBICfunc.Name = "WBIC2"
    WBICfunc.Params = {}
    return WBICfunc