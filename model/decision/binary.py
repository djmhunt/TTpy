# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import warnings
import random
import itertools
import collections

import numpy as np

from types import NoneType


def decSingle(expResponses=(0, 1)):
    """Decisions using a switching probability

    Parameters
    ----------
    expResponses : tuple of length two, optional
        Provides the two action responses expected by the experiment

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or NoneType
        The action to be taken by the model
    probabilities : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    See Also
    --------
    models.BPMS, model.BHMM

    Examples
    --------
    >>> from model.decision.binary import decSingle
    >>> lastAct = 0
    >>> dec = decSingle()
    >>> dec(0.23, lastAct)
    (0, {0,:0.77, 1:0.23})
    """

    expResponseSet = set(expResponses)

    def decisionFunc(prob, lastAction, stimulus=None, validResponses=None):

        if type(validResponses) is not NoneType:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif len(validResponses) == 0:
                return None, prob
            elif set(validResponses) != expResponseSet:
                warnings.warn("Bad validResponses: " + str(validResponses))
            else:
                warnings.warn("Bad number of validResponses: " + str(validResponses))

        randNum = random.random()

        lastNotAction = list(expResponseSet.difference([lastAction]))[0]

        if prob >= randNum:
            # The decision is to switch
            decision = lastNotAction
        else:
            # Keep the same decision
            decision = lastAction

        pSet = {lastNotAction: prob,
                lastAction: 1-prob}

        probDict = collections.OrderedDict({k: pSet[k] for k in expResponses})

        return decision, probDict

    decisionFunc.Name = "binary.decSingle"
    decisionFunc.Params = {}

    return decisionFunc


def decEta(expResponses=(0, 1), eta=0):
    """Decisions using a probability difference threshold

    Parameters
    ----------
    expResponses : tuple of length two containing non-negative ints, optional
        Provides the two action responses expected by the experiment
    eta : float, optional :math:`\\eta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \\eta`
        If true a decision is taken. If false the function responds ``None``

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on their probabilities and returns the
        decision and the probabilities for that decision.
    decision : int or NoneType
        The action to be taken by the model
    probabilities : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    See Also
    --------
    models.BP, models.MS, models.EP, models.MSRev, models.QLearn, models.QLearn2, models.OpAL

    """

    expResponseSet = set(expResponses)

    def decisionFunc(prob, lastAction, stimulus=None, validResponses=None):

        probabilities = np.array(prob).flatten()

        probDict = collections.OrderedDict({k: v for k, v in itertools.izip(expResponses, probabilities)})

        if type(validResponses) is not NoneType:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif len(validResponses) == 0:
                return None, probDict
            elif set(validResponses) != expResponseSet:
                warnings.warn("Bad validResponses: " + str(validResponses))
            elif len(validResponses) > 2:
                warnings.warn("Bad number of validResponses: " + str(validResponses))

        prob = probabilities[0]

        if abs(prob-0.5) >= eta:
            if prob > 0.5:
                decision = expResponses[0]
            elif prob == 0.5:
                decision = random.choice(expResponses)
            else:
                decision = expResponses[1]
        else:
            decision = None

        return decision, probDict

    decisionFunc.Name = "binary.decEta"
    decisionFunc.Params = {"expResponses": expResponses,
                           "eta": eta}

    return decisionFunc


def decRandom(expResponses=(0, 1)):
    """Decisions using a comparison with a uniform random number

    Parameters
    ----------
    expResponses : tuple of length two containing non-negative ints, optional
        Provides the two action responses expected by the experiment

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on their probabilities and returns the
        decision and the probabilities for that decision.
    decision : int or NoneType
        The action to be taken by the model
    probabilities : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    See Also
    --------
    models.BP, models.MS, models.EP, models.MSRev, models.QLearn, models.QLearn2, models.OpAL

    """

    expResponseSet = set(expResponses)

    def decisionFunc(prob, lastAction, stimulus=None, validResponses=None):

        probabilities = np.array(prob).flatten()

        probDict = collections.OrderedDict({k: v for k, v in itertools.izip(expResponses, probabilities)})

        if type(validResponses) is not NoneType:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif len(validResponses) == 0:
                return None, probDict
            elif set(validResponses) != expResponseSet:
                warnings.warn("Bad validResponses: " + str(validResponses))
            elif len(validResponses) > 2:
                warnings.warn("Bad number of validResponses: " + str(validResponses))

        prob = probabilities[0]
        decVal = random.random()

        if prob > decVal:
            decision = expResponses[0]
        elif prob == decVal:
            decision = random.choice(expResponses)
        else:
            decision = expResponses[1]

        return decision, probDict

    decisionFunc.Name = "binary.decRandom"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc


def decEtaSets(expResponses=(0, 1), eta=0):
    """
    Decisions using a probability difference threshold for the expectation
    from two sets of action-response value probabilities.

    It is assumed that the response values are increasing and evenly spaced.

    The two sets of probabilities are provided as a one dimensional array,
    with one set after the other.

    Parameters
    ----------
    expResponses : tuple of length two, optional
        Provides the two action responses expected by the experiment
    eta : float, optional :math:`\\eta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \\eta`
        If true a decision is taken. If false the function responds ``None``
        Default ``0``

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or NoneType
        The action to be taken by the model
    probDict : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    See Also
    --------
    models.BP, models.MSRev, experiment.decks.deckStimAllInfo

    Examples
    --------
    >>> import numpy as np
    >>> from model.decision.binary import decEtaSets
    >>> lastAct = 0
    >>> dec = decEtaSets()
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct)
    (0, {0:0.8, 1:0.2})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, validResponses=[0])
    (0, {0: 1})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, validResponses=[0,3])
    model\decision\binary.py:223: UserWarning: Bad number of validResponses: [0, 3]
    warn("Bad number of validResponses: " + str(validResponses))
    (0, {0:0.8, 1:0.2})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, validResponses=[])
    (None, {0:0.8, 1:0.2})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, stimulus=[1,0])
    (0, {0:0.6, 1:0.4})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, stimulus=[1,1])
    (0, {0:0.8, 1:0.2})
    >>> dec(np.array([0.3,0.7,0.2,0.05]), lastAct, stimulus=[0,0])
    (0, {0:0.8, 1:0.2})
    """

    expResponseSet = set(expResponses)

    def decisionFunc(probabilities, lastAction, stimulus=None, validResponses=None):

        probLen = len(probabilities)

        if probLen == 2:
            numStim = 1
        else:
            numStim = int(probLen / 2)

        if type(stimulus) is not NoneType and numStim == len(stimulus) and not (np.array(stimulus) == 0).all():
            respWeights = stimulus
        else:
            respWeights = np.ones(numStim)

        probSets = np.reshape(probabilities, (2, numStim))
        expectSet = np.sum(respWeights * probSets, 1)

        probPair = expectSet / np.sum(expectSet)

        probDict = collections.OrderedDict({k: v for k, v in itertools.izip(expResponses, probPair)})

        if type(validResponses) is not NoneType:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif len(validResponses) == 0:
                return None, probDict
            elif set(validResponses) != expResponseSet:
                warnings.warn("Bad validResponses: " + str(validResponses))
            elif len(validResponses) > 2:
                warnings.warn("Bad number of validResponses: " + str(validResponses))

        probDec = probPair[0]

        if abs(probDec-0.5) >= eta:
            if probDec > 0.5:
                decision = expResponses[0]
            elif probDec == 0.5:
                decision = random.choice(expResponses)
            else:
                decision = expResponses[1]
        else:
            decision = None

        return decision, probDict

    decisionFunc.Name = "binary.decEtaSets"
    decisionFunc.Params = {"expResponses": expResponses,
                           "eta": eta}

    return decisionFunc
