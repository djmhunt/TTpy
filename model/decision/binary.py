# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""

from __future__ import division, print_function

from warnings import warn

from random import choice, random
from numpy import sum, array, arange, reshape
from itertools import izip
from collections import OrderedDict


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

    def decisionFunc(prob, lastAction, validResponses=None):

        if validResponses:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif set(validResponses) != expResponseSet:
                warn("Bad validResponses: " + str(validResponses))
            else:
                warn("Bad number of validResponses: " + str(validResponses))

        randNum = random()

        lastNotAction = list(expResponseSet.difference([lastAction]))[0]

        if prob >= randNum:
            # The decision is to switch
            decision = lastNotAction
        else:
            # Keep the same decision
            decision = lastAction

        pSet = {lastNotAction: prob,
                lastAction: 1-prob}

        probs = OrderedDict({k: pSet[k] for k in expResponses})

        return decision, probs

    decisionFunc.Name = "binary.decSingle"
    decisionFunc.Params = {}

    return decisionFunc


def decEta(expResponses=(0, 1), eta=0):
    """Decisions using a probability difference threshold

    Parameters
    ----------
    expResponses : tuple of length two, optional
        Provides the two action responses expected by the experiment
    eta : float, optional :math:`\\eta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \\eta`
        If true a decision is taken. If false the function responds ``None``

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision

    See Also
    --------
    models.BP, models.MS, models.EP, models.MS_rev, models.qLearn, models.qLearn2, models.OpAL

    """

    expResponseSet = set(expResponses)

    def decisionFunc(probabilities, lastAction, validResponses=None):

        if validResponses:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif set(validResponses) != expResponseSet:
                warn("Bad validResponses: " + str(validResponses))
            else:
                warn("Bad number of validResponses: " + str(validResponses))

        prob = probabilities[0]

        if abs(prob-0.5) >= eta:
            if prob > 0.5:
                decision = expResponses[0]
            elif prob == 0.5:
                decision = choice(expResponses)
            else:
                decision = expResponses[1]
        else:
            decision = None

        probs = OrderedDict({k: v for k, v in izip(expResponses, probabilities)})

        return decision, probs

    decisionFunc.Name = "binary.decEta"
    decisionFunc.Params = {"expResponses": expResponses,
                           "eta": eta}

    return decisionFunc


def decIntEtaReac(expResponses=(0, 1), eta=0):
    """
    Decisions using a probability difference threshold for the expectation
    from two sets of response value probabilities.

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

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision

    See Also
    --------
    models.BP, models.MS_rev, experiment.decks.deckStimAllInfo

    Examples
    --------
    >>> from numpy import array
    >>> from model.decision.binary import decIntEtaReac
    >>> lastAct = 0
    >>> dec = decIntEtaReac()
    >>> dec(array([0.4,0.1,0.25,0.25]), lastAct)
    (1, {0:0.44444444444, 1:0.5555555556})
    >>> dec(array([0.4,0.1,0.25,0.25]), lastAct, validResponses=[0])
    (0, [1])
    >>> dec(array([0.4,0.1,0.25,0.25]), lastAct, validResponses=[0,3])
    model\decision\binary.py:120: UserWarning: Bad number of validResponses: [0, 3]
    warn("Bad number of validResponses: " + str(validResponses))
    (1, {0:0.44444444444, 1:0.5555555556})

    """

    expResponseSet = set(expResponses)

    def decisionFunc(probabilities, lastAction, validResponses=None):

        if validResponses:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, {resp: 1}
            elif set(validResponses) != expResponseSet:
                warn("Bad validResponses: " + str(validResponses))
            else:
                warn("Bad number of validResponses: " + str(validResponses))

        numResp = int(len(probabilities) / 2)
        respWeights = arange(1, numResp + 1)

        probSets = reshape(probabilities, (2, numResp))
        expectSet = sum(respWeights * probSets, 1)

        probPair = expectSet / sum(expectSet)

        prob = probPair[0]

        if abs(prob-0.5) >= eta:
            if prob > 0.5:
                decision = expResponses[0]
            elif prob == 0.5:
                decision = choice(expResponses)
            else:
                decision = expResponses[1]
        else:
            decision = None

        probs = OrderedDict({k: v for k, v in izip(expResponses, probPair)})

        return decision, probs

    decisionFunc.Name = "binary.decIntEtaReac"
    decisionFunc.Params = {"expResponses": expResponses,
                           "eta": eta}

    return decisionFunc

