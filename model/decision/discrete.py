# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are no limits on the
number of actions, but they are countable.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

from warnings import warn

from random import choice
from numpy import argmax, array, where, amax, ones, reshape, sum
from itertools import izip
from collections import OrderedDict
from types import NoneType


def decMaxProb(expResponses=None):
    """Decisions for an arbitrary number of choices

    Choice made by choosing the most likely

    Parameters
    ----------
    expResponses : tuple or None, optional
        Provides the action responses expected by the experiment for each
        probability estimate. If ``None`` then the responses for :math:`N`
        probabilities will be :math:`\\left[0,1,\\ldots,N-1\\right]`

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
    models.qLearn, models.qLearn2, models.OpAL

    Examples
    --------
    >>> from model.decision.discrete import decMaxProb
    >>> lastAct = 0
    >>> d = decMaxProb([0,1,2,3])
    >>> d([0.2,0.6,0.3,0.5], lastAct)
    ￼(1, {0:0.2,1:0.6,2:0.3,3:0.5})
    >>> d([0.2,0.5,0.3,0.5], lastAct)
    ￼(1, {0:0.2,1:0.5,2:0.3,3:0.5})
    >>> d([0.2,0.5,0.3,0.5], lastAct)
    ￼(3, {0:0.2,1:0.5,2:0.3,3:0.5})
    >>> d([0.2,0.5,0.3,0.5], lastAct, validResponses=[0,2])
    ￼(2, {0:0.2,1:0.5,2:0.3,3:0.5})
    >>> d = decMaxProb([1,2,3])
    >>> d([0.2,0.3,0.5], lastAct, validResponses=[1,2])
    ￼(2, {1:0.2,2:0.3,3:0.5})
    >>> d([0.2,0.3,0.5], lastAct, validResponses=[0,2])
    model\decision\discrete.py:83: UserWarning: Some of the validResponses are not in expResponses: [0, 2]
    warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
    ￼(3, {1:0.2,2:0.3,3:0.5})
    """

    expResp = array(expResponses)

    def decisionFunc(probabilities, lastAction, stimulus=None, validResponses=None):

        probDict = OrderedDict({k: v for k, v in izip(expResponses, probabilities)})

        if type(validResponses) is not NoneType:
            if len(validResponses) == 0:
                return None, probDict
            resp = array([r for r in expResp if r in validResponses])
            if len(resp) != len(validResponses):
                warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
                resp = expResp
                prob = probabilities
            else:
                prob = array([probabilities[i] for i, r in enumerate(expResp) if r in validResponses])
        else:
            resp = expResp
            prob = probabilities

        probMax = amax(prob)

        # In case there are multiple choices with the same probability, pick
        # one at random
        probIndexes = where(prob == probMax)[0]

        decision = choice(resp[probIndexes])

        return decision, probDict

    decisionFunc.Name = "discrete.decMaxProb"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc

def decMaxProbSets(expResponses=None):
    """Decisions for an arbitrary number of sets of action-response value probabilities

    Choice made by choosing the most likely

    Parameters
    ----------
    expResponses : tuple or None, optional
        Provides the action responses expected by the experiment for each
        probability estimate. If ``None`` then the responses for :math:`N`
        probabilities will be :math:`\\left[0,1,\\ldots,N-1\\right]`

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
    models.qLearn, models.qLearn2, models.OpAL

    Examples
    --------
    >>> from model.decision.discrete import decMaxProbSets
    >>> lastAct = 0
    >>> d = decMaxProbSets([0,1,2])
    >>> d([0.1,0.4,0.2,0.3,0.5,0.5], lastAct)
    ￼(2, {0:0.25,1:0.25,2:0.5})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct)
    ￼(1, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct)
    ￼(2, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, validResponses=[0,2])
    ￼(2, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, validResponses=[0])
    ￼(0, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, validResponses=[0,3])
    model\decision\discrete.py:181: UserWarning: Some of the validResponses are not in expResponses: [0, 3]
    warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
    ￼(2, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, validResponses=[])
    ￼(None, {0:0.2,1:0.4,2:0.4})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, stimulus=[1,0])
    ￼(2, {0:0.1,1:0.4,2:0.5})
    >>> d([0.1,0.4,0.4,0.6,0.5,0.5], lastAct, stimulus=[0,0])
    ￼(2, {0:0.2,1:0.4,2:0.4})
    """

    expResp = array(expResponses)
    expRespSize = len(expResp)

    def decisionFunc(probabilities, lastAction, stimulus=None, validResponses=None):

        numStim = int(len(probabilities) / expRespSize)

        if type(stimulus) is not NoneType and numStim == len(stimulus) and not (array(stimulus) == 0).all():
            respWeights = stimulus
        else:
            #warn("Stimuli invalid: " + repr(stimulus) + ". Ignoring stimuli")
            respWeights = ones(numStim)

        probLists = reshape(probabilities, (expRespSize, numStim))
        expectList = sum(respWeights * probLists, 1)

        probNormList = expectList / sum(expectList)

        probDict = OrderedDict({k: v for k, v in izip(expResp, probNormList)})

        if type(validResponses) is not NoneType:
            if len(validResponses) == 0:
                return None, probDict
            resp = array([r for r in expResp if r in validResponses])
            if len(resp) != len(validResponses):
                warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
                resp = expResp
                prob = probNormList
            else:
                prob = array([probNormList[i] for i, r in enumerate(expResp) if r in validResponses])
        else:
            resp = expResp
            prob = probNormList

        probMax = amax(prob)

        # In case there are multiple choices with the same probability, pick
        # one at random
        probIndexes = where(prob == probMax)[0]

        decision = choice(resp[probIndexes])

        return decision, probDict

    decisionFunc.Name = "discrete.decMaxProbSets"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc
