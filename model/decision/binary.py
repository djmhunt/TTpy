# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import warnings
import collections

import numpy as np


def single(expResponses=(0, 1)):
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
    decision : int or None
        The action to be taken by the model
    probabilities : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    Examples
    --------
    >>> np.random.seed(100)
    >>> dec = single()
    >>> dec(0.23)
    (0, OrderedDict([(0, 0.77), (1, 0.23)]))
    >>> dec(0.23, 0)
    (0, OrderedDict([(0, 0.77), (1, 0.23)]))
    """

    expResponseSet = set(expResponses)

    def decisionFunc(prob, lastAction=0, validResponses=None):

        if validResponses is not None:
            if len(validResponses) == 1:
                resp = validResponses[0]
                return resp, collections.OrderedDict([(k, 1) if k == resp else (k, 0) for k in expResponseSet])
            elif len(validResponses) == 0:
                return None, collections.OrderedDict([(k, 1-prob) if k == lastAction else (k, prob) for k in expResponseSet])
            elif set(validResponses) != expResponseSet:
                warnings.warn("Bad validResponses: " + str(validResponses))
            else:
                warnings.warn("Bad number of validResponses: " + str(validResponses))

        randNum = np.random.rand()

        lastNotAction = list(expResponseSet.difference([lastAction]))[0]

        if prob >= randNum:
            # The decision is to switch
            decision = lastNotAction
        else:
            # Keep the same decision
            decision = lastAction

        pSet = {lastNotAction: prob,
                lastAction: 1-prob}

        probDict = collections.OrderedDict([(k, pSet[k]) for k in expResponses])

        return decision, probDict

    decisionFunc.Name = "binary.single"
    decisionFunc.Params = {}

    return decisionFunc
