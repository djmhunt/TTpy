# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are no limits on the
number of actions, but they are countable.
"""

from __future__ import division, print_function

from warnings import warn

from random import choice
from numpy import argmax, array, where, amax
from itertools import izip
from collections import OrderedDict

def decMaxProb(expResponses = None):
    """Decisions using a probability difference threshold

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

    See Also
    --------
    models.qLearn, models.qLearn2, models.OpAL

    Examples
    --------
    >>> from model.decision.discrete import decMaxProb
    >>> lastAct = 0
    >>> d = decMaxProb([0,1,2,3])
    >>> d([0.2,0.6,0.3,0.5], lastAct)
    ￼(1, {0:0.2,1:0.6,2:0.3,4:0.5})
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
    model\decision\discrete.py:66: UserWarning: Some of the validResponses are not in expResponses: [0, 2]
    warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
    ￼(3, {1:0.2,2:0.3,3:0.5})
    """

    expResp = array(expResponses)

    def decisionFunc(probabilities, lastAction, validResponses = None):

        if validResponses != None:
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
        probIndexes = where(prob==probMax)[0]

        decision = choice(resp[probIndexes])

        probs = OrderedDict({k:v for k,v in izip(expResponses,probabilities)})

        return decision, probs

    decisionFunc.Name = "discrete.decMaxProb"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc