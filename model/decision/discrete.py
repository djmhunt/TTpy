# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are no limits on the
number of actions, but they are countable.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import warnings
import itertools
import collections

import numpy as np

from types import NoneType


# TODO: provide default values for expResponses
def weightProb(expResponses=(0, 1)):
    """Decisions for an arbitrary number of choices

    Choice made by choosing randomly based on which are valid and what their associated probabilities are

    Parameters
    ----------
    expResponses : tuple
        Provides the action responses expected by the experiment for each
        probability estimate.

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
    models.QLearn, models.QLearn2, models.OpAL

    Examples
    --------
    >>> np.random.seed(100)
    >>> d = weightProb([0, 1, 2, 3])
    >>> d([0.4, 0.8, 0.3, 0.5])
    (1, OrderedDict([(0, 0.2), (1, 0.4), (2, 0.15), (3, 0.25)]))
    >>> d([0.1, 0.3, 0.4, 0.2])
    (1, OrderedDict([(0, 0.1), (1, 0.3), (2, 0.4), (3, 0.2)]))
    >>> d([0.2, 0.5, 0.3, 0.5], validResponses=[0, 2])
    (2, OrderedDict([(0, 0.4), (1, 0), (2, 0.6), (3, 0)]))
    >>> d = weightProb(["A", "B", "C"])
    >>> d([0.2, 0.3, 0.5], validResponses=["A", "B"])
    (u'B', OrderedDict([(u'A', 0.4), (u'B', 0.6), (u'C', 0)]))
    >>> d([0.2, 0.3, 0.5], validResponses=[])
    (None, OrderedDict([(u'A', 0.2), (u'B', 0.3), (u'C', 0.5)]))
    """

    expResp = np.array(expResponses)

    def decisionFunc(probabilities, lastAction=None, validResponses=None):

        probArray = np.array(probabilities).flatten()

        probDict = collections.OrderedDict([(k, v) for k, v in itertools.izip(expResponses, probArray)])

        prob, resp = _validProbabilities(probArray, expResp, validResponses)

        if type(prob) is NoneType:
            return None, probDict

        normProb = prob / np.sum(prob)

        decision = np.random.choice(resp, p=normProb)

        abridgedProbDict = {k: v for k, v in itertools.izip(resp, normProb)}
        abridgedProbDict.update({k: 0 for k in expResp if k not in resp})
        probDict = collections.OrderedDict([(k, abridgedProbDict[k]) for k in expResp])

        return decision, probDict

    decisionFunc.Name = "discrete.weightProb"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc


def maxProb(expResponses=(0, 1)):
    """Decisions for an arbitrary number of choices

    Choice made by choosing the most likely

    Parameters
    ----------
    expResponses : tuple
        Provides the action responses expected by the experiment for each
        probability estimate.

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
    models.QLearn, models.QLearn2, models.OpAL

    Examples
    --------
    >>> np.random.seed(100)
    >>> d = maxProb([1,2,3])
    >>> d([0.6, 0.3, 0.5])
    (1, OrderedDict([(1, 0.6), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.3, 0.5], validResponses=[1, 2])
    (2, OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.3, 0.5], validResponses=[])
    (None, OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
    >>> d = maxProb(["A", "B", "C"])
    >>> d([0.6, 0.3, 0.5], validResponses=["A", "B"])
    ('A', OrderedDict([('A', 0.6), ('B', 0.3), ('C', 0.5)]))
    """

    expResp = np.array(expResponses)

    def decisionFunc(probabilities, lastAction=None, validResponses=None):

        probArray = np.array(probabilities).flatten()

        probDict = collections.OrderedDict([(k, v) for k, v in itertools.izip(expResponses, probArray)])

        prob, resp = _validProbabilities(probArray, expResp, validResponses)

        if type(prob) is NoneType:
            return None, probDict

        # In case there are multiple choices with the same probability, pick
        # one at random
        probIndexes = np.where(prob == np.amax(prob))[0]

        decision = np.random.choice(resp[probIndexes])

        return decision, probDict

    decisionFunc.Name = "discrete.maxProb"
    decisionFunc.Params = {"expResponses": expResponses}

    return decisionFunc


def probThresh(expResponses=(0, 1), eta=0.8):
    # type : (list, float) -> (float, collections.OrderedDict)
    """Decisions for an arbitrary number of choices

    Choice made by choosing when certain (when probability above a certain value), otherwise randomly

    Parameters
    ----------
    expResponses : tuple
        Provides the action responses expected by the experiment for each
        probability estimate.
    eta : float, optional
        The value above which a non-random decision is made. Default value is 0.8

    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or NoneType
        The action to be taken by the model
    probDict : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    Examples
    --------
    >>> np.random.seed(100)
    >>> d = probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
    >>> d([0.2, 0.8, 0.3, 0.5])
    (1, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.8, 0.3, 0.5], validResponses=[0, 2])
    (0, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.8, 0.3, 0.5], validResponses=[])
    (None, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d = probThresh(["A","B","C"])
    >>> d([0.2, 0.3, 0.8], validResponses=["A", "B"])
    ('A', OrderedDict([('A', 0.2), ('B', 0.3), ('C', 0.8)]))
    """

    expResp = np.array(expResponses)

    def decisionFunc(probabilities, lastAction=None, validResponses=None):

        probArray = np.array(probabilities).flatten()

        probDict = collections.OrderedDict([(k, v) for k, v in itertools.izip(expResponses, probArray)])

        prob, resp = _validProbabilities(probArray, expResp, validResponses)

        if type(prob) is NoneType:
            return None, probDict

        # If probMax is above a threshold, we pick the best one, otherwise we pick at random
        if np.amax(prob) >= eta:
            probIndexes = np.where(prob >= eta)[0]
            decision = np.random.choice(resp[probIndexes])
        else:
            decision = np.random.choice(resp)

        return decision, probDict

    decisionFunc.Name = "discrete.probThresh"
    decisionFunc.Params = {"expResponses": expResponses,
                           "eta": eta}

    return decisionFunc


def _validProbabilities(probabilities, expResp, validResponses):
    """
    Takes the list of probabilities, valid responses and possible responses and returns the appropriate probabilities
    and responses

    Parameters
    ----------
    probabilities : 1D list or array
        The probabilities for all possible actions
    expResp : tuple or None
        Provides the action responses expected by the experiment for each
        probability estimate.
    validResponses : 1D list or array
        The responses allowed for this trial

    Returns
    -------
        probabilities : 1D list or array or None
            The probabilities to be evaluated in this trial
        responses: 1D list or None
            The responses associated with each probability

    Examples
    --------
    >>> _validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["B", "C"])
    (array([0.1, 0.7]), array(['B', 'C'], dtype='<U1'))
    """

    if type(validResponses) is NoneType:
        resp = expResp
        prob = probabilities
    else:
        resp = np.array([r for r in expResp if r in validResponses])
        prob = np.array([probabilities[i] for i, r in enumerate(expResp) if r in validResponses])
        if len([r for r in validResponses if r not in expResp]) > 0:
            warnings.warn("Some of the validResponses are not in expResponses: " + repr(validResponses))
        elif len(resp) != len(validResponses):
            warnings.warn("Some of the validResponses are repeated: " + repr(validResponses))
        elif len(validResponses) == 0:
            resp = None
            prob = None

    return prob, resp

