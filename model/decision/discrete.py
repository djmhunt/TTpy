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


def weightProb(task_responses=(0, 1)):
    """Decisions for an arbitrary number of choices

    Choice made by choosing randomly based on which are valid and what their associated probabilities are

    Parameters
    ----------
    task_responses : tuple
        Provides the action responses expected by the task for each
        probability estimate.

    Returns
    -------
    decision_function : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or None
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
    >>> d([0.2, 0.5, 0.3, 0.5], trial_responses=[0, 2])
    (2, OrderedDict([(0, 0.4), (1, 0), (2, 0.6), (3, 0)]))
    >>> d = weightProb(["A", "B", "C"])
    >>> d([0.2, 0.3, 0.5], trial_responses=["A", "B"])
    (u'B', OrderedDict([(u'A', 0.4), (u'B', 0.6), (u'C', 0)]))
    >>> d([0.2, 0.3, 0.5], trial_responses=[])
    (None, OrderedDict([(u'A', 0.2), (u'B', 0.3), (u'C', 0.5)]))
    """

    def decision_function(probabilities, last_action=None, trial_responses=None):

        probArray = np.array(probabilities).flatten()

        trial_probabilities, valid_responses = _validProbabilities(probArray, task_responses, trial_responses)

        if trial_probabilities is None:
            return None, collections.OrderedDict([(k, v) for k, v in itertools.izip(task_responses, probArray)])

        normalised_trial_probabilities = trial_probabilities / np.sum(trial_probabilities)

        decision = np.random.choice(valid_responses, p=normalised_trial_probabilities)

        abridged_probability_dict = {k: v for k, v in itertools.izip(valid_responses, normalised_trial_probabilities)}
        probability_list = [(k, abridged_probability_dict[k]) if k in valid_responses else (k, 0) for k in task_responses]
        probDict = collections.OrderedDict(probability_list)

        return decision, probDict

    decision_function.Name = "discrete.weightProb"
    decision_function.Params = {"task_responses": task_responses}

    return decision_function


def maxProb(task_responses=(0, 1)):
    """Decisions for an arbitrary number of choices

    Choice made by choosing the most likely

    Parameters
    ----------
    task_responses : tuple
        Provides the action responses expected by the tasks for each
        probability estimate.

    Returns
    -------
    decision_function : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or None
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
    >>> d([0.2, 0.3, 0.5], trial_responses=[1, 2])
    (2, OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.3, 0.5], trial_responses=[])
    (None, OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
    >>> d = maxProb(["A", "B", "C"])
    >>> d([0.6, 0.3, 0.5], trial_responses=["A", "B"])
    ('A', OrderedDict([('A', 0.6), ('B', 0.3), ('C', 0.5)]))
    """

    def decision_function(probabilities, last_action=None, trial_responses=None):

        probArray = np.array(probabilities).flatten()

        probDict = collections.OrderedDict([(k, v) for k, v in itertools.izip(task_responses, probArray)])

        trial_probabilities, responses = _validProbabilities(probArray, task_responses, trial_responses)

        if trial_probabilities is None:
            return None, probDict

        max_probability = np.amax(trial_probabilities)
        max_responses = [r for r, p in itertools.izip(responses, trial_probabilities) if p == max_probability]
        decision = np.random.choice(max_responses)

        return decision, probDict

    decision_function.Name = "discrete.maxProb"
    decision_function.Params = {"task_responses": task_responses}

    return decision_function


def probThresh(task_responses=(0, 1), eta=0.8):
    # type : (list, float) -> (float, collections.OrderedDict)
    """Decisions for an arbitrary number of choices

    Choice made by choosing when certain (when probability above a certain value), otherwise randomly

    Parameters
    ----------
    task_responses : tuple
        Provides the action responses expected by the tasks for each
        probability estimate.
    eta : float, optional
        The value above which a non-random decision is made. Default value is 0.8

    Returns
    -------
    decision_function : function
        Calculates the decisions based on the probabilities and returns the
        decision and the probability of that decision
    decision : int or None
        The action to be taken by the model
    probDict : OrderedDict of valid responses
        A dictionary of considered actions as keys and their associated probabilities as values

    Examples
    --------
    >>> np.random.seed(100)
    >>> d = probThresh(task_responses=[0, 1, 2, 3], eta=0.8)
    >>> d([0.2, 0.8, 0.3, 0.5])
    (1, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.8, 0.3, 0.5], trial_responses=[0, 2])
    (0, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d([0.2, 0.8, 0.3, 0.5], trial_responses=[])
    (None, OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
    >>> d = probThresh(["A","B","C"])
    >>> d([0.2, 0.3, 0.8], trial_responses=["A", "B"])
    ('A', OrderedDict([('A', 0.2), ('B', 0.3), ('C', 0.8)]))
    """

    def decision_function(probabilities, last_action=None, trial_responses=None):

        probArray = np.array(probabilities).flatten()

        probDict = collections.OrderedDict([(k, v) for k, v in itertools.izip(task_responses, probArray)])

        trial_probabilities, responses = _validProbabilities(probArray, task_responses, trial_responses)

        if trial_probabilities is None:
            return None, probDict

        # If probMax is above a threshold, we pick the best one, otherwise we pick at random
        eta_responses = [r for r, p in itertools.izip(responses, trial_probabilities) if p >= eta]
        if eta_responses:
            decision = np.random.choice(eta_responses)
        else:
            decision = np.random.choice(responses)

        return decision, probDict

    decision_function.Name = "discrete.probThresh"
    decision_function.Params = {"task_responses": task_responses,
                                "eta": eta}

    return decision_function


def _validProbabilities(probabilities, task_responses, trial_responses):
    """
    Takes the list of probabilities, valid responses and possible responses and returns the appropriate probabilities
    and responses

    Parameters
    ----------
    probabilities : 1D list or array
        The probabilities for all possible actions
    task_responses : tuple or None
        Provides the action responses expected by the tasks for each
        probability estimate.
    trial_responses : 1D list or array, or ``None``
        The responses allowed for this trial. If ``None`` all are used.

    Returns
    -------
        probabilities : 1D list or array or None
            The probabilities to be evaluated in this trial
        responses: 1D list or None
            The responses associated with each probability

    Examples
    --------
    >>> _validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["B", "C"])
    ([0.1, 0.7], ['B', 'C'])
    """

    if trial_responses is None:
        responses = task_responses
        reduced_probabilities = probabilities
    else:
        responses = [r for r in task_responses if r in trial_responses]

        if not responses:
            responses = None
            reduced_probabilities = None
        else:
            reduced_probabilities = [probabilities[i] for i, r in enumerate(task_responses) if r in trial_responses]

    return reduced_probabilities, responses

