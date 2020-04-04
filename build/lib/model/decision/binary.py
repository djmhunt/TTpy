# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""
import warnings
import collections

import numpy as np


def single(task_responses=(0, 1)):
    """Decisions using a switching probability

    Parameters
    ----------
    task_responses : tuple of length two, optional
        Provides the two action responses expected by the task

    Returns
    -------
    decision_function : function
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

    def decision_function(prob, last_action=0, trial_responses=None):

        if trial_responses is not None:
            if len(trial_responses) == 1:
                resp = trial_responses[0]
                return resp, collections.OrderedDict([(k, 1) if k == resp else (k, 0) for k in task_responses])
            elif len(trial_responses) == 0:
                return None, collections.OrderedDict([(k, 1-prob) if k == last_action else (k, prob) for k in task_responses])
            elif set(trial_responses) != task_responses:
                warnings.warn("Bad trial_responses: " + str(trial_responses))
            else:
                warnings.warn("Bad number of trial_responses: " + str(trial_responses))

        randNum = np.random.rand()

        lastNotAction = [action for action in task_responses if action != last_action][0]

        if prob >= randNum:
            # The decision is to switch
            decision = lastNotAction
        else:
            # Keep the same decision
            decision = last_action

        pSet = {lastNotAction: prob,
                last_action: 1-prob}

        probDict = collections.OrderedDict([(k, pSet[k]) for k in task_responses])

        return decision, probDict

    decision_function.Name = "binary.single"
    decision_function.Params = {}

    return decision_function
