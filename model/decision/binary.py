# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""
import warnings

import numpy as np

from typing import Union, Tuple, List, Optional, Dict, Callable, NewType

Action = NewType('Action', Union[int, str])


def single(task_responses: List[Union[int, str]] = (0, 1)
           ) -> Callable[[List[float], Optional[Action], Optional[List[Action]]], Tuple[Optional[Action], Dict[Action, float]]]:
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
    >>> dec([0.23])
    (0, {0: 0.77, 1: 0.23})
    >>> dec([0.23], 0)
    (0, {0: 0.77, 1: 0.23})
    """

    def decision_function(probabilities: List[float],
                          last_action: Optional[Action] = 0,
                          trial_responses: Optional[List[Action]] = None
                          ) -> Tuple[Optional[Action], Dict[Action, float]]:

        prob = probabilities[0]

        if trial_responses is not None:
            if len(trial_responses) == 1:
                resp = trial_responses[0]
                return resp, dict([(k, 1) if k == resp else (k, 0) for k in task_responses])
            elif len(trial_responses) == 0:
                return None, dict([(k, 1-prob) if k == last_action else (k, prob) for k in task_responses])
            elif set(trial_responses) != task_responses:
                warnings.warn("Bad trial_responses: " + str(trial_responses))
            else:
                warnings.warn("Bad number of trial_responses: " + str(trial_responses))

        rand_num = np.random.rand()

        last_not_action = [action for action in task_responses if action != last_action][0]

        if prob >= rand_num:
            # The decision is to switch
            decision = last_not_action
        else:
            # Keep the same decision
            decision = last_action

        p_set = {last_not_action: prob,
                 last_action: 1-prob}

        prob_dict = {k: p_set[k] for k in task_responses}

        return decision, prob_dict

    decision_function.Name = "binary.single"
    decision_function.Params = {}

    return decision_function
