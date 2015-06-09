# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are no limits on the 
number of actions, but they are countable.
"""

from __future__ import division

from random import choice
from numpy import argmax, array, where, amax
from itertools import izip
from collections import OrderedDict

def decMaxProb(responses = None):
    """Decisions using a probability difference threshold
    
    Parameters
    ----------
    responses : tuple or None, optional
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
    >>> d = decMaxProb([0,1,2,3])
    >>> d([0.2,0.6,0.3,0.5])
    ￼(1, {0:0.2,1:0.6,2:0.3,4:0.5})
    >>> d([0.2,0.5,0.3,0.5])
    ￼(1, {0:0.2,1:0.5,2:0.3,3:0.5})
    >>> d([0.2,0.5,0.3,0.5])
    ￼(3, {0:0.2,1:0.5,2:0.3,3:0.5})
    
    """
    
    resp = array(responses)
        
    def decisionFunc(probabilities):
        
        prob = probabilities
                
        probMax = amax(prob)
        
        # In case there are multiple choices with the same probability, pick
        # one at random
        probIndexes = where(prob==probMax)[0]
        
        decision = choice(resp[probIndexes])
            
        probs = OrderedDict({k:v for k,v in izip(responses,prob)})
            
        return decision, probs
        
    decisionFunc.Name = "discrete.decMaxProb"
    decisionFunc.Params = {"responses": resp}
        
    return decisionFunc