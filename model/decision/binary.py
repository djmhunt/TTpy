# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""

from __future__ import division

from random import choice
from numpy import sum, array, arange, reshape

def decBeta(responses = (0,1),beta = 0):
    """Decisions using a probability difference threshold
    
    Parameters
    ----------
    responses : tuple of length two, optional
        Provides the two action responses expected by the experiment
    beta : float, optional :math:`\\beta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \\beta`
        If true a decision is taken. If false the function responds ``None``
    
    Returns
    -------
    decisionFunc : function
        Calculates the decisions based on the probabilities and returns the 
        decision and the probability of that decision
        
    See Also
    --------
    models.BP, models.MS, models.EP, models.MS_rev, models.qLearn, models.qLearn2
    
    """
        
    def decisionFunc(probabilities):
                
        prob = probabilities[0]

        if abs(prob-0.5)>beta:
            if prob>0.5:
                decision = responses[0]
            elif prob == 0.5:
                decision = choice(responses)
            else:
                decision = responses[1]
        else:
            decision = None
            
        return decision, probabilities
        
    decisionFunc.Name = "binary.decBeta"
    decisionFunc.Params = {"responses": responses,
                           "beta": beta}
        
    return decisionFunc

def decIntBetaReac(responses = (0,1), beta = 0):
    """
    Decisions using a probability difference threshold for the expectation
    from two sets of response value probabilities.
    
    It is assumed that the response values are increasing and evenly spaced.
    
    The two sets of probabilities are provided as a one dimtentional array, 
    with one set after the other.
    
    Parameters
    ----------
    responses : tuple of length two, optional
        Provides the two action responses expected by the experiment
    beta : float, optional :math:`\\beta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \\beta`
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
    >>> from model.decision.binary import decIntBetaReac
    >>> dec = decIntBetaReac()
    >>> dec(array([0.4,0.1,0.25,0.25]))
    1
    
    """
        
    def decisionFunc(probabilities):
        
        numResp = int(len(probabilities) / 2)
        respWeights = arange(numResp)
            
        probSets = reshape(probabilities,(2,numResp))
        expectSet = sum(respWeights * probSets,1)
        
        probPair = expectSet / sum(expectSet)
        
        prob = probPair[0]

        if abs(prob-0.5)>beta:
            if prob>0.5:
                decision = responses[0]
            elif prob == 0.5:
                decision = choice(responses)
            else:
                decision = responses[1]
        else:
            decision = None
            
        return decision, probPair
        
    decisionFunc.Name = "binary.decIntBetaReac"
    decisionFunc.Params = {"responses": responses,
                           "beta": beta}
        
    return decisionFunc