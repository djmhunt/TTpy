# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

A collection of decision making functions where there are only two possible actions
"""

from __future__ import division

from random import choice

def beta(responses = (0,1),beta = 0):
    """Decisions using a probability difference threshold
    
    Parameters
    ----------
    responses : tuple of length two, optional
        Provides the two action responses expected by the experiment
    beta : float, optional :math:`beta`
        The threshold for decisions. :math:`\Vert p_0-0.5\Vert> \beta`
        If true a decision is taken. If false the function responds ``None``
    
    Returns
    -------
    decisionFunc : function
        Calculates the decisions 
    
    """
    
    Name = "binaryBeta"
        
    def decisionFunc(probabilities):
        """
        Decisions using a probability difference threshold
    
        Parameters
        ----------
        probabilities : tuple of length two
            The probabilities of the two decisions. They are compared by: :math:`\left\Vert p_0-0.5\right\Vert>\Beta`
            .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
            .. math:: \left\Vert p_0-0.5\right\Vert>\Beta
            If true a decision is taken. If false the function responds ``None``
        
        Returns
        -------
        decision : int or None
            The vlaue of the decision
        """
        
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
            
        return decision
        
    return decisionFunc