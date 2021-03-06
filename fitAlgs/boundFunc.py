# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

import numpy as np


def infBound(base=0):
    """Boundary excess of ``inf`` when over bounds

    Parameters
    ----------
    base : float, optional
        The cost at the boundary. Default 0

    Returns
    -------
    cost : function
        Calculates the cost of exceeding the bounday using the parameters and
        the boundaries, and returns the cost.

    Examples
    --------
    >>> cst = infBound(base = 160)
    >>> cst([0.5, 2], [(0, 1), (0, 5)])
    160
    >>> cst([0.5, 7], [(0, 1), (0, 5)])
    inf
    """

    response = np.inf

    def cost(parameters, bounds):

        boundArr = np.array(bounds)

        if (parameters < boundArr[:, 0]).any() or any(parameters > boundArr[:, 1]):
            return response
        else:
            return base

    cost.Name = "boundFunc.infBound"
    cost.Params = {"base": base}

    return cost


def scalarBound(base=0):
    """Boundary excess calculated as a scalar increase based on difference with
    bounds

    Parameters
    ----------
    base : float, optional
        The cost at the boundary. Default 0

    Returns
    -------
    cost : function
        Calculates the cost of exceeding the boundary using the parameters and
        the boundaries, and returns the cost.

    Examples
    --------
    >>> cst = scalarBound(base=160)
    >>> cst([0.5, 2], [(0, 1), (0, 5)])
    160.0
    >>> cst([0.5, 7], [(0, 1), (0, 5)])
    162.0
    """

    def cost(parameters, bounds):

        boundArr = np.array(bounds)

        minOut = sum((boundArr[:, 0] - parameters) * (parameters < boundArr[:, 0]))

        maxOut = sum((parameters - boundArr[:, 1]) * (parameters > boundArr[:, 1]))

        response = base + minOut + maxOut

        return response

    cost.Name = "boundFunc.scalarBound"
    cost.Params = {"base": base}

    return cost
