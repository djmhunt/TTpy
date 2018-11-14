# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

from __future__ import division, print_function, unicode_literals, absolute_import

from numpy import array


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
    >>> cst([0.5, 2], [(0, 1),(0, 5)])
    160
    >>> cst([0.5, 7], [(0, 1),(0, 5)])
    inf
    >>> cst([2, 7], [(0, 1),(0, 5)])
    inf
    >>> cst([-1, 7], [(0, 1),(0, 5)])
    inf
    >>> cst([-1, -2], [(0, 1),(0, 5)])
    inf
    """

    response = float("inf")

    def cost(parameters, bounds, fitQualFunc):

        boundArr = array(bounds)

        if any(parameters < boundArr[:, 0]) or any(parameters > boundArr[:, 1]):
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
    >>> cst([0.5, 2], [(0, 1),(0, 5)])
    160
    >>> cst([0.5, 7], [(0, 1),(0, 5)])
    162
    >>> cst([2, 7], [(0, 1),(0, 5)])
    163
    >>> cst([-1, 7], [(0, 1),(0, 5)])
    163
    >>> cst([-1, -2], [(0, 1),(0, 5)])
    163
    """

    def cost(parameters, bounds, fitQualFunc):

        boundArr = array(bounds)

        minOut = sum((boundArr[:, 0] - parameters) * (parameters < boundArr[:, 0]))

        maxOut = sum((parameters - boundArr[:, 1]) * (parameters > boundArr[:, 1]))

        response = base + minOut + maxOut

        return response

    cost.Name = "boundFunc.scalarBound"
    cost.Params = {"base": base}

    return cost
