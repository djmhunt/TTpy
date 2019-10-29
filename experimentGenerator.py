# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import itertools
import copy

import utils


class ExperimentGen(object):

    """
    Generates experiment class instances based on an experiment and a set of varying parameters

    Parameters
    ----------
    experiment :  experiment.experimentTemplate.experiment
        The experiment that will be run
    parameters : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across experiment instances. When a parameter contains a list, an instance
        of the experiment will be created for every combination of this parameter with all the others.
    otherOptions : dictionary of float, string or binary valued elements
        These contain all the the experiment options that describe the experiment being studied but do not vary across experiment instances.
    """

    def __init__(self, experiment, parameters, otherOptions):
        """ """

        self.experimentClass = experiment
        self.experimentDetailList = params(parameters, otherOptions)

        self.countLen = len(self.experimentDetailList)

    def __iter__(self):
        """ Returns the iterator for the creation of experiments"""

        self.count = -1

        return self

    def next(self):
        """ Produces the next experiment instance for the iterator

        Returns
        -------
        instance : experiment.experimentTemplate.experiment instance
        """

        self.count += 1
        if self.count >= self.countLen:
            raise StopIteration

        return self.newExp(self.count)

    def iterExpID(self):
        """
        Yields the experiment IDs. To be used with self.newExp(expID) to receive the next experiment instance

        Returns
        -------
        expID : int
            The ID number that refers to the next experiment parameter combination.
        """

        for c in range(self.countLen):
            yield c

    def newExp(self, expNum):
        """
        Produces the next experiment instance

        Parameters
        ----------
        expNum : int
            The number of the experiment instance to be initialised

        Returns
        -------
        instance : experiment.experiment.experiment instance
        """

        if expNum >= self.countLen:
            return None

        record = self.experimentDetailList[expNum]

        return self.experimentClass(**record)


def params(parameters, otherOptions):
    """
    For the given experiment returns the appropriate list for constructing the experiment instances

    Parameters
    ----------
    parameters : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across experiment instances. When a parameter contains a list, an instance
        of the experiment will be created for every combination of this parameter with all the others.
    otherOptions : dictionary of float, string or binary valued elements
        These contain all the the experiment options that describe the experiment being studied but do not vary across experiment instances.

    Returns
    -------
    experimentDetails : list of dict
        Each dict contains the full set of arguments needed to initialise the experiment instance
    """

    if not parameters:
        return [otherOptions]

    paramKeys = parameters.keys()
    paramValues = parameters.values()

    paramCombs = utils.listMergeGen(*paramValues)

    experimentDetailList = []
    for p in paramCombs:

        args = {k: copy.copy(v) for k, v in itertools.izip(paramKeys, p)}
        for k, v in otherOptions.iteritems():
            args[k] = copy.copy(v)

        experimentDetailList.append(args)

    return experimentDetailList
