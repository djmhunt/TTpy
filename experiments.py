# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from itertools import izip

from utils import listMerGen

class experiments(object):

    """A factory that generates lists of experiment classes

    Parameters
    ----------
    args : a list of tuples of the form (experiment,variables, parameters)
        Each tuple is an an experiment package, describing an experiment and
        the different parameter combinations that will be tried.

    args tuples components

    experiment :  experiment.experiment.experiment
    variables : dictonary of floats or lists of floats
        Variables are the parameters that you are or are likely to change across
        model instances. When a variable contains a list, an instance of the
        experiment will be created for every combination of this variable with
        all the others.
    parameters : dictionary of float, string or binary valued elements
        These contain all the the experiment parameters that define the version
        of the experiment being studied.
    """

    def __init__(self,*args):
        """ """

        self.experiments = []

        for a in args:
            exp = a[0]
            variables = a[1]
            other = a[2]
            self._params(exp,variables,other)

        self.countLen = len(self.experiments)

    def __iter__(self):
        """ Returns the iterator for the creation of experiments"""

        self.count = -1

        return self

    def next(self):
        """ Produces the next item for the iterator

        Returns
        -------
        count : int
            The number of the next experiment instance. This is entered into
            experiments.create(count) to receive the experiment instance"""

        self.count += 1
        if self.count >= self.countLen:
            raise StopIteration

        return self.count

    def create(self,expNum):
        """
        Produces the next experiment instance

        Parameters
        ----------
        expNum : int
            The number of the experiment instance to be intiailised

        Returns
        -------
        instance : experiment.experiment.experiment instance
        """

        if expNum >= self.countLen:
            return None

        return self._instance(expNum)

    def _instance(self, expNum):

        record = self.experiments[expNum]

        exp = record[0]

        return exp(**record[1])

    def _params(self,exp, parameters, otherArgs):

        """
        For the given experiment returns the appropreate list for
        constructing the experiment instances

        Each line has:
        (exp, {dict of exp arguments})
        """

        if not parameters:
            self.experiments.append([exp,otherArgs])
            return

        params = parameters.keys()
        paramVals = parameters.values()

        paramCombs = listMerGen(*paramVals)

        experiments = []
        for p in paramCombs:

            args = {k:v for k,v in izip(params,p)}
            for k,v in otherArgs.iteritems():
                args[k] = v

            experiments.append([exp, args])

        self.experiments.extend(experiments)