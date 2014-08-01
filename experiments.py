# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from itertools import izip

from utils import listMerGen

class experiments(object):

    """A factory that generates lists of experiment classes

    experiments(*experimentSets)
    Recieves a series of experiment packages. the packages are in the form of:
    (experiments,variables, parameters)
    experiments:  An experiments class object
    variables: A dictionary of varibles with a list for their values
    parameters: A dictionary of other, probably text or binary parameters"""

    def __init__(self,*args):
        """ """

        self.experiments = []

        for a in args:
            exp = a[0]
            variables = a[1]
            other = a[2]
            self._params(exp,variables,other)

    def __iter__(self):
        """ Returns the iterator for the creation of experiments"""

        self.count = -1
        self.countLen = len(self.experiments)

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        self.count += 1
        if self.count >= self.countLen:
            raise StopIteration

        return self.count

    def create(self,expNum):

        return self._instance(expNum)

    def _instance(self, expNum):

        record = self.experiments[expNum]

        exp = record[0]

        return exp(**record[1])

    def _params(self,exp, parameters, otherArgs):

        """ For the given experiment returns the appropreate list for constructing the experiment instances

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
            for k,v in otherArgs:
                args[k] = v

            experiments.append([exp, args])

        self.experiments.extend(experiments)