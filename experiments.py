# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from ittertools import izip

from utils import listMerGen

class experiments(object):

    """The documentation for the class"""

    def __init__(self,*args,**kwargs):
        """ """

        self.experiments = []

        for a in args:
            exp = a[0]
            variables = a[1]
            other = a[2]
            self._params(exp,variables,other)

    def reset(self):
        """Resets the generator of experiments """

    def __iter__(self):
        """ Returns the iterator for the creation of experiments"""

        self.count = -1

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        self.count += 1

        record = self.experiments[self.count]

        exp = record[0]

        yield exp(**record[1])

    def params(self):
        """Returns the relevent dictionary of parameters for the current experiment"""

        return self.experiments[self.count][1]

    def _params(self,exp, parameters, otherArgs):

        """ For the given experiment returns the appropreate list for constructing the experiment instances

        Each line has:
        (exp, {dict of exp arguments})
        """

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