# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from itertools import izip

from utils import listMerGen

class models(object):

    """A factory that generates lists of model classes

    models(*modelSets)
    Recieves a series of model packages. the packages are in the form of:
    (model,variables, parameters)
    model:  A model class object
    variables: A dictionary of varibles with a list for their values
    parameters: A dictionary of other, probably text or binary parameters"""

    def __init__(self,*args):
        """ """

        self.models = []

        for a in args:
            model = a[0]
            variables = a[1]
            other = a[2]
            self._params(model,variables,other)

    def reset(self):
        """Resets the generator of models """

    def __iter__(self):
        """ Returns the iterator for the creation of models"""

        self.count = -1
        self.countLen = len(self.models)

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        self.count += 1
        if self.count >= self.countLen:
            raise StopIteration

        return (model(**record) for model,record in self.models[self.count])

    def _params(self,model, parameters, otherArgs):

        """ For the given model returns the appropreate list for constructing the model instances

        Each line has:
        (model, {dict of model arguments})
        """

        params = parameters.keys()
        paramVals = parameters.values()

        paramCombs = listMerGen(*paramVals)

        modelSet = []
        for p in paramCombs:

            args = {k:v for k,v in izip(params,p)}
            for k,v in otherArgs:
                args[k] = v

            modelSet.append([model, args])

        self.models.append(modelSet)