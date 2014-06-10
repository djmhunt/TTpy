# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

from ittertools import izip

from utils import listMerGen

class models(object):

    """The documentation for the class"""

    def __init__(self,*args,**kwargs):
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

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        self.count += 1

        record = self.models[self.count]

        model = record[0]

        yield model(**record[1])

    def params(self):
        """Returns the relevent dictionary of parameters for the current model"""

        return self.models[self.count][1]

    def _params(self,model, parameters, otherArgs):

        """ For the given model returns the appropreate list for constructing the model instances

        Each line has:
        (model, {dict of model arguments})
        """

        params = parameters.keys()
        paramVals = parameters.values()

        paramCombs = listMerGen(*paramVals)

        models = []
        for p in paramCombs:

            args = {k:v for k,v in izip(params,p)}
            for k,v in otherArgs:
                args[k] = v

            models.append([model, args])

        self.models.extend(models)