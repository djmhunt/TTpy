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

        self.count = 0

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        record = self.models[self.count]

        model = record[0]

        self.count += 1

        return model(**record[1]), record[2], record[3]

    def _params(self,model, parameters, otherArgs):

        """ For the given model returns the appropreate list for constructing the model instances

        Each line has:
        (model, {dict of model arguments},descriptor,plotLabel)
        """

        params = parameters.keys()
        paramVals = parameters.values()
        name = model.Name

        paramCombs = listMerGen(*paramVals)

        labelCount = self.lastLabelID

        models = []
        for p in paramCombs:

            args = {k:v for k,v in izip(params,p)}
            for k,v in otherArgs:
                args[k] = v

            descriptors = [k + ' = ' + str(v).strip('[]()') for k,v in izip(params,p)]
            descriptor = name + ": " + ", ".join(descriptors)

            if len(descriptor)>18:
                plotLabel = name + ": " + "Group " + str(labelCount)
                labelCount += 1
            else:
                plotLabel = descriptor

            models.append([model, args, descriptor, plotLabel])

        self.models.extend(models)