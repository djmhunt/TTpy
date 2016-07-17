# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from itertools import izip
from numpy import amax, amin
from collections import OrderedDict
from copy import copy

from utils import listMerGen


class models(object):

    """A factory that generates lists of model classes

    Parameters
    ----------
    args : a list of tuples of the form (model, parameters, modifiers)
        Each tuple is a model package, describing a model and
        the different parameter combinations that will be tried.
        
    args tuples components

    model :  model.model.model
    parameters : dictionary containing floats or lists of floats
        Parameters are the variables that you are or are likely to change across 
        model instances. When a parameter contains a list, an instance of the 
        model will be created for every combination of this parameter with 
        all the others.
    modifiers : dictionary of float, string or binary valued elements
        These contain all the the model modifiers that define the version 
        of the model being studied.
    """

    def __init__(self, *args):
        """ """

        self.models = []

        for a in args:
            model = a[0]
            variables = OrderedDict(a[1])
            other = a[2]
            self.models.append((model, variables, other))

    def reset(self):
        """
        Resets the generator of models 
        
        Once this is reset the iterator will produce model instances from the 
        beginning again
        """

        self.count = -1

    def __iter__(self):
        """ Returns the iterator for the creation of models"""

        self.count = -1
        self.countLen = len(self.models)

        return self

    def next(self):
        """
        Produces the next item for the iterator
        
        Returns
        -------
        models : list of model.model.model instances
        """

        self.count += 1
        if self.count >= self.countLen:
            raise StopIteration

        model, variables, other = self.models[self.count]

        modelSet = self._params(model, variables, other)

        return (model(**record) for model, record in modelSet)

    def iterFitting(self):
        """ 
        Yields a list containing a model object and parameters to initialise them
        If a model variable was introduced with multiple values a value will 
        be taken half way between the max and min.
        
        This is expected to be used only for introducing models to the fitting
        functions in dataFitting.
        
        Returns
        -------
        model : model.model.model
            The model to be initialised
        initialVars : ordered dictionary of floats
            The model instance parameters
        otherArgs : dictionary of floats, strings and binary values
        """

        for m in self.models:

            model = m[0]
            otherArgs = m[2]
            initialVars = OrderedDict()
            for k, v in m[1].iteritems():
                if amax(v) == amin(v):
                    initialVars[k] = amax(v)
                else:
                    initialVars[k] = (amax(v)-amin(v))/2.0
#            initialVars = {k:(amax(v)-amin(v))/2.0 for k,v in m[1].iteritems()}

            yield (model, initialVars, otherArgs)

    def _params(self, model, parameters, otherArgs):

        """ 
        For the given model returns a list of all that goes in to the model.

        Each line has:         (model, {dict of model arguments})
        
        Parameters
        ----------
        model : model.model.model
            The model to be initialised
        parameters : dictonary (preferably ordered) containing floats or lists of floats
            Frequently changing values of the model. When a parameter contains 
            a list, an version of the model will be set-up for every combination 
            of this parameter with all the others.
        otherArgs : dictionary of float, string or binary valued elements
            These contain all the the model modifiers that define the version 
            of the model being studied.
            
        Returns
        -------
        modelSet : list of tuples with two elements
            Each tuple contains a (model,args) pair, where args is a unique 
            combination of the parameters and otherArgs in the form of a dict.
        """

        params = parameters.keys()
        paramVals = parameters.values()

        paramCombs = listMerGen(*paramVals)

        modelSet = []
        for p in paramCombs:

            args = {k: copy(v) for k, v in izip(params, p)}
            for k, v in otherArgs.iteritems():
                args[k] = copy(v)

            modelSet.append([model, args])

        return modelSet
