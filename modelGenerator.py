# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import itertools
import collections
import copy

import numpy as np

import utils


class ModelGen(object):

    """
    Generates model class instances based on a model and a set of varying parameters

    Parameters
    ----------
    model :  model.modelTemplate.Model
    parameters : dictionary containing floats or lists of floats
        Parameters are the options that you are or are likely to change across 
        model instances. When a parameter contains a list, an instance of the 
        model will be created for every combination of this parameter with 
        all the others.
    otherOptions : dictionary of float, string or binary valued elements
        These contain all the the model options that define the version 
        of the model being studied.
    """

    def __init__(self, model, parameters, otherOptions):

        self.count = -1

        self.modelClass = model
        self.otherOptions = otherOptions

        self.modelDetailList, self.paramCombList = params(parameters, otherOptions)

        self.countLen = len(self.modelDetailList)

    def __iter__(self):
        """ 
        Returns the iterator for the creation of models
        """

        self.count = -1

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

        record = self.modelDetailList[self.count]

        return self.modelClass(**record)

    def iterInitDetails(self):
        """ 
        Yields a list containing a model object and parameters to initialise them
        
        Returns
        -------
        model : model.modelTemplate.Model
            The model to be initialised
        parameters : ordered dictionary of floats or bools
            The model instance parameters
        otherOptions : dictionary of floats, strings and binary values
        """

        for p in self.paramCombList:
            yield (self.modelClass, p, self.otherOptions)


def params(parameters, otherOptions):
    """
    For the given model returns a list of all that goes in to the model.

    Parameters
    ----------
    parameters : dictionary of floats or lists of floats
        Parameters are the options that you are or are likely to change across model instances. When a parameter contains a list, an instance
        of the model will be created for every combination of this parameter with all the other parameters.
    otherOptions : dictionary of float, string or binary valued elements
        These contain all the the model options that describe the model being studied but do not vary across model instances.

    Returns
    -------
    modelDetails : list of dict
        Each dict contains the full set of arguments needed to initialise the model instance
    paramCombDicts : list of dicts
        A list of dictionaries containing the parameter combinations
    """

    if not parameters:
        return [otherOptions]

    paramKeys = parameters.keys()
    paramValues = parameters.values()

    paramCombs = utils.listMergeGen(*paramValues)

    modelDetailList = []
    paramCombDicts = []
    for p in paramCombs:
        paramComb = collections.OrderedDict((k, copy.copy(v)) for k, v in itertools.izip(paramKeys, p))
        d = copy.copy(otherOptions)
        d.update(paramComb)
        paramCombDicts.append(paramComb)
        modelDetailList.append(d)

    return modelDetailList, paramCombDicts
