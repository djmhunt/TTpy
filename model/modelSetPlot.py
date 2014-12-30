# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

class modelSetPlot(object):

    """Abstract class for the creation of plots for sets of results from a model
    
    Parameters
    ----------
    modelSet : list of dictionaries
        A list of sets of data produced by each model
    modelParams : list with dictionaries
        List of parameters associated with each model
    modelLabels : list with strings
        A list of unique identifiers for each model"""

    def __init__(self, modelSet, modelParams, modelLabels):

        self.modelStore = modelSet
        self.modelParams = modelParams
        self.modelLabels = modelLabels

        self._figSets()

    def _figSets(self):
        """ Contains all the figures """

        self.figSets = []

        # Create all the plots and place them in in a list to be iterated

    def __iter__(self):
        """ Returns the iterator for the release of plots"""

        self.counter = 0

        return self

    def next(self):
        """ 
        Returns the plots
        
        Returns
        -------
        figure : matplotlib.pyplot.figure
        
        Raises
        ------
        StopIteration
            When there are no more figures to return
        """

        if self.counter < len(self.figSets):
            figure = self.figSets[self.counter]

            self.counter += 1

            return figure
        else:
            raise StopIteration
