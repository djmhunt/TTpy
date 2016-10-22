# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

class experimentPlot(object):
    """
    Abstract class for the creation of plots relevant to a experiment
    
    Parameters
    ----------
    expSet : list of dicts
        The data from each experiment run
    expParams : list of dicts
        The input parameters of each experiment run
    expLabels : list of strings
        The labels for each experiment run
    modelSet : list dicts
        The data from each model run
    modelParams : list of dicts
        The input parameters of each model run
    modelLabels : list of strings
        The labels for each model run
    plotArgs : dict
        The arguments for the plotting functions
    """

    def __init__(self, expSet, expParams, expLabels, modelSet, modelParams, modelLabels, plotArgs):

        self.expStore = expSet
        self.expParams = expParams
        self.expLabels = expLabels
        self.modelStore = modelSet
        self.modelParams = modelParams
        self.modelLabels = modelLabels
        self.plotArgs = plotArgs

        self._figSets()

    def _figSets(self):
        """
        Sets up a container for all the figures
        """

        self.figSets = []

        # Create all the plots and place them in in a list to be iterated

    def __iter__(self):
        """
        Returns the iterator for the release of plots
        """

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