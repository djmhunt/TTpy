# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

class experimentPlot(object):

    """Abstract class for the creation of plots relevant to a experiment"""

    def __init__(self, expSet, expParams, expLabel, modelSet, modelParams, modelLabels):

        self.expStore = expSet
        self.expParams = expParams
        self.expLabel = expLabel
        self.modelStore = modelSet
        self.modelParams = modelParams
        self.modelLabels = modelLabels

        self._figSets()

    def _figSets(self):
        """ Contains all the figures """

        # Create all the plots and place them in in a list to be iterated

    def __iter__(self):
        """ Returns the iterator for the release of plots"""

        self.counter = 0

        return self

    def next(self):
        """ Produces the next item for the iterator"""

        figure = self.figSets[self.counter]

        self.counter += 1

        return figure