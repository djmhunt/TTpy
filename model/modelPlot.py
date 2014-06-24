# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

class modelPlot(object):

    """Abstract class for the creation of plots relevant to a model"""

    def __init__(self, model, modelParams, modelLabel):

        self.model = model
        self.modelParams = modelParams
        self.modelLabel = modelLabel

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
        """ Produces the next item for the iterator"""

        if self.counter < len(self.figSets):
            figure = self.figSets[self.counter]

            self.counter += 1

            return figure
        else:
            return None