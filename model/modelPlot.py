# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

class modelPlot(object):

    """Abstract class for the creation of plots relevant to a model"""

    def __init__(self, model, modelParams, modelLabel):

        # Create all the plots and place them in in a list to be iterated


    def __iter__(self):
        """ Returns the iterator for the release of plots"""

        return self

    def next(self):
        """ Produces the next item for the iterator"""