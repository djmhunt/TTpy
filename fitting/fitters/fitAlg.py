# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

class fitAlg(object):

    """The abstact class for fitting data

    fitAlg()

    """


    def __init__(self,dataShaper = None):

        self.fitness = self.null

    def null(self,*params):

        modVals = self.sim(*params)

        return modVals

    def fit(self, sim, mInitialParams):

        self.sim = sim

        return 0

