# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
from __future__ import division

class fitAlg(object):

    """The abstact class for fitting data

    fitAlg()

    """

    Name = 'none'


    def __init__(self,dataShaper = None):

        self.fitness = self.null

        self.fitInfo = {'Name':self.Name}

    def null(self,*params):

        modVals = self.sim(*params)

        return modVals

    def fit(self, sim, mInitialParams):

        self.sim = sim

        return 0

    def info(self):

        return self.fitInfo

