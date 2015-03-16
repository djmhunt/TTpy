# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import logging

from fit import fit

from itertools import izip
from numpy import log, concatenate, array
from numpy import sum as asum
#
#from utils import listMerGen

class fitter(fit):

    """
    A class for fitting data by passing the participant data through the model.
    
    Used only for fitting action-response models

    Parameters
    ----------
    partChoiceParam : string
        The key to be compared in the participant data
    partRewardParam : string
        The key containing the participant reward data
    modelParam : string
        The key to be compared in the model data
    fitAlg : fitting.fitters.fitAlg instance
        An instance of one of the fitting algorithms
    scaler : function
        Transforms the participant action form to match that of the model
        
    Attributes
    ----------
    Name : string
        The Name of the fitting type
        
    See Also
    --------
    fitting.fit.fit : The class this inherits many functions from
    fitting.fitters.fitAlg.fitAlg : The general fitting class
    """
    
    Name = "fitness"

    def fitness(self, *modelParameters):
        """
        Used by a fitter to calculate the quality of a fit for given model 
        parameters
        
        Parameters
        ----------
        modelParameters : list of floats
            A list of the parameters used by the model in the order previously
            defined
            
        Returns
        -------
        fitQuality : list of floats
            The quality of the fit. In this case defined as the choices
            made by the model
            
        See Also
        --------
        fitting.fit.fit.participant : Fits participant data
        fitting.fitters.fitAlg.fitAlg : The general fitting class
        """

        #Run model with given parameters
        model = self._simSetup(*modelParameters[0])

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoices = modelData[self.modelparam]

        return modelChoices

    def _fittedModel(self,*fitVals):
        """
        Return the best fit model
        """

        model = self._simSetup(*fitVals)

        return model

    def _simSetup(self, *modelParameters):
        """ 
        Initialises the model for the running of the 'simulation'
        """

        args = self.getModInput(*modelParameters)

        model = self.model(**args)

        self._simRun(model)

        return model

    def _simRun(self, model):
        """
        Simulates the events of a simulation from the perspective of a model
        """

        parAct = self.partChoices
        parReward = self.partRewards

        for action, reward in izip(parAct, parReward):

            model.currAction = action
            model.storeState()
            model.feedback(reward)
            




