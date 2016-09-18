# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from fit import fit

from itertools import izip
from types import NoneType

class fitter(fit):

    """A class for fitting data by running through an experiment

    To be fixed later
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
        
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
    scalar : function
        Transforms the participant action form to match that of the model
        
    Attributes
    ----------
    Name : string
        The name of the fitting type
        
    See Also
    --------
    fitting.fit.fit : The class this inherits many functions from
    fitting.fitters.fitAlg.fitAlg : The general fitting class
    """
    
    Name = "experimentFitter"

    def fitness(self, *modelParameters):
        """
        Used by a fitter to generate the list of values characterising how well given model parameters perform
        
        Parameters
        ----------
        modelParameters : list of floats
            A list of the parameters used by the model in the order previously defined
            
        Returns
        -------
        modelPerformance : list of floats
            The performance metric for the model that will be used to characterise the quality of the fit.
            
        See Also
        --------
        fitting.fit.fit.participant : Fits participant data
        fitting.fitters.fitAlg.fitAlg : The general fitting class
        fitting.fitters.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        # Run model with given parameters
        exp, model = self.fittedModel(*modelParameters)

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoices = modelData[self.modelparam]
        partChoices = self.partChoices

        # Check lengths
        if len(partChoices) != len(modelChoices):
            raise ValueError("The length of the model and participatiant data are different. %s:%s to %s:%s " %
                             (self.partChoiceParam, len(partChoices), self.modelparam, len(modelChoices)))

        # Find the difference

        diff = modelChoices - partChoices

        if self.fitSubsetChosen is not NoneType:
            modelPerformance = diff[self.fitSubsetChosen]
        else:
            modelPerformance = diff

        return modelPerformance

    def fittedModel(self, *modelParameters):
        """
        Return the model run of the model with specific parameter values

        Parameters
        ----------
        *modelParameters : floats
            The model parameters provided in the order defined in the model setup

        Returns
        -------
        modelInstance : model class instance
        """

        exp, modelInstance = self._simExpSetup(*modelParameters)

        return exp, modelInstance

    def _simExpSetup(self, *modelParameters):
        """
        Initialises the model for the running of the 'simulation'

        Parameters
        ----------
        *modelParameters : floats
            The model parameters provided in the order defined in the model setup

        Returns
        -------
        exp : experiment.experimentTemplate.experimentTemplate class instance
        modelInstance : model.modelTemplate.modelTemplate class instance
        """

        args = self.getModInput(*modelParameters)

        modelInstance = self.model(**args)
        exp = self.exp.reset()

        self._simExpRun(exp, modelInstance)

        return exp, modelInstance

    def _simExpRun(self, exp, model):
        """
        Simulates the events of a simulation from the perspective of a model
        """

        for state in exp:
            model.observe(state)
            act = model.action()
            exp.receiveAction(act)
            response = exp.feedback()
            model.feedback(response)
            exp.procede()
