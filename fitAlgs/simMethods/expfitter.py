# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from simMethods.simMethod import simMethod

from itertools import izip
from types import NoneType

class fitter(simMethod):

    """A class for simMethods data by running through an experiment

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
    modelFitVar : string
        The variable to be compared in the model data
    stimuliParams : list of strings or None, optional
        The keys containing the observational parameters seen by the
        participant before taking a decision on an action. Default ``None``
    actChoiceParams : string or None or list of ints, optional
        The name of the key in partData where the list of valid actions
        can be found. If ``None`` then the action list is considered to
        stay constant. If a list then the list will be taken as the list
        of actions that can be taken at each instance. Default ``None``
    fpRespVal : float, optional
        If a floating point error occurs when running a simMethod the simMethod function
        will return a value for each element of fpRespVal.
        Default is 1/1e100
    fitSubset : ``float('Nan')``, ``None`` or list of int, optional
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing ``float('Nan')``, all those trials whose
        feedback was ``float('Nan')``. Default ``None``, which means all trials will be used.
        
    Attributes
    ----------
    Name : string
        The name of the simMethods type
        
    See Also
    --------
    simMethods.simMethod.simMethod : The class this inherits many functions from
    simMethods.fitAlgs.fitAlg.fitAlg : The general simMethods class
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
            The performance metric for the model that will be used to characterise the quality of the simMethod.
            
        See Also
        --------
        simMethods.simMethod.simMethod.participant : Fits participant data
        simMethods.fitAlgs.fitAlg.fitAlg : The general simMethods class
        simMethods.fitAlgs.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        # Run model with given parameters
        exp, model = self.fittedModel(*modelParameters)

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoices = modelData[self.modelFitVar]
        partChoices = self.partChoices

        # Check lengths
        if len(partChoices) != len(modelChoices):
            raise ValueError("The length of the model and participatiant data are different. %s:%s to %s:%s " %
                             (self.partChoiceParam, len(partChoices), self.modelFitVar, len(modelChoices)))

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
