# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from fit import fit

from itertools import izip

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
    scaler : function
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
        Used by a fitter to generate a fit for given model parameters
        
        Parameters
        ----------
        modelParameters : list of floats
            A list of the parameters used by the model in the order previously
            defined
            
        Returns
        -------
        fitQuality : list of floats
            The quality of the fit. In this case defined the differences 
            between the model choices and the participant choices
            
        See Also
        --------
        fitting.fit.fit.participant : Fits participant data
        fitting.fitters.fitAlg.fitAlg : The general fitting class
        """

        #Run model with given parameters
        exp, model = self._simSetup(*modelParameters[0])

        # Pull out the values to be compared

        modelData = model.outputEvolution()
        modelChoices = modelData[self.modelparam]
        partChoices = self.partChoices

        #Check lengths
        if len(partChoices) != len(modelChoices):
            raise ValueError("The length of the model and participatiant data are different. %s:%s to %s:%s " % (self.partChoiceParam,len(partChoices),self.modelparam,len(modelChoices)))

        # Find the difference

        diff = modelChoices - partChoices

        return diff

    def _fittedModel(self,*fitVals):
        """
        Return the best fit model
        """

        exp, model = self._simSetup(*fitVals)

        return model

    def _simSetup(self, *modelParameters):
        """ 
        Initialises the model for the running of the 'simulation'
        """

        args = self.getModInput(*modelParameters)

        model = self.model(**args)
        exp = self.exp.reset()

        self._simRun(exp,model)

        return exp, model

    def _simRun(self, exp, model):
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



