# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

from itertools import izip
from utils import mergeDicts
from numpy import array

class fit(object):

    """The abstact class for fitting data
        
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
    fitting.fitters.fitAlg.fitAlg : The general fitting class
    """

    Name = 'none'


    def __init__(self,partChoiceParam, partRewardParam, modelParam, fitAlg, scaler):

        self.partChoiceParam = partChoiceParam
        self.partRewardParam = partRewardParam
        self.modelparam = modelParam
        self.fitAlg = fitAlg
        self.scaler = scaler

        self.fitInfo = {'Name':self.Name,
                        'participantChoiceParam':partChoiceParam,
                        'participantRewardParam':partRewardParam,
                        'modelParam':modelParam}
        try: 
            self.fitInfo['scalerName'] = self.scaler.Name
            self.fitInfo['scalerEffect'] = self._scalerEffect()
        except AttributeError:
            self.fitInfo['scalerEffect'] = self._scalerEffect()

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
        fitQuality : float
            The quality of the fit. In this case defined as always zero
            
        See Also
        --------
        fitting.fit.fit.participant : Fits participant data
        fitting.fitters.fitAlg.fitAlg : The general fitting class
        """

        return 0

    def participant(self, exp, model, modelSetup, partData):
        """
        Fit participant data to a model for a given experiment
        
        Parameters
        ----------
        exp : experiment.experiment.experiment inherited class
            The experiment being fitted. If you are fitting using 
            participant responses only it will not be used, so can be anything
        model : model.model.model inherited class
            The model you wish to try and fit values to
        modelSetup : (dict,dict)
            The first dictionary is the model inital parameters. The second 
            are the other model parameters
        partData : dict
            The participant data
        
        Returns
        -------
        model : model.model.model inherited class instance
            The model with the best fit parameters
        fitQuality : float
            Specifies the fit quality for this participant to the model
        """

        self.exp = exp
        self.model = model
        self.mInitialParams = modelSetup[0].values()
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scaler(partData[self.partChoiceParam])
        
        self.partRewards = partData[self.partRewardParam]

        fitVals, fitQuality = self.fitAlg.fit(self.fitness, self.mInitialParams[:])

        model = self._fittedModel(*fitVals)

        return model, fitQuality

    def info(self):
        """
        The information relating to the fitting method used
        
        Includes information on the fitting algorithm used
        
        Returns
        -------
        info : (dict,dict)
            The fitting info and the fitting.fitters info
            
        See Also
        --------
        fitting.fitters.fitAlg.fitAlg.info
        """

        fitAlgInfo = self.fitAlg.info()

#        labeledFitAlgInfo = {"fitAlg_"+k:v for k,v in fitAlgInfo.iteritems()}

#        labeledFitInfo = {"fit_" + k : v for k,v in self.fitInfo.iteritems()}

#        fitInfo = mergeDicts(labeledFitAlgInfo, labeledFitInfo)

        return (self.fitInfo,fitAlgInfo)

    def _fittedModel(self,*fitVals):
        """
        Return the best fit model
        """

        model = self._simSetup(*fitVals)

        return model

    def getModInput(self, *modelParameters):
        """
        Compiles the kwarg model arguments based on the modelParameters and
        previously specified other parameters
        
        Parameters
        ----------
        modelParameters : list of floats
            The parameter values in the order extacted from the modelSetup 
            parameter dictionary
            
        Returns
        -------
        inputs : dict
            The kwarg model arguments
        """

        optional = self.mOtherParams

        inputs = {k : v for k,v in izip(self.mParamNames, modelParameters)}

        for k, v in optional.iteritems():
            inputs[k] = v

        return inputs

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

        pass

    def _scalerEffect(self):
        """
        Presents the transformation provided by the scaler
        
        Returns
        -------
        description : string
            The description of the effect of the scaler
        """

        testBed = [0,1,2,3,10]

        response = self.scaler(array(testBed))

        return repr(testBed) + " --> " + repr(list(response))



