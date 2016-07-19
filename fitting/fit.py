# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from itertools import izip
from utils import mergeDicts
from numpy import array, concatenate
from types import NoneType
from copy import deepcopy


class fit(object):

    """The abstract class for fitting data
        
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
    stimuliParams : list of strings or None, optional
        The keys containing the observational parameters seen by the
        participant before taking a decision on an action. Default ``None``
    actChoiceParams : string or None or list of ints, optional
        The name of the key in partData where the list of valid actions
        can be found. If ``None`` then the action list is considered to
        stay constant. If a list then the list will be taken as the list
        of actions that can be taken at each instance. Default ``None``
    fpRespVal : float, optional
        If a floating point error occurs when running a fit the fit function
        will return a value for each element of fpRespVal.
        Default is 1/1e100
        
    Attributes
    ----------
    Name : string
        The name of the fitting type
        
    See Also
    --------
    fitting.fitters.fitAlg.fitAlg : The general fitting class
    """

    Name = 'none'

    def __init__(self, partChoiceParam, partRewardParam, modelParam, fitAlg, scalar, **kwargs):

        self.partChoiceParam = partChoiceParam
        self.partRewardParam = partRewardParam
        self.modelparam = modelParam
        self.fitAlg = fitAlg
        self.scalar = scalar
        self.partStimuliParams = kwargs.pop('stimuliParams', None)
        self.partActChoiceParams = kwargs.pop('actChoiceParams', None)
        self.fpRespVal = kwargs.pop('fpRespVal', 1/1e100)

        self.fitInfo = {'Name': self.Name,
                        'participantChoiceParam': partChoiceParam,
                        'participantRewardParam': partRewardParam,
                        'participantStimuliParams': self.partStimuliParams,
                        'participantActChoiceParams': self.partActChoiceParams,
                        'modelParam': modelParam}
        try: 
            self.fitInfo['scalarName'] = self.scalar.Name
            self.fitInfo['scalarEffect'] = self._scalarEffect()
        except AttributeError:
            self.fitInfo['scalarEffect'] = self._scalarEffect()

    def fitness(self, *modelParameters):
        """
        Used by a fitter to generate the list of values characterising how well the model parameters describe the
        participants actions.
        
        Parameters
        ----------
        modelParameters : list of floats
            A list of the parameters used by the model in the order previously
            defined
            
        Returns
        -------
        modelChoices : list of floats
            The choices made by the model that will be used to characterise the quality of the fit.
            In this case defined as ``[0]``
            
        See Also
        --------
        fitting.fit.fit.participant : Fits participant data
        fitting.fitters.fitAlg.fitAlg : The general fitting class
        fitting.fitters.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        return [0]

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
            The first dictionary is the model initial parameters. The second
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
        self.mInitialParams = modelSetup[0].values()  # These are passed seperately to define at this point the order of the parameters
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scalar(partData[self.partChoiceParam])

        self.partRewards = partData[self.partRewardParam]

        self.partObs = self.formatPartStim(partData, self.partStimuliParams, self.partActChoiceParams)

        fitVals, fitQuality = self.fitAlg.fit(self.fitness, self.mParamNames, self.mInitialParams[:])

        model = self.fittedModel(*fitVals)

        return model, fitQuality

    def participantMatchResult(self, exp, model, modelSetup, partData):
        """
        Run the participant data with a model with specified parameters for a given experiment

        Parameters
        ----------
        exp : experiment.experiment.experiment inherited class
            The experiment being fitted. If you are fitting using
            participant responses only it will not be used, so can be anything
        model : model.model.model inherited class
            The model you wish to try and fit values to
        modelSetup : (dict,dict)
            The first dictionary is the model varying parameters. The second
            are the other model parameters
        partData : dict
            The participant data

        Returns
        -------
        model : model.model.model inherited class instance
            The model with the best fit parameters
        """

        self.exp = exp
        self.model = model
        fitVals = modelSetup[0].values()  # These are passed seperately to define at this point the order of the parameters
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = self.scalar(partData[self.partChoiceParam])

        self.partRewards = partData[self.partRewardParam]

        self.partObs = self.formatPartStim(partData, self.partStimuliParams, self.partActChoiceParams)

        model = self.fittedModel(*fitVals)

        return model

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

        return self.fitInfo, fitAlgInfo

    def fittedModel(self, *modelParameters):
        """
        Return the model run of the model with specific parameter values

        Parameters
        ----------
        *modelParameters : floats
            The model parameters provided in the order defined in the model setup

        Returns
        -------
        model : model class instance
        """

        return self.model()

    def getModInput(self, *modelParameters):
        """
        Compiles the kwarg model arguments based on the modelParameters and
        previously specified other parameters
        
        Parameters
        ----------
        modelParameters : list of floats
            The parameter values in the order extracted from the modelSetup
            parameter dictionary
            
        Returns
        -------
        inputs : dict
            The kwarg model arguments
        """

#        inputs = {k : v for k,v in izip(self.mParamNames, modelParameters)}
        inputs = self.getModParams(*modelParameters)

        for k, v in self.mOtherParams.iteritems():
            inputs[k] = deepcopy(v)

        return inputs
        
    def getModParams(self, *modelParameters):
        """
        Compiles the kwarg model parameter arguments based on the 
        modelParameters
        
        Parameters
        ----------
        modelParameters : list of floats
            The parameter values in the order extacted from the modelSetup 
            parameter dictionary
            
        Returns
        -------
        params : dict
            The kwarg model parameter arguments
        """

        params = {k: v for k, v in izip(self.mParamNames, modelParameters)}

        return params
        
    def formatPartStim(self, partData, stimuli, validActions):
        """
        Finds the stimuli in the participant data and returns formatted observations
        
        Parameters
        ----------
        partData : dict
            The participant data
        stimuli : list of strings or ``None``
            A list of the keys in partData representing participant stimuli
        validActions : string or ``None`` or list of ints
            The name of the key in partData where the list of valid actions
            can be found. If ``None`` then the action list is considered to 
            stay constant. If a list then the list will be taken as the list
            of actions that can be taken at every timestep.
        
        Returns
        -------
        observation : list of tuples
            The tuples contain the stimuli and the valid actions for each 
            observation instance.
        """

        if type(stimuli) is NoneType:
            partDataLen = len(partData[self.partRewardParam])
            stimuliData = (None for i in xrange(partDataLen))
        else:
            stimuliData = concatenate([partData[s] for s in stimuli], 1)
            partDataLen = len(stimuliData)

        if type(validActions) is NoneType:
            actionData = (None for i in xrange(partDataLen))
        elif isinstance(validActions, basestring):
            actionData = partData[validActions]
        else:
            actionData = (validActions for i in xrange(partDataLen))

        observation = [(s, a) for a, s in izip(actionData, stimuliData)]
        
        return observation

    def _scalarEffect(self):
        """
        Presents the transformation provided by the scalar
        
        Returns
        -------
        description : string
            The description of the effect of the scalar
        """

        testBed = [0, 1, 2, 3, 10]

        response = self.scalar(array(testBed))

        return repr(testBed) + " --> " + repr(list(response))
