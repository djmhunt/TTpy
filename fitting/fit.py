# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

from itertools import izip, product
from collections import OrderedDict
from numpy import array, concatenate, isnan, ndarray
from types import NoneType
from copy import deepcopy


class fit(object):

    """The abstract class for fitting data

    Parameters
    ----------
    partChoiceParam : string
        The key to be compared in the participant data
    partRewardParam : string
        The variable containing the participant reward data
    modelFitVar : string
        The key to be compared in the model data
    fitAlg : fitting.fitAlgs.fitAlg instance
        An instance of one of the fitting algorithms
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
    fitSubset : ``float('Nan')``, ``None`` or list of int, optional
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing ``float('Nan')``, all those trials whose
        feedback was ``float('Nan')``. Default ``None``, which means all trials will be used.
    calcCov : bool, optional
        Estimating the covariance

    Attributes
    ----------
    Name : string
        The name of the fitting type

    See Also
    --------
    fitting.fitAlgs.fitAlg.fitAlg : The general fitting class
    """

    Name = 'none'

    def __init__(self, partChoiceParam, partRewardParam, modelFitVar, **kwargs):

        self.partChoiceParam = partChoiceParam
        self.partRewardParam = partRewardParam
        self.modelFitVar = modelFitVar
        self.partStimuliParams = kwargs.pop('stimuliParams', None)
        self.partActChoiceParams = kwargs.pop('actChoiceParams', None)
        self.fpRespVal = kwargs.pop('fpRespVal', 1/1e100)
        self.fitSubset = kwargs.pop('fitSubset', None)

        self.fitInfo = {'Name': self.Name,
                        'participantChoiceParam': partChoiceParam,
                        'participantRewardParam': partRewardParam,
                        'participantStimuliParams': self.partStimuliParams,
                        'participantActChoiceParams': self.partActChoiceParams,
                        'modelFitVar': modelFitVar,
                        'fitSubset': self.fitSubset}

        self.fitInfo.update(kwargs.copy())

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
        fitting.fitAlgs.fitAlg.fitAlg : The general fitting class
        fitting.fitAlgs.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        return [0]

    def getSim(self, model, modelSetup, partData, exp=None):
        """
        Fit participant data to a model for a given experiment

        Parameters
        ----------
        model : model.model.model inherited class
            The model you wish to try and fit values to
        modelSetup : (dict,dict)
            The first dictionary is the model initial parameters. The second
            are the other model parameters
        partData : dict
            The participant data
        exp : experiment.experiment.experiment inherited class, optional
            The experiment being fitted. If you are fitting using
            participant responses only it will not be used. Default ``None``

        Returns
        -------
        model : model.model.model inherited class instance
            The model with the best fit parameters
        fitQuality : float
            Specifies the fit quality for this participant to the model
        testedParams : tuple of OrderedDict and list
            They are an ordered dictionary containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.
        """
        fitSubset = self.fitSubset

        if exp is not None:
            self.exp = exp
        else:
            self.exp = None
        self.model = model
        self.mInitialParams = modelSetup[0].values()  # These are passed seperately to define at this point the order of the parameters
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = partData[self.partChoiceParam]

        self.partRewards = partData[self.partRewardParam]

        self.partObs = self.formatPartStim(partData, self.partStimuliParams, self.partActChoiceParams)

        if fitSubset is not None:
            if isinstance(fitSubset, (list, ndarray)):
                self.fitSubsetChosen = fitSubset
            elif isnan(fitSubset):
                self.fitSubsetChosen = isnan(self.partRewards)
            else:
                self.fitSubsetChosen = None
        else:
            self.fitSubsetChosen = None

        return self.fitness

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

        self.partChoices = partData[self.partChoiceParam]

        self.partRewards = partData[self.partRewardParam]

        self.partObs = self.formatPartStim(partData, self.partStimuliParams, self.partActChoiceParams)

        model = self.fittedModel(*fitVals)

        return model

    def info(self):
        """
        The dictionary describing the fitting algorithm chosen

        Returns
        -------
        fitInfo : dict
            The dictionary of fitting class information
        """

        return self.fitInfo

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
            The parameter values in the order extracted from the modelSetup
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
            of actions that can be taken at every trialstep.

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
            if len(stimuli) > 1:
                stimuliData = array([partData[s] for s in stimuli]).T
            else:
                stimuliData = array(partData[stimuli[0]]).T
            partDataLen = stimuliData.shape[0]

        if type(validActions) is NoneType:
            actionData = (None for i in xrange(partDataLen))
        elif isinstance(validActions, basestring):
            actionData = partData[validActions]
        else:
            actionData = (validActions for i in xrange(partDataLen))

        observation = [(s, a) for a, s in izip(actionData, stimuliData)]

        return observation
