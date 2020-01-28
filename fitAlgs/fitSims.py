# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

import logging
import itertools
import copy
import types

import utils


class FitSim(object):
    """
    A class for fitting data by passing the participant data through the model.

    This has been setup for fitting action-response models

    Parameters
    ----------
    partChoiceParam : string, optional
        The participant data key of their action choices. Default ``'Actions'``
    partRewardParam : string, optional
        The participant data key of the participant reward data. Default ``'Rewards'``
    modelFitVar : string, optional
        The key to be compared in the model data. Default ``'ActionProb'``
    stimuliParams : list of strings or None, optional
        The keys containing the observational parameters seen by the
        participant before taking a decision on an action. Default ``None``
    actChoiceParams : string or None or list of ints, optional
        The name of the key in partData where the list of valid actions
        can be found. If ``None`` then the action list is considered to
        stay constant. If a list then the list will be taken as the list
        of actions that can be taken at each instance. Default ``None``
    fpRespVal : float, optional
        If a floating point error occurs when running a fit the fitter function
        will return a value for each element of fpRespVal. Default is ``1/1e100``
    fitSubset : ``float('Nan')``, ``None``, ``"rewarded"``, ``"unrewarded"``, ``"all"`` or list of int, optional
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing
        - ``"all"`` for fitting all trials
        - ``float('Nan')`` or ``"unrewarded"`` for all those trials whose feedback was ``float('Nan')``
        - ``"rewarded"`` for those who had feedback that was not ``float('Nan')``
        Default ``None``, which means all trials will be used.

    Attributes
    ----------
    Name : string
        The name of the fitting type

    See Also
    --------
    fitAlgs.fitAlg.fitAlg : The general fitting class
    """
## TODO: Change the way in which the fitSubset parameter refers to reward trials with no feedback to be more consistent.

    def __init__(self, partChoiceParam='Actions',
                 partRewardParam='Rewards',
                 modelFitVar='ActionProb',
                 stimuliParams=None,
                 fitSubset=None,
                 actChoiceParams=None,
                 fpRespVal=1/1e100
                 ):

        self.partChoiceParam = partChoiceParam
        self.partRewardParam = partRewardParam
        self.modelFitVar = modelFitVar
        self.partStimuliParams = stimuliParams
        self.partActChoiceParams = actChoiceParams
        self.fpRespVal = fpRespVal
        self.fitSubset = fitSubset

        self.Name = self.findName()

        self.fitInfo = {'Name': self.Name,
                        'participantChoiceParam': partChoiceParam,
                        'participantRewardParam': partRewardParam,
                        'participantStimuliParams': stimuliParams,
                        'participantActChoiceParams': actChoiceParams,
                        'modelFitVar': modelFitVar,
                        'fpRespVal': fpRespVal,
                        'fitSubset': fitSubset}

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
        modelPerformance : list of floats
            The choices made by the model that will be used to characterise the quality of the fit.
            In this case defined as ``[0]``

        See Also
        --------
        fitAlgs.fitSim.fitSim.participant : Fits participant data
        fitAlgs.fitAlg.fitAlg : The general fitting class
        fitAlgs.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        # Run model with given parameters
        try:
            modelInstance = self.fittedModel(*modelParameters)
        except FloatingPointError:
            message = utils.errorResp()
            logger = logging.getLogger('Fitter')
            logger.warning(
                    u"{0}\n. Abandoning fitting with parameters: {1} Returning an action choice probability for each trialstep of {2}".format(message,
                                                                                                                                              repr(
                                                                                                                                                  self.getModParams(
                                                                                                                                                      *modelParameters)),
                                                                                                                                              repr(
                                                                                                                                                  self.fpRespVal)))
            return np.ones(np.array(self.partRewards).shape) * self.fpRespVal
        except ValueError as e:
            logger = logging.getLogger('Fitter')
            logger.warn(
                "{0} in fitted model. Abandoning fitting with parameters: {1}  Returning an action choice probability for each trialstep of {2} - {3}, - {4}".format(
                    type(e),
                    repr(self.getModParams(*modelParameters)),
                    repr(self.fpRespVal),
                    e.message,
                    e.args))
            return np.ones(np.array(self.partRewards).shape) * self.fpRespVal

        # Pull out the values to be compared

        modelData = modelInstance.returnTaskState()
        modelChoices = modelData[self.modelFitVar]

        if self.fitSubsetChosen is not types.NoneType:
            modelPerformance = modelChoices[self.fitSubsetChosen]
        else:
            modelPerformance = modelChoices

        if np.isnan(modelPerformance).any():
            logger = logging.getLogger('Fitter')
            message = "model performance values contain NaN"
            logger.warning(message + ".\n Abandoning fitting with parameters: "
                           + repr(self.getModParams(*modelParameters))
                           + " Returning an action choice probability for each trialstep of "
                           + repr(self.fpRespVal))
            return np.ones(np.array(self.partRewards).shape) * self.fpRespVal

        return modelPerformance

    def getSim(self, model, modelSetup, partData):
        """
        Fit participant data to a model for a given task

        Parameters
        ----------
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
        testedParams : tuple of OrderedDict and list
            They are an ordered dictionary containing the parameter values tested, in the order they were tested, and the
            fit qualities of these parameters.
        """
        fitSubset = self.fitSubset

        self.model = model
        self.mInitialParams = modelSetup[0].values()  # These are passed separately to define at this point the order of the parameters
        self.mParamNames = modelSetup[0].keys()
        self.mOtherParams = modelSetup[1]

        self.partChoices = partData[self.partChoiceParam]

        self.partRewards = partData[self.partRewardParam]

        self.partObs = self.formatPartStim(partData, self.partStimuliParams, self.partActChoiceParams)

        if fitSubset is not None:
            if isinstance(fitSubset, (list, np.ndarray)):
                self.fitSubsetChosen = fitSubset
            elif fitSubset == "rewarded":
                self.fitSubsetChosen = ~np.isnan(self.partRewards)
            elif fitSubset == "unrewarded":
                self.fitSubsetChosen = ~np.isnan(self.partRewards)
            elif np.isnan(fitSubset):
                self.fitSubsetChosen = np.isnan(self.partRewards)
            elif fitSubset == "all":
                self.fitSubsetChosen = None
            else:
                self.fitSubsetChosen = None
        else:
            self.fitSubsetChosen = None

        return self.fitness

    def participantMatchResult(self, model, modelSetup, partData):
        """
        Run the participant data with a model with specified parameters for a given task

        Parameters
        ----------
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
        The dictionary describing the fitters algorithm chosen

        Returns
        -------
        fitInfo : dict
            The dictionary of fitters class information
        """

        return self.fitInfo

    def findName(self):
        """
        Returns the name of the class
        """

        return self.__class__.__name__

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

        partAct = self.partChoices
        partReward = self.partRewards
        partObs = self.partObs

        modelInstance = self._simSetup(partAct, partReward, partObs, *modelParameters)

        return modelInstance

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
            inputs[k] = copy.deepcopy(v)

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

        params = {k: v for k, v in itertools.izip(self.mParamNames, modelParameters)}

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
        validActions : string or None or list of strings, ints or None
            The name of the key in partData where the list of valid actions
            can be found. If ``None`` then the action list is considered to
            stay constant. If a list then the list will be taken as the list
            of actions that can be taken at every trialstep. If the list is
            shorter than the number of trialsteps, then it will be considered
            to be a list of valid actions for each trialstep.

        Returns
        -------
        observation : list of tuples
            The tuples contain the stimuli and the valid actions for each
            observation instance.
        """

        partDataShape = None
        if stimuli is None:
            partDataLen = len(partData[self.partRewardParam])
            stimuliData = [None] * partDataLen
        elif isinstance(stimuli, basestring):
            stimuliData = np.array(partData[stimuli])
            partDataShape = stimuliData.shape
        elif isinstance(stimuli, list):
            if len(stimuli) > 1:
                stimuliData = np.array([partData[s] for s in stimuli]).T
            else:
                stimuliData = np.array(partData[stimuli[0]]).T
            partDataShape = stimuliData.shape
        else:
            raise
        if partDataShape:
            if len(partDataShape) > 1:
                if max(partDataShape) == partDataShape[0]:
                    partDataLen = partDataShape[0]
                else:
                    stimuliData = stimuliData.T
                    partDataLen = partDataShape[1]
            else:
                partDataLen = partDataShape

        if validActions in partData:
            action_data_raw = partData[validActions]
        else:
            action_data_raw = validActions
        partDataLen = len(stimuliData)
        if len(action_data_raw) != partDataLen:
            actionData = [action_data_raw] * partDataLen
        else:
            actionData = action_data_raw

        observation = [(s, a) for s, a in itertools.izip(stimuliData, actionData)]

        return observation

    def _simSetup(self, partAct, partReward, partObs, *modelParameters):
        """
        Initialises the model for the running of the 'simulation'

        Parameters
        ----------
        partAct : list
            The list of actions taken by the participant
        partReward : list
            The feedback received by the participant
        partObs : list
            The observations received by the participant
        *modelParameters : floats
            The model parameters provided in the order defined in the model setup

        Returns
        -------
        modelInstance : model.modelTemplate.modelTemplate class instance
        """

        args = self.getModInput(*modelParameters)

        modelInstance = self.model(**args)

        _simRun(modelInstance, partAct, partReward, partObs)

        return modelInstance


def _simRun(modelInstance, partAct, partReward, partObs):
    """
    Simulates the events of a simulation from the perspective of a model

    Parameters
    ----------
    modelInstance : model.modelTemplate.modelTemplate class instance
    partAct : list
        The list of actions taken by the participant
    partReward : list
        The feedback received by the participant
    partObs : list
        The observations received by the participant
    """

    for action, reward, observation in itertools.izip(partAct, partReward, partObs):
        modelInstance.observe(observation)
        modelInstance.overrideActionChoice(action)
        modelInstance.feedback(reward)
