# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging

from fitting.fit import fit

from itertools import izip
from numpy import array, ones, isnan
from utils import errorResp
from types import NoneType
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
    fitSubset : ``float('Nan')``, ``None`` or list of int, optional
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing ``float('Nan')``, all those trials whose
        feedback was ``float('Nan')``. Default ``None``, which means all trials will be used.

    Attributes
    ----------
    Name : string
        The Name of the fitting type

    See Also
    --------
    fitting.fit.fit : The class this inherits many functions from
    fitting.fitters.fitAlg.fitAlg : The general fitting class
    """

    Name = "actReactFitter"

    def fitness(self, *modelParameters):
        """
        Used by a fitter to generate the list of values characterising how well the model parameters describe the
        participants actions.

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
        try:
            modelInstance = self.fittedModel(*modelParameters)
        except FloatingPointError:
            message = errorResp()
            logger = logging.getLogger('Fitter')
            logger.warning(message + "\n. Abandoning fitting with parameters: "
                                   + repr(self.getModParams(*modelParameters))
                                   + " Returning fit value " + repr(self.fpRespVal))
            return ones(array(self.partRewards).shape)*self.fpRespVal

        # Pull out the values to be compared

        modelData = modelInstance.outputEvolution()
        modelChoices = modelData[self.modelparam]

        if self.fitSubsetChosen is not NoneType:
            modelPerformance = modelChoices[self.fitSubsetChosen]
        else:
            modelPerformance = modelChoices

        if isnan(modelPerformance).any():
            logger = logging.getLogger('Fitter')
            message = "model performance values contain NaN"
            logger.warning(message + ".\n Abandoning fitting with parameters: "
                                   + repr(self.getModParams(*modelParameters))
                                   + " Returning fit value " + repr(self.fpRespVal))
            return ones(array(self.partRewards).shape)*self.fpRespVal


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
        model : model class instance
        """

        partAct = self.partChoices
        partReward = self.partRewards
        partObs = self.partObs

        modelInstance = self._simSetup(partAct, partReward, partObs, *modelParameters)

        return modelInstance

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

        self._simRun(modelInstance, partAct, partReward, partObs)

        return modelInstance

    def _simRun(self, modelInstance, partAct, partReward, partObs):
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

        for action, reward, observation in izip(partAct, partReward, partObs):

            modelInstance.observe(observation)
            modelInstance.overrideActionChoice(action)
            modelInstance.feedback(reward)
