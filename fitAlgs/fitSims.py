# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import numpy as np

import logging
import itertools
import copy
import types

import utils


class FitSubsetError(Exception):
    pass


class ActionError(Exception):
    pass


class StimuliError(Exception):
    pass


class FitSim(object):
    """
    A class for fitting data by passing the participant data through the model.

    This has been setup for fitting action-response models

    Parameters
    ----------
    participant_choice_property : string, optional
        The participant data key of their action choices. Default ``'Actions'``
    participant_reward_property : string, optional
        The participant data key of the participant reward data. Default ``'Rewards'``
    model_fitting_variable : string, optional
        The key to be compared in the model data. Default ``'ActionProb'``
    task_stimuli_property : list of strings or None, optional
        The keys containing the stimuli seen by the
        participant before taking a decision on an action. Default ``None``
    action_options_property : string or None or list of ints, optional
        The name of the key in partData where the list of valid actions
        can be found. If ``None`` then the action list is considered to
        stay constant. If a list then the list will be taken as the list
        of actions that can be taken at each instance. Default ``None``
    float_error_response_value : float, optional
        If a floating point error occurs when running a fit the fitter function
        will return a value for each element of fpRespVal. Default is ``1/1e100``
    fit_subset : ``float('Nan')``, ``None``, ``"rewarded"``, ``"unrewarded"``, ``"all"`` or list of int, optional
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
    fitAlgs.fitAlg.FitAlg : The general fitting class
    """

    def __init__(self,
                 participant_choice_property='Actions',
                 participant_reward_property='Rewards',
                 model_fitting_variable='ActionProb',
                 task_stimuli_property=None,
                 fit_subset=None,
                 action_options_property=None,
                 float_error_response_value=1 / 1e100
                 ):

        self.participant_choice_property = participant_choice_property
        self.participant_reward_property = participant_reward_property
        self.model_fitting_variable = model_fitting_variable
        self.task_stimuli_property = task_stimuli_property
        self.action_options_property = action_options_property
        self.float_error_response_value = float_error_response_value
        self.fit_subset = fit_subset
        self.fit_subset_described = self._preprocess_fit_subset(fit_subset)

        self.Name = self.find_name()

        self.sim_info = {'Name': self.Name,
                         'participant_choice_property': participant_choice_property,
                         'participant_reward_property': participant_reward_property,
                         'task_stimuli_property': task_stimuli_property,
                         'action_options_property': action_options_property,
                         'model_fitting_variable': model_fitting_variable,
                         'float_error_response_value': float_error_response_value,
                         'fit_subset': fit_subset}

        self.model = None
        self.initial_parameter_values = None
        self.model_parameter_names = None
        self.model_other_properties = None

        self.participant_observations = None
        self.participant_actions = None
        self.participant_rewards = None

    def fitness(self, *model_parameters):
        """
        Used by a fitter to generate the list of values characterising how well the model parameters describe the
        participants actions.

        Parameters
        ----------
        model_parameters : list of floats
            A list of the parameters used by the model in the order previously defined

        Returns
        -------
        model_performance : list of floats
            The choices made by the model that will be used to characterise the quality of the fit.

        See Also
        --------
        fitAlgs.fitSims.FitSim.participant : Fits participant data
        fitAlgs.fitAlg.fitAlg : The general fitting class
        fitAlgs.fitAlg.fitAlg.fitness : The function that this one is called by
        """

        try:
            model_instance = self.fitted_model(*model_parameters)
        except FloatingPointError:
            message = utils.errorResp()
            logger = logging.getLogger('Fitter')
            logger.warning(
                    u"{0}\n. Abandoning fitting with parameters: {1} Returning an action choice probability for each trialstep of {2}".format(message,
                                                                                                                                              repr(
                                                                                                                                                  self.get_model_parameters(
                                                                                                                                                      *model_parameters)),
                                                                                                                                              repr(
                                                                                                                                                  self.float_error_response_value)))
            return np.ones(np.array(self.participant_rewards).shape) * self.float_error_response_value
        except ValueError as e:
            logger = logging.getLogger('Fitter')
            logger.warn(
                "{0} in fitted model. Abandoning fitting with parameters: {1}  Returning an action choice probability for each trialstep of {2} - {3}, - {4}".format(
                    type(e),
                    repr(self.get_model_parameters(*model_parameters)),
                    repr(self.float_error_response_value),
                    e.message,
                    e.args))
            return np.ones(np.array(self.participant_rewards).shape) * self.float_error_response_value

        # Pull out the values to be compared
        model_data = model_instance.return_task_state()
        model_choice_probabilities = model_data[self.model_fitting_variable]

        if self.fit_subset_described is None:
            model_performance = model_choice_probabilities
        else:
            model_performance = model_choice_probabilities[self.fit_subset_described]

        if np.isnan(model_performance).any():
            logger = logging.getLogger('Fitter')
            message = "model performance values contain ``Not a Number`` (NaN), i.e. the model had a problem."
            logger.warning(message + ".\n Abandoning fitting with parameters: "
                           + repr(self.get_model_parameters(*model_parameters))
                           + " Returning an action choice probability for each trialstep of "
                           + repr(self.float_error_response_value))
            return np.ones(np.array(self.participant_rewards).shape) * self.float_error_response_value

        return model_performance

    def prepare_sim(self, model, model_parameters, model_properties, participant_data):
        """
        Set up the simulation of a model following the behaviour of a participant

        Parameters
        ----------
        model : model.modelTemplate.Model inherited class
            The model you wish to try and fit values to
        model_parameters : dict
            The model initial parameters
        model_properties : dict
            The model static properties
        participant_data : dict
            The participant data

        Returns
        -------
        fitness
        """

        self.model = model
        self.initial_parameter_values = list(model_parameters.values())
        self.model_parameter_names = list(model_parameters.keys())
        self.model_other_properties = model_properties

        participant_sequence = self.participant_sequence_generation(participant_data,
                                                                    self.participant_choice_property,
                                                                    self.participant_reward_property,
                                                                    self.task_stimuli_property,
                                                                    self.action_options_property)

        self.participant_observations, self.participant_actions, self.participant_rewards = participant_sequence

        if not self.fit_subset_described and self.fit_subset_described is not None:
            self.fit_subset_described = self._set_fit_subset(self.fit_subset, self.participant_rewards)

        return self.fitness

    @staticmethod
    def participant_sequence_generation(participant_data,
                                        choice_property,
                                        reward_property,
                                        stimuli_property,
                                        action_options_property):
        """
        Finds the stimuli in the participant data and returns formatted observations

        Parameters
        ----------
        participant_data : dict
            The participant data
        choice_property : string
            The participant data key of their action choices.
        reward_property : string
            The participant data key of the participant reward data
        stimuli_property : string or None or list of strings
            A list of the keys in partData representing participant stimuli
        action_options_property : string or None or list of strings, ints or None
            The name of the key in partData where the list of valid actions
            can be found. If ``None`` then the action list is considered to
            stay constant. If a list then the list will be taken as the list
            of actions that can be taken at every trialstep. If the list is
            shorter than the number of trialsteps, then it will be considered
            to be a list of valid actions for each trialstep.

        Returns
        -------
        participant_sequence : list of three element tuples
            Each list element contains the observation, action and feedback for each trial taken
            by the participant
        """

        actions = participant_data[choice_property]
        rewards = participant_data[reward_property]

        participant_data_length = len(actions)

        partDataShape = None
        if stimuli_property is None:
            stimuli_data = [None] * participant_data_length
        elif isinstance(stimuli_property, str):
            stimuli_data = np.array(participant_data[stimuli_property])
            partDataShape = stimuli_data.shape
        elif isinstance(stimuli_property, list):
            if len(stimuli_property) > 1:
                stimuli_data = np.array([participant_data[s] for s in stimuli_property]).T
            else:
                stimuli_data = participant_data[stimuli_property[0]]
            partDataShape = stimuli_data.shape
        else:
            raise StimuliError('Unknown representation of stimuli')

        if partDataShape:
            if max(partDataShape) != partDataShape[0]:
                stimuli_data = stimuli_data.T

        if isinstance(action_options_property, str) and action_options_property in participant_data:
            available_actions = participant_data[action_options_property]
        elif action_options_property is None or len(action_options_property) != participant_data_length:
            available_actions = [action_options_property] * participant_data_length
        else:
            available_actions = action_options_property

        mismatches = [True if (trial_available_actions is not None and trial_action not in trial_available_actions)
                           else False
                      for trial_action, trial_available_actions in zip(actions, available_actions)]

        if any(mismatches):
            mismatch_actions = [a for a, m in zip(actions, mismatches) if m is True]
            mismatch_available_actions = [a for a, m in zip(available_actions, mismatches) if m is True]
            raise ActionError('An action is chosen that is not listed as available for the trial \n{}\n {}'.format(mismatch_actions,
                                                                                                               mismatch_available_actions))

        observations = [(s, a) for s, a in zip(stimuli_data, available_actions)]

        return observations, actions, rewards

    def info(self):
        """
        The dictionary describing the fitters algorithm chosen

        Returns
        -------
        fitInfo : dict
            The dictionary of fitters class information
        """

        return self.sim_info

    def find_name(self):
        """
        Returns the name of the class
        """

        return self.__class__.__name__

    def fitted_model(self, *model_parameters):
        """
        Simulating a model run with specific parameter values

        Parameters
        ----------
        *model_parameters : floats
            The model parameters provided in the order defined in the model setup

        Returns
        -------
        model_instance : model.modelTemplate.Model class instance
        """

        model_arguments = self.get_model_properties(*model_parameters)

        model_instance = self.model(**model_arguments)

        model_instance = self._simulation_run(model_instance,
                                              self.participant_observations,
                                              self.participant_actions,
                                              self.participant_rewards)

        return model_instance

    def get_model_properties(self, *model_parameters):
        """
        Compiles the kwarg model arguments based on the model_parameters and
        previously specified other parameters

        Parameters
        ----------
        model_parameters : list of floats
            The parameter values in the order extracted from the modelSetup parameter dictionary

        Returns
        -------
        model_properties : dict
            The kwarg model arguments
        """

        model_properties = self.get_model_parameters(*model_parameters)

        for k, v in self.model_other_properties.items():
            model_properties[k] = copy.deepcopy(v)

        return model_properties

    def get_model_parameters(self, *model_parameters):
        """
        Compiles the model parameter arguments based on the model parameters

        Parameters
        ----------
        model_parameters : list of floats
            The parameter values in the order extracted from the modelSetup parameter dictionary

        Returns
        -------
        parameters : dict
            The kwarg model parameter arguments
        """

        parameters = {k: v for k, v in zip(self.model_parameter_names, model_parameters)}

        return parameters

    @staticmethod
    def _simulation_run(model_instance, observations, actions, rewards):
        """
        Simulates the events of a simulation from the perspective of a model

        Parameters
        ----------
        model_instance : model.modelTemplate.modelTemplate class instance
        observations : list of tuples
            The sequence of (stimuli, valid actions) for each trial
        actions : list
            The sequence of participant actions for each trial
        rewards : list
            The sequence of participant rewards for each trial
        model_instance : model.modelTemplate.Model class instance
            The same instance that is returned

        Returns
        -------
        model_instance : model.modelTemplate.Model class instance
            The same instance that was passed in
        """

        for observation, action, reward in zip(observations, actions, rewards):
            model_instance.observe(observation)
            model_instance.override_action_choice(action)
            model_instance.feedback(reward)

        return model_instance

    @staticmethod
    def _preprocess_fit_subset(fit_subset):
        """
        Prepare as many possible combinations of fit_subset as possible.
        If it needs knowledge of the rewards, return ``[]``

        Parameters
        ----------
        fit_subset : ``float('Nan')``, ``None``, ``"rewarded"``, ``"unrewarded"``, ``"all"`` or list of int
        Describes which, if any, subset of trials will be used to evaluate the performance of the model.
        This can either be described as a list of trial numbers or, by passing
        - ``"all"`` or ``None`` for fitting all trials
        - ``float('Nan')`` or ``"unrewarded"`` for all those trials whose feedback was ``float('Nan')``
        - ``"rewarded"`` for those who had feedback that was not ``float('Nan')``

        Returns
        -------
        fit_subset_described : None, or list of ints
            A description of the trials to be used, with ``None`` being all of them.
            If more information was needed ``[]`` was returned
        """

        if fit_subset is None:
            fit_subset_described = None
        elif isinstance(fit_subset, (list, np.ndarray)):
            fit_subset_described = fit_subset
        elif fit_subset == "rewarded":
            fit_subset_described = []
        elif fit_subset == "unrewarded":
            fit_subset_described = []
        elif fit_subset == "all":
            fit_subset_described = None
        elif isinstance(fit_subset, float) and np.isnan(fit_subset):
            fit_subset_described = []
        else:
            raise FitSubsetError('{} is not a known fit_subset'.format(fit_subset))

        return fit_subset_described

    @staticmethod
    def _set_fit_subset(fit_subset, part_rewards):
        """
        Identify any fit_subset options that required part_rewards, i.e. subsets of trials where there was or was not
        ``np.nan`` as the feedback.

        Parameters
        ----------
        fit_subset : ``float('Nan')``, ``"rewarded"``, ``"unrewarded"``
        Describes which, subset of trials will be used to evaluate the performance of the model.
        This can either be described by passing
        - ``float('Nan')`` or ``"unrewarded"`` for all those trials whose feedback was ``float('Nan')``
        - ``"rewarded"`` for those who had feedback that was not ``float('Nan')``
        part_rewards: list of float
            The rewards received by the participant

        Returns
        -------
        fit_subset_described : list of bool the length of part_reward
            A description of the trials to be used
        """

        if fit_subset == "rewarded":
            fit_subset_described = ~np.isnan(part_rewards)
        elif fit_subset == "unrewarded":
            fit_subset_described = np.isnan(part_rewards)
        elif isinstance(fit_subset, float) and np.isnan(fit_subset):
            fit_subset_described = np.isnan(part_rewards)
        else:
            raise FitSubsetError('{} is not a known fit_subset'.format(fit_subset))

        return fit_subset_described
