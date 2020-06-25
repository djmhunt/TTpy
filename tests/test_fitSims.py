# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import pytest

import numpy as np

from modelGenerator import ModelGen

import fitAlgs.fitSims as fitSims


@pytest.fixture(scope="function")
def model_setup():
    number_actions = 2
    number_cues = 1

    model_parameters = {'alpha': (0, 1),
                        'beta': (0, 30)}
    model_static_args = {'number_actions': number_actions,
                         'number_cues': number_cues,
                         'action_codes': {1: 0, 2: 1},
                         'expect': np.full(number_actions, 0.5, float),
                         'prior': np.full(number_actions, 1 / number_actions, float),
                         'stimulus_shaper_name': 'StimulusDecksLinear',
                         'reward_shaper_name': 'RewardDecksNormalised',
                         'decision_function_name': 'weightProb',
                         'task_responses': [1, 2]}

    models = ModelGen(model_name='QLearn',
                      parameters=model_parameters,
                      other_options=model_static_args)

    modelInfos = [m for m in models]
    modelInfo = modelInfos[0]
    model = modelInfo[0]
    modelSetup = modelInfo[1:]

    return model, modelSetup


@pytest.fixture(scope='session')
def participant_data_setup():

    participant_data = {'Rewards': [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        'Stimuli': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        'non_action': u'None',
                        'valid_actions_combined': [['E', 'F'], ['E', 'F'], ['A', 'B'], ['E', 'F'], ['E', 'F'],
                                                   ['E', 'F'], ['D', 'B'], ['E', 'D'], ['C', 'B'], ['E', 'C'],
                                                   ['A', 'D'], ['A', 'F']],
                        'simID': '0',
                        'Decisions': ['E', 'F', 'B', 'E', 'E', 'F', 'D', 'E', 'C', 'C', 'A', 'A'],
                        'Choices': [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2]}

    return participant_data


class TestClass_fitSims:

    def test_FS_basic(self):
        fit_sim = fitSims.FitSim()
        assert isinstance(fit_sim, fitSims.FitSim)

    def test_FS_info(self):
        fit_sim = fitSims.FitSim()
        result = fit_sim.info()
        correct_result = {'Name': 'FitSim',
                          'participant_choice_property': 'Actions',
                          'participant_reward_property': 'Rewards',
                          'task_stimuli_property': None,
                          'action_options_property': None,
                          'model_fitting_variable': 'ActionProb',
                          'float_error_response_value': 1 / 1e100,
                          'fit_subset': None}
        for k in result.keys():
            assert result[k] == correct_result[k]

    def test_FS_subset(self):
        fit_sim = fitSims.FitSim()
        subset_values = [np.nan, 'all', 'unrewarded', 'rewarded', [1, 2, 3], None]
        subset_returns = [[], None, [], [], [1, 2, 3], None]
        for values, returns in zip(subset_values, subset_returns):
            results = fit_sim._preprocess_fit_subset(values)
            assert results == returns

    def test_FS_subset2(self):
        fit_sim = fitSims.FitSim()
        subset_values = ['boo', 1, {}]
        for values in subset_values:
            with pytest.raises(fitSims.FitSubsetError, match='{} is not a known fit_subset'.format(values)):
                assert fit_sim._preprocess_fit_subset(values)

    def test_FS_subset3(self):
        fit_sim = fitSims.FitSim()
        part_rewards = [1, 2, np.nan, np.nan, 5]
        subset_values = [np.nan, 'unrewarded', 'rewarded']
        subset_returns = [[False, False, True, True, False],
                          [False, False, True, True, False],
                          [True, True, False, False, True]]
        for values, returns in zip(subset_values, subset_returns):
            results = fit_sim._set_fit_subset(values, part_rewards)
            assert all(results == returns)

    def test_FS_subset4(self):
        fit_sim = fitSims.FitSim()
        part_rewards = [1, 2, np.nan, np.nan, 5]
        subset_values = ['all', None]
        for values in subset_values:
            with pytest.raises(fitSims.FitSubsetError, match='{} is not a known fit_subset'.format(values)):
                assert fit_sim._set_fit_subset(values, part_rewards)

class TestClass_participant:

    def test_participant_processing(self, participant_data_setup):
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim()
        result = fit_sim.participant_sequence_generation(participant_data,
                                                         'Decisions',
                                                         'Rewards',
                                                         None,
                                                         'valid_actions_combined')
        correct_result = [[(None, ['E', 'F']), (None, ['E', 'F']), (None, ['A', 'B']), (None, ['E', 'F']),
                           (None, ['E', 'F']), (None, ['E', 'F']), (None, ['D', 'B']), (None, ['E', 'D']),
                           (None, ['C', 'B']), (None, ['E', 'C']), (None, ['A', 'D']), (None, ['A', 'F'])],
                          ['E', 'F', 'B', 'E', 'E', 'F', 'D', 'E', 'C', 'C', 'A', 'A'],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        for row, correct_row in zip(result, correct_result):
            assert row == correct_row

    def test_participant_processing2(self, participant_data_setup):
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim()
        result = fit_sim.participant_sequence_generation(participant_data,
                                                         'Decisions',
                                                         'Rewards',
                                                         None,
                                                         None)
        correct_result = [[(None, None), (None, None), (None, None), (None, None), (None, None), (None, None),
                           (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)],
                          ['E', 'F', 'B', 'E', 'E', 'F', 'D', 'E', 'C', 'C', 'A', 'A'],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        for row, correct_row in zip(result, correct_result):
            assert row == correct_row

    def test_participant_processing3(self, participant_data_setup):
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim()
        result = fit_sim.participant_sequence_generation(participant_data,
                                                         'Decisions',
                                                         'Rewards',
                                                         None,
                                                         ['A', 'B', 'C', 'D', 'E', 'F'])
        correct_result = [[(None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F']),
                           (None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F']),
                           (None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F']),
                           (None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F']),
                           (None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F']),
                           (None, ['A', 'B', 'C', 'D', 'E', 'F']), (None, ['A', 'B', 'C', 'D', 'E', 'F'])],
                          ['E', 'F', 'B', 'E', 'E', 'F', 'D', 'E', 'C', 'C', 'A', 'A'],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        for row, correct_row in zip(result, correct_result):
            assert row == correct_row

    def test_participant_processing4(self, participant_data_setup):
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim()
        with pytest.raises(fitSims.ActionError):
            fit_sim.participant_sequence_generation(participant_data,
                                                             'Decisions',
                                                             'Rewards',
                                                             None,
                                                             ['A', 'C', 'D', 'E', 'F'])


class TestClass_model:

    def test_model_parameters(self, model_setup):
        model, (model_parameters, model_properties) = model_setup

        fit_sim = fitSims.FitSim()
        fit_sim.model_parameter_names = list(model_parameters.keys())
        results = fit_sim.get_model_parameters(*list(model_parameters.values()))
        correct_results = model_parameters
        for k in results:
            assert results[k] == correct_results[k]

    def test_model_properties(self, model_setup):
        model, (model_parameters, model_properties) = model_setup

        fit_sim = fitSims.FitSim()
        fit_sim.model_parameter_names = list(model_parameters.keys())
        fit_sim.model_other_properties = model_properties
        results = fit_sim.get_model_properties(*list(model_parameters.values()))

        correct_results = model_parameters.copy()
        correct_results.update(model_properties)

        for k in results:
            result = results[k]
            if isinstance(result, np.ndarray):
                assert all(result == correct_results[k])
            else:
                assert result == correct_results[k]

    def test_prepare_sim(self, model_setup, participant_data_setup):
        model, model_other = model_setup
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim(participant_choice_property='Decisions',
                                 participant_reward_property='Rewards',
                                 model_fitting_variable='ActionProb',
                                 task_stimuli_property=None,
                                 fit_subset=None,
                                 action_options_property='valid_actions_combined')

        fitting = fit_sim.prepare_sim(model, model_other[0], model_other[1], participant_data)

        assert fitting == fit_sim.fitness

    def test_simulation(self, model_setup, participant_data_setup):
        model, model_other = model_setup
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim(participant_choice_property='Choices',
                                 participant_reward_property='Rewards',
                                 model_fitting_variable='ActionProb',
                                 task_stimuli_property=None,
                                 fit_subset=None,
                                 action_options_property=[1, 2])

        fitting = fit_sim.prepare_sim(model, model_other[0], model_other[1], participant_data)

        model_instance = fit_sim.fitted_model(0.5, 3)
        model_data = model_instance.return_task_state()
        result = model_data['ActionProb']
        correct_result = np.array([0.5, 0.64565631, 0.46257015, 0.62831619, 0.55601389, 0.51874122, 0.5093739,
                                   0.4906261, 0.4906261, 0.5093739, 0.5093739, 0.5093739])

        np.testing.assert_array_almost_equal(result, correct_result)

    def test_fitness(self, model_setup, participant_data_setup):
        model, model_other = model_setup
        participant_data = participant_data_setup

        fit_sim = fitSims.FitSim(participant_choice_property='Choices',
                                 participant_reward_property='Rewards',
                                 model_fitting_variable='ActionProb',
                                 task_stimuli_property=None,
                                 fit_subset=None,
                                 action_options_property=[1, 2])

        fitness = fit_sim.prepare_sim(model, model_other[0], model_other[1], participant_data)
        result = fitness(0.5, 3)
        correct_result = np.array([0.5, 0.64565631, 0.46257015, 0.62831619, 0.55601389, 0.51874122, 0.5093739,
                                   0.4906261, 0.4906261, 0.5093739, 0.5093739, 0.5093739])

        np.testing.assert_array_almost_equal(result, correct_result)