# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
import sys
sys.path.append("../")

import pytest
import itertools

import numpy as np

from fitAlgs.fitAlg import FitAlg
from modelGenerator import ModelGen
from fitAlgs.fitSims import FitSim

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

    modelInfos = [m for m in models.iter_details()]
    modelInfo = modelInfos[0]
    model = modelInfo[0]
    modelSetup = modelInfo[1:]

    return model, modelSetup

@pytest.fixture(scope='function')
def sim_setup():
    fit_sim = FitSim(participant_choice_property='Choices',
                     participant_reward_property='Rewards',
                     model_fitting_variable='ActionProb',
                     task_stimuli_property=None,
                     fit_subset=None,
                     action_options_property=[1, 2])

    return fit_sim

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


class TestClass_fitAlg:

    def test_FA_none(self):

        with pytest.raises(NameError, match='Please specify bounds for your parameters'):
            fit_alg = FitAlg()

    def test_FA_bounds(self):
        fit_alg = FitAlg(bounds={'alpha': (0, 1), 'beta': (0, np.inf)})
        assert isinstance(fit_alg, FitAlg)

    def test_FA_info(self):
        bounds = {'alpha': (0, 1), 'beta': (0, np.inf)}
        fit_alg = FitAlg(bounds=bounds)
        results = fit_alg.info()
        correct_results_alg = {'Name': 'FitAlg',
                               'fit_measure_function': '-loge',
                               'fit_measure_arguments': {},
                               'boundary_cost_function': None,
                               'bounds': bounds,
                               'extra_fit_measures': [],
                               'calculate_covariance': False,
                               'bound_ratio': 10**-6}

        correct_results_sim = {'Name': 'FitSim',
                               'participant_choice_property': 'Actions',
                               'participant_reward_property': 'Rewards',
                               'task_stimuli_property': None,
                               'action_options_property': None,
                               'model_fitting_variable': 'ActionProb',
                               'float_error_response_value': 10 ** -100,
                               'fit_subset': None}

        for res_key, res_val in results.items():
            if res_key == 'FitSim':
                for sim_key, sim_val in results[res_key].items():
                    assert sim_val == correct_results_sim[sim_key]
            else:
                assert res_val == correct_results_alg[res_key]

    def test_FA_basic(self, sim_setup):
        fit_alg = FitAlg(fit_sim=sim_setup,
                         fit_measure='-loge',
                         fit_measure_args={"numParams": 2,
                                           "number_actions": 2,
                                           "qualityThreshold": 20,
                                           "randActProb": 1/2},
                         extra_fit_measures=['-2log', 'BIC', 'r2', 'bayesFactor', 'BIC2norm'],
                         bounds={'alpha': (0, 1), 'beta': (0, np.inf)})
        assert isinstance(fit_alg, FitAlg)

if __name__ == '__main__':
    pytest.main()
    
#    pytest.set_trace()