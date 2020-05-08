# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each successful investigation and storing it
with the data.
"""
#%% Import useful functions
import numpy as np

import pathlib

import collections

import simulation

#%% Set the model sets and task sets
number_actions = 6
number_cues = 1
repetitions = 1
alphaSet = np.repeat(np.array([0.1, 0.5, 0.9]), repetitions)
betaSet = np.array([0.1, 0.5, 1, 4, 16])

task_parameters = {}
task_static_properties = {'number_actions': number_actions,
                          'learning_length': 100,
                          'test_length': 50,
                          'reward_size': 1,
                          'action_reward_probabilities': collections.OrderedDict([('A', 0.80),
                                                                                  ('B', 0.20),
                                                                                  ('C', 0.70),
                                                                                  ('D', 0.30),
                                                                                  ('E', 0.60),
                                                                                  ('F', 0.40)]),
                          'learning_action_pairs': [('A', 'B'), ('C', 'D'), ('E', 'F')]}

model_parameters = {'alpha': alphaSet,
                    'beta': betaSet}
model_static_properties = {'number_actions': number_actions,
                           'number_cues': number_cues,
                           'action_codes': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5},
                           'expect': np.ones((number_actions, number_cues)) / 2,
                           'prior': np.ones(number_actions) / number_actions,
                           'stimulus_shaper_name': 'StimulusProbSelectDirect',
                           'reward_shaper_name': 'RewardProbSelectDirect',
                           'decision_function_name': 'weightProb',
                           'task_responses': ['A', 'B', 'C', 'D', 'E', 'F']}

#%% For simulating tasks
simulation.run(task_name='ProbSelect',
               task_changing_properties=task_parameters,
               task_constant_properties=task_static_properties,
               model_name='QLearn',
               model_changing_properties=model_parameters,
               model_constant_properties=model_static_properties,
               label='qLearn_probSelectSimSet',
               output_path=pathlib.Path(__file__).parent.absolute(),
               pickle=True,
               numpy_error_level='log') # 'raise','log'
