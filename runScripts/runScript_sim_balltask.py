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

import collections

import simulation

#%% Set the model sets and task sets

number_actions = 3
number_cues = 3
#repetitions = 30
#alphaSet = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
#betaSet = np.array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])
repetitions = 1
alphaSet = np.repeat(np.array([0.5]), repetitions)
betaSet = np.array([0.7])

task_parameters = {}
task_static_properties = {}

model_parameters = {'alpha': alphaSet,
                    'beta': betaSet}
model_static_properties = {'number_actions': number_actions,
                           'number_cues': number_cues,
                           'action_codes': {0: 0, 1: 1, 2: 2},
                           'expect': np.ones((number_actions, number_cues)) / 2,
                           'prior': np.ones(number_actions) / number_actions,
                           'stimulus_shaper_name': 'StimulusBalltaskSimple',
                           'reward_shaper_name': 'RewardBalltaskDirect',
                           'decision_function_name': 'weightProb',
                           'task_responses': ([0, 1, 2])}

#%% For simulating tasks
simulation.run(task_name='Balltask',
               task_changing_properties=task_parameters,
               task_constant_properties=task_static_properties,
               model_name='QLearn',
               model_changing_properties=model_parameters,
               model_constant_properties=model_static_properties,
               label='qLearn_balltask_simulation',
               pickle=True,
               numpy_error_level='log') # 'raise','log'
