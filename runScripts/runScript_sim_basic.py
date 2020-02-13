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
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
import os
filePath = os.path.realpath(__file__)
codePath = filePath.split('\\')[:-2]
sys.path.append("/".join(codePath))  # So code can be found from the main folder

# Other used function
import numpy as np

import simulation

#%% Set the model sets and tasks sets
number_actions = 2
number_cues = 1
repetitions = 1
alphaSet = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
betaSet = np.array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])

model_parameters = {'alpha': alphaSet,
                    'beta': betaSet}
model_static_properties = {'number_actions': number_actions,
                           'number_cues': number_cues,
                           'expect': [0.5, 0.5],
                           'prior': [0.5, 0.5],
                           'stimulus_shaper_name': 'StimulusBasicSimple',
                           'reward_shaper_name': 'RewardBasicDirect',
                           'decision_function_name': 'weightProb',
                           'task_responses': range(number_actions)}

#%% For simulating tasks
simulation.run(task_name='Basic',
               task_changing_properties=None,
               task_constant_properties=None,
               model_name='QLearn',
               model_changing_properties=model_parameters,
               model_constant_properties=model_static_properties,
               label='qLearn_BasicSimSet',
               pickle=True,
               numpy_error_level='log') # 'raise','log'
