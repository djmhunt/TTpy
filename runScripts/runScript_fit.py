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

#%%For data fitting
from dataFitting import data_fitting


#%% Set the model sets
number_actions = 6
number_cues = 1

modelParameters = {'alpha': (0, 1),
                   'beta': (0, 30)}
modelStaticArgs = {'number_actions': number_actions,
                   'number_cues': number_cues,
                   'action_codes': {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
                   'expect': np.ones((number_actions, number_cues)) / 2,
                   'prior': np.ones(number_actions) / number_actions,
                   'stimulus_shaper_name': 'StimulusProbSelectDirect',
                   'reward_shaper_name': 'RewardProbSelectDirect',
                   'decision_function_name': 'weightProb',
                   'expResponses': ["A", "B", "C", "D", "E", "F"]}

def data_processing(dat):
    for i, d in enumerate(dat['ValidActions']):
        dat['ValidActions_{}'.format(i)] = d
    return dat

#%% Run the data fitter
data_fitting(data_folder="./Outputs/test_sim_2020-1-10/Pickle/",
             data_format='pkl',
             data_file_filter="QLearn_modelData_sim-",
             data_extra_processing=data_processing,
             model_name='QLearn',
             model_changing_properties=modelParameters,
             model_constant_properties=modelStaticArgs,
             participantID='simID',
             participant_choices='Decisions',
             participant_rewards='Rewards',
             model_fit_value='ActionProb',
             fit_subset=float('Nan'),  # float('Nan'), None, range(0,40)
             task_stimuli=None,  #["stimCues"],
             participant_action_options=['ValidActions_0', 'ValidActions_1'],
             fit_method='Evolutionary',
             fit_measure="BIC2norm",
             fit_extra_measures=['-2log', 'BIC', 'r2', 'bayesFactor'],
             fit_measure_args={"numParams": len(modelParameters),
                               "number_actions": number_actions,
                               "qualityThreshold": 20,
                               "randActProb": 1/number_actions},
             label='qLearn_probSelect_fromSim',
             save_fitting_progress=True,
             pickle=True,
             numpy_error_level='log')  # 'raise','log'
