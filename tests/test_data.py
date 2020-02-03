# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest
import itertools

import cPickle as pickle

import scipy as sp
import numpy as np
import pandas as pd

import tempfile

import data

MAT_DATA = {'__globals__': [],
            '__header__': 'MATLAB 5.0 MAT-file, Platform: PCWIN, Created on: Thu Jan 28 13:40:07 2010',
            '__version__': '1.0',
            'bonuscrit': 450,
            'choiceRT': [0, 0, 0],
            'cumpts': [7,  9, 11],
            'dfile': 'subj1.mat',
            'subchoice': [2, 1, 1],
            'taskver': 1.05}

MAT_POINTS_SETS = [[7, 9, 11], [21, 26, 27], [37, 43, 44]]
MAT_CHOICE_SETS = [[2, 1, 1], [2, 2, 1], [2, 2, 1]]

SINGLE_DATA = {'Phase': ['Training', 'Training', 'Training'],
               'Trial': [1, 2, 3],
               'left_symbol': ['A', 'A', 'C'],
               'right_symbol': ['B', 'B', 'D'],
               'Left_PRESSED_0no_1yes': [0, 1, 0],
               'name': ['AB', 'AB', 'CD_inv'],
               'corr_resp': ['A', 'A', 'D']}

MULTI_DATA_PART_ORDER = {0: 0, 2: 1, 3: 2, 4: 3, 1: 4}
MULTI_DATA = {'subno': ['s1', 's1', 's1', 's1', 's1', 's1', 's1', 's1', 's1', 's10', 's10', 's10', 's10', 's10',
                        's10', 's10', 's10', 's10', 's11', 's11', 's11', 's11', 's11', 's11', 's11', 's11', 's11',
                        's19', 's19', 's19', 's19', 's19', 's19', 's19', 's19', 's19', 's2', 's2', 's2', 's2', 's2',
                        's2', 's2', 's2', 's2'],
              'task': ['weather', 'weather', 'weather', 'weather', 'weather', 'weather', 'weather', 'weather',
                       'weather', 'weather', 'weather', 'weather', 'weather', 'weather', 'weather', 'weather',
                       'weather', 'weather', 'disease', 'disease', 'disease', 'disease', 'disease', 'disease',
                       'disease', 'disease', 'disease', 'disease', 'disease', 'disease', 'disease', 'disease',
                       'disease', 'disease', 'disease', 'disease', 'weather', 'weather', 'weather', 'weather',
                       'weather', 'weather', 'weather', 'weather', 'weather'],
              'trialnum': [1, 2, 3, 4, 5, 6, 57, 58, 59, 1, 2, 3, 4, 5, 6, 57, 58, 59, 1, 2, 3, 4, 5, 6, 57, 58, 59,
                           1, 2, 3, 4, 5, 6, 57, 58, 59, 1, 2, 3, 4, 5, 6, 57, 58, 59],
              'phase': ['train', 'train', 'train', 'train', 'train', 'train', 'test', 'test', 'test', 'train',
                        'train', 'train', 'train', 'train', 'train', 'test', 'test', 'test', 'train', 'train',
                        'train', 'train', 'train', 'train', 'test', 'test', 'test', 'train', 'train', 'train',
                        'train', 'train', 'train', 'test', 'test', 'test', 'train', 'train', 'train', 'train',
                        'train', 'train', 'test', 'test', 'test'],
              'cue1': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
              'cue2': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
                       1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
              'cue3': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
              'cue4': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
              'outcome': [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, np.nan, np.nan, np.nan, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, np.nan,
                          np.nan, np.nan, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, np.nan, np.nan, np.nan, 2.0, 1.0, 1.0, 1.0, 1.0,
                          2.0, np.nan, np.nan, np.nan, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, np.nan, np.nan, np.nan],
              'response': [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0,
                           2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1],
              'resp_rew': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, np.nan, np.nan, np.nan, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, np.nan,
                           np.nan, np.nan, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, np.nan, np.nan, np.nan, 0.0, 1.0, 1.0, 0.0,
                           1.0, 1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan]}

SIM_DATA = [{'ActionProb': np.array([0.5, 0.5, 0.51874122, 0.53742985, 0.5, 0.51874122, 0.5, 0.48125878, 0.53183189]),
             'Actions': np.array([3, 5, 2, 2, 1, 5, 3, 3, 2]),
             'Decisions': np.array(['D', 'F', 'C', 'C', 'B', 'F', 'D', 'D', 'C'], dtype='<U1'),
             'ExpectedReward': np.array([0.35, 0.65, 0.65, 0.755, 0.35, 0.455, 0.455, 0.455, 0.455]),
             'Name': 'QLearn',
             'Rewards': np.array([0.0,  1.0,  1.0,  1.0,  0.0,  0.0, np.nan, np.nan, np.nan]),
             'Stimuli': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
             'ValidActions': np.array([['C', 'D'], ['E', 'F'], ['C', 'D'], ['C', 'D'], ['A', 'B'], ['E', 'F'], ['B', 'D'], ['D', 'A'], ['E', 'C']], dtype='<U1'),
             'actionCode': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5},
             'alpha': 0.3,
             'beta': 0.5,
             'decision_function': u"discrete.weightProb with expResponses : 'A', 'B', 'C', 'D', 'E', 'F'",
             'expectation': np.array([[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]]),
             'non_action': None,
             'number_actions': 6,
             'number_critics': 6,
             'number_cues': 1,
             'prior': np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667]),
             'reward_shaper': 'probSelect.RewardProbSelectDirect with Name : probSelect.RewardProbSelectDirect',
             'simID': '0',
             'stimulus_shaper': 'probSelect.StimulusProbSelectDirect with Name : probSelect.StimulusProbSelectDirect'},
            {'ActionProb': np.array([0.55045271, 0.42843235, 0.51518283, 0.50562476, 0.48125878, 0.46194872, 0.45636131, 0.54363869, 0.39258123]),
             'Actions': np.array([2, 3, 2, 4, 1, 4, 1, 3, 1]),
             'Decisions': np.array(['C', 'D', 'C', 'E', 'B', 'E', 'B', 'D', 'B'], dtype='<U1'),
             'ExpectedReward': np.array([0.9265, 0.805, 0.97795, 0.15, 0.105, 0.745, 0.745, 0.745, 0.745]),
             'Name': 'QLearn',
             'Rewards': np.array([1.0,  1.0,  1.0,  0.0,  0.0,  1.0, np.nan, np.nan, np.nan]),
             'Stimuli': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
             'ValidActions': np.array([['C', 'D'], ['C', 'D'], ['C', 'D'], ['E', 'F'], ['A', 'B'], ['E', 'F'], ['F', 'B'], ['D', 'F'], ['B', 'C']], dtype='<U1'),
             'actionCode': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5},
             'alpha': 0.7,
             'beta': 0.5,
             'decision_function': u"discrete.weightProb with expResponses : 'A', 'B', 'C', 'D', 'E', 'F'",
             'expectation': np.array([[0.5], [0.35], [0.755], [0.35], [0.5], [0.455]]),
             'non_action': None,
             'number_actions': 6,
             'number_critics': 6,
             'number_cues': 1,
             'prior': np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667]),
             'reward_shaper': 'probSelect.RewardProbSelectDirect with Name : probSelect.RewardProbSelectDirect',
             'simID': '1',
             'stimulus_shaper': 'probSelect.StimulusProbSelectDirect with Name : probSelect.StimulusProbSelectDirect'},
            {'ActionProb': np.array([0.33363277, 0.16005754, 0.17079548, 0.84631675, 0.75138264, 0.66507652, 0.89010362, 0.88347941, 0.48350599]),
             'Actions': np.array([3, 3, 1, 0, 0, 0, 2, 2, 5]),
             'Decisions': np.array(['D', 'D', 'B', 'A', 'A', 'A', 'C', 'C', 'F'], dtype='<U1'),
             'ExpectedReward': np.array([0.5635, 0.39445, 0.0735, 0.35, 0.245, 0.4715, 0.4715, 0.4715, 0.4715]),
             'Name': 'QLearn',
             'Rewards': np.array([0.0, 0.0, 0.0, 0.0,  0.0, 1.0, np.nan, np.nan, np.nan]),
             'Stimuli': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
             'ValidActions': np.array([['C', 'D'], ['C', 'D'], ['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B'], ['F', 'C'], ['C', 'A'], ['A', 'F']], dtype='<U1'),
             'actionCode': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5},
             'alpha': 0.3,
             'beta': 4.0,
             'decision_function': u"discrete.weightProb with expResponses : 'A', 'B', 'C', 'D', 'E', 'F'",
             'expectation': np.array([[0.5], [0.105], [0.97795], [0.805], [0.745], [0.455]]),
             'non_action': None,
             'number_actions': 6,
             'number_critics': 6,
             'number_cues': 1,
             'prior': np.array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667]),
             'reward_shaper': 'probSelect.RewardProbSelectDirect with Name : probSelect.RewardProbSelectDirect',
             'simID': '2',
             'stimulus_shaper': 'probSelect.StimulusProbSelectDirect with Name : probSelect.StimulusProbSelectDirect'}]

@pytest.fixture(scope="session")
def single_files(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data")
    file_names = {}

    mat_file_name = folder_name.join(MAT_DATA['dfile'])
    file_names['mat'] = str(mat_file_name)
    sp.io.savemat(str(mat_file_name), MAT_DATA)

    single_df = pd.DataFrame(SINGLE_DATA).set_index('Trial')
    csv_file_name = folder_name.join('subj1.csv')
    file_names['csv_single'] = csv_file_name
    single_df.to_csv(csv_file_name)
    xlsx_file_name = folder_name.join('subj1.xlsx')
    file_names['xlsx_single'] = xlsx_file_name
    single_df.to_excel(xlsx_file_name)

    multi_df = pd.DataFrame(MULTI_DATA).set_index(['subno', 'trialnum'])
    csv_file_name = folder_name.join('multisubject.csv')
    file_names['csv_multi'] = csv_file_name
    multi_df.to_csv(csv_file_name)
    xlsx_file_name = folder_name.join('multisubject.xlsx')
    file_names['xlsx_multi'] = xlsx_file_name
    multi_df.to_excel(xlsx_file_name)

    pkl_data = SIM_DATA[0]
    pkl_file_name = folder_name.join('QLearn_modelData_sim-0.pkl')
    file_names['pkl'] = str(pkl_file_name)
    with open(str(pkl_file_name), 'w') as w:
        pickle.dump(pkl_data, w)

    return str(folder_name), file_names

@pytest.fixture(scope="session")
def multi_files(tmpdir_factory):
    folder_name = tmpdir_factory.mktemp("data")
    file_names = {}

    mat_file_names = []
    for i, (cumulative_points, subject_choices) in enumerate(itertools.izip(MAT_POINTS_SETS, MAT_CHOICE_SETS)):
        mat_file_name = 'subj{}.mat'.format(i)
        MAT_DATA['dfile'] = mat_file_name
        file_path = folder_name.join(mat_file_name)
        mat_file_names.append(mat_file_name)

        MAT_DATA['cumpts'] = cumulative_points
        MAT_DATA['subchoice'] = subject_choices
        sp.io.savemat(str(file_path), MAT_DATA)
    file_names['mat'] = mat_file_names

    multi_df = pd.DataFrame(MULTI_DATA)
    csv_file_names = []
    xlsx_file_names = []
    for subj, subj_dat in multi_df.groupby('subno'):
        subj_dat_ind = subj_dat.set_index('trialnum')
        csv_file_name = folder_name.join('subj{}.csv'.format(subj))
        csv_file_names.append(str(csv_file_name).split('\\')[-1])
        subj_dat_ind.to_csv(csv_file_name)
        xlsx_file_name = folder_name.join('subj{}.xlsx'.format(subj))
        xlsx_file_names.append(str(xlsx_file_name).split('\\')[-1])
        subj_dat_ind.to_excel(xlsx_file_name)
    file_names['csv'] = sorted(csv_file_names, key=lambda x: int(x[5:].split('.')[0]))
    file_names['xlsx'] = sorted(xlsx_file_names, key=lambda x: int(x[5:].split('.')[0]))

    pkl_file_names = []
    for i, pkl_data in enumerate(SIM_DATA):
        pkl_file_name = folder_name.join('QLearn_modelData_sim-{}.pkl'.format(i))
        pkl_file_names.append(str(pkl_file_name).split('\\')[-1])
        with open(str(pkl_file_name), 'w') as w:
            pickle.dump(pkl_data, w)
    file_names['pkl'] = pkl_file_names

    return str(folder_name), file_names


@pytest.fixture(scope='session')
def multi_folders(tmpdir_factory):
    folder_name = tmpdir_factory.mktemp("data", numbered=False)
    folder_name_str = str(folder_name).replace('\\', '/')

    file_names = []
    for i, (cumulative_points, subject_choices) in enumerate(itertools.izip(MAT_POINTS_SETS, MAT_CHOICE_SETS)):
        mat_file_name = 'subj{}.mat'.format(i)
        MAT_DATA['dfile'] = mat_file_name
        folder_path = '{}/subj{}'.format(folder_name_str, i)
        folder_sub_path = tmpdir_factory.mktemp("data/subj{}".format(i), numbered=False)
        file_path = '{}/{}'.format(folder_path, mat_file_name)
        file_names.append(mat_file_name)

        MAT_DATA['cumpts'] = cumulative_points
        MAT_DATA['subchoice'] = subject_choices
        sp.io.savemat(str(file_path), MAT_DATA)

    return folder_name_str, file_names

#%% For Data generally
class TestClass_Data:
    def test_D_insufficient(self):
        dat = {}
        with pytest.raises(KeyError, match='participantID key not found in participant data: `ID`'):
            assert data.Data([dat, dat])

    def test_D_insufficient2(self):
        dat = {'ID': '1'}
        with pytest.raises(KeyError, match='choices key not found in participant 1 data: `actions`'):
            assert data.Data([dat, dat])

    def test_D_insufficient3(self):
        dat = {'ID': '1', 'actions': []}
        with pytest.raises(KeyError, match='feedbacks key not found in participant 1 data: `feedbacks`'):
            assert data.Data([dat, dat])

    def test_D_insufficient4(self):
        dat = {'ID': '1', 'actions': [], 'feedbacks': []}
        with pytest.raises(data.IDError, match='participantID must be unique. Found more than one instance of `1`'):
            assert data.Data([dat, dat])

    def test_D_minimal(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result = repr(data.Data([dat1, dat2]))
        correct_result = """[{u'ID': u'1', u'actions': [], u'feedbacks': []}, {u'ID': u'2', u'actions': [], u'feedbacks': []}]"""
        assert result == correct_result

    def test_D_minimal2(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.participantID
        correct_result = 'ID'
        assert result == correct_result

    def test_D_minimal3(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.choices
        correct_result = 'actions'
        assert result == correct_result

    def test_D_minimal4(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.feedbacks
        correct_result = 'feedbacks'
        assert result == correct_result

    def test_D_minimal5(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.stimuli
        correct_result = None
        assert result == correct_result

    def test_D_minimal6(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.action_options
        correct_result = None
        assert result == correct_result

    def test_D_minimal7(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2])
        result = result_dat.IDs
        correct_result = {'1': 0, '2': 1}
        assert result == correct_result

    def test_D_explicit(self):
        dat1 = {'partID': '1', 'actions': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['partID'] = '2'
        result_dat = data.Data([dat1, dat2], participantID='partID')
        result = result_dat.participantID
        correct_result = 'partID'
        assert result == correct_result

    def test_D_explicit2(self):
        dat1 = {'ID': '1', 'act': [], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], choices='act')
        result = result_dat.choices
        correct_result = 'act'
        assert result == correct_result

    def test_D_explicit3(self):
        dat1 = {'ID': '1', 'actions': [], 'rew': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], feedbacks='rew')
        result = result_dat.feedbacks
        correct_result = 'rew'
        assert result == correct_result

    def test_D_explicit4(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': [], 'stim': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], stimuli='stim')
        result = result_dat.stimuli
        correct_result = 'stim'
        assert result == correct_result

    def test_D_explicit5(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': [], 'val_act': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], action_options='val_act')
        result = result_dat.action_options
        correct_result = 'val_act'
        assert result == correct_result

    def test_D_length1(self):
        dat1 = {'ID': '1', 'actions': [1], 'feedbacks': []}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.LengthError, match='The number of values for choices and feedbacks must be the same: 1 choices and 0 feedbacks for participant `1`'):
            assert data.Data([dat1, dat2])

    def test_D_length2(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': [1]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.LengthError, match='The number of values for choices and feedbacks must be the same: 0 choices and 1 feedbacks for participant `1`'):
            assert data.Data([dat1, dat2])

    def test_D_length3(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': [], 'stimuli': [[1, 1]]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.LengthError, match='The number of values for choices and stimuli must be the same: 0 choices and 1 stimuli for participant `1`'):
            assert data.Data([dat1, dat2], stimuli='stimuli')

    def test_D_length4(self):
        dat1 = {'ID': '1', 'actions': [], 'feedbacks': [], 'valid_actions': [[1, 2]]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.LengthError, match='The number of values for choices and valid actions must be the same: 0 choices and 1 action_options for participant `1`'):
            assert data.Data([dat1, dat2], action_options='valid_actions')

    def test_D_combining(self):
        dat1 = {'ID': '1', 'actions': ['a', 'a', 'a'], 'feedbacks': [1, 1, 1], 'cue1': [1, 2, 3], 'cue2': [4, 5, 6]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], stimuli=['cue1', 'cue2'])
        result = result_dat[0][result_dat.stimuli]
        correct_result = np.array([[1, 4], [2, 5], [3, 6]])
        for r, cr in itertools.izip(result, correct_result):
            assert all(r == cr)

    def test_D_combining2(self):
        dat1 = {'ID': '1', 'actions': ['a', 'a', 'a'], 'feedbacks': [1, 1, 1], 'cue1': [[1], [2], [3]], 'cue2': [4, 5, 6]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.DimentionError, match='If you are using separate keys for each stimulus cue, they must all be 1D lists'):
            assert data.Data([dat1, dat2], stimuli=['cue1', 'cue2'])

    def test_D_combining3(self):
        dat1 = {'ID': '1', 'actions': ['a', 'a', 'a'], 'feedbacks': [1, 1, 1], 'act1': [1, 2, 3], 'act2': [4, 5, 6]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        result_dat = data.Data([dat1, dat2], action_options=['act1', 'act2'])
        result = result_dat[0][result_dat.action_options]
        correct_result = np.array([[1, 4], [2, 5], [3, 6]])
        for r, cr in itertools.izip(result, correct_result):
            assert all(r == cr)

    def test_D_combining4(self):
        dat1 = {'ID': '1', 'actions': ['a', 'a', 'a'], 'feedbacks': [1, 1, 1], 'act1': [[1], [2], [3]], 'act2': [4, 5, 6]}
        dat2 = dat1.copy()
        dat2['ID'] = '2'
        with pytest.raises(data.DimentionError, match='If you are using separate keys for each action option, they must all be 1D lists'):
            assert data.Data([dat1, dat2], action_options=['act1', 'act2'])

#%% For importing mat into Data
class TestClass_Mat:
    def test_mat_single(self, single_files):
        folder_name, file_names = single_files
        dat = data.Data.from_mat(folder=folder_name, participantID='dfile', choices='subchoice', feedbacks='cumpts')
        result = dat[0]
        correct_result = {'filename': file_names['mat'].split('\\')[-1],
                          'participant_ID': 'all',
                          'folder': folder_name.replace('\\', '/') + '/',
                          'bonuscrit': 450,
                          'choiceRT': np.array([0, 0, 0]),
                          'cumpts': np.array([7,  9, 11]),
                          'dfile': 'subj1.mat',
                          'subchoice': np.array([2, 1, 1]),
                          'taskver': 1.05
                          }

        for key, value in result.iteritems():
            try:
                assert all(value == correct_result[key])
            except TypeError:
                assert value == correct_result[key]

    def test_mat_triple(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_mat(folder=folder_name, participantID='dfile', choices='subchoice', feedbacks='cumpts')
        for result, file_name, points, choices in itertools.izip(dat, file_names['mat'], MAT_POINTS_SETS, MAT_CHOICE_SETS):
            correct_result = {'filename': file_name,
                              'participant_ID': file_name.split('.')[0][-1],
                              'folder': folder_name.replace('\\', '/') + '/',
                              'bonuscrit': 450,
                              'choiceRT': np.array([0, 0, 0]),
                              'cumpts': np.array(points),
                              'dfile': file_name,
                              'subchoice': np.array(choices),
                              'taskver': 1.05
                              }

            for key, value in result.iteritems():
                try:
                    assert all(value == correct_result[key])
                except TypeError:
                    assert value == correct_result[key]

    def test_mat_filtered(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_mat(folder=folder_name,
                                 file_name_filter='subj',
                                 participantID='dfile',
                                 choices='subchoice',
                                 feedbacks='cumpts')
        for result, file_name, points, choices in itertools.izip(dat, file_names['mat'], MAT_POINTS_SETS, MAT_CHOICE_SETS):
            correct_result = {'filename': file_name,
                              'participant_ID': file_name.split('.')[0][-1],
                              'folder': folder_name.replace('\\', '/') + '/',
                              'bonuscrit': 450,
                              'choiceRT': np.array([0, 0, 0]),
                              'cumpts': np.array(points),
                              'dfile': file_name,
                              'subchoice': np.array(choices),
                              'taskver': 1.05
                              }

            for key, value in result.iteritems():
                try:
                    assert all(value == correct_result[key])
                except TypeError:
                    assert value == correct_result[key]

    def test_mat_filtered2(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_mat(folder=folder_name,
                                 file_name_filter='boo',
                                 participantID='dfile',
                                 choices='subchoice',
                                 feedbacks='cumpts')
        result = dat
        correct_result = data.Data([], participantID='dfile', choices='subchoice', feedbacks='cumpts')
        assert result == correct_result

#%% For importing multiple folders into Data
class TestClass_Folders:
    def test_folders(self, multi_folders):
        folder_name, file_names = multi_folders
        dat_folders = ['{}/{}/'.format(folder_name, f[:-4]) for f in file_names]
        dat = data.Data.load_data(file_type='mat',
                                  folders=dat_folders,
                                  participantID='dfile',
                                  choices='subchoice',
                                  feedbacks='cumpts')
        for result, folder, file_name, points, choices in itertools.izip(dat, dat_folders, file_names, MAT_POINTS_SETS,
                                                                         MAT_CHOICE_SETS):
            correct_result = {'filename': file_name,
                              'participant_ID': file_name.split('.')[0][-1],
                              'folder': folder,
                              'bonuscrit': 450,
                              'choiceRT': np.array([0, 0, 0]),
                              'cumpts': np.array(points),
                              'dfile': file_name,
                              'subchoice': np.array(choices),
                              'taskver': 1.05
                              }

            for key, value in result.iteritems():
                try:
                    assert all(value == correct_result[key])
                except TypeError:
                    assert value == correct_result[key]

class TestClass_csv:
    def test_csv_single(self, single_files):
        folder_name, file_names = single_files
        dat = data.Data.from_csv(folder=folder_name, file_name_filter='subj1', participantID=None, feedbacks='corr_resp',
                                 stimuli=['left_symbol', 'right_symbol'], choices='Left_PRESSED_0no_1yes')
        result = dat[0]
        correct_result = SINGLE_DATA
        correct_result[data.DATA_KEYWORDS['filename']] = 'subj1.csv'
        correct_result[data.DATA_KEYWORDS['ID']] = 'all'
        correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
        correct_result['cues_combined'] = [[q1, q2] for q1, q2 in itertools.izip(SINGLE_DATA['left_symbol'], SINGLE_DATA['right_symbol'])]

        for key, value in result.iteritems():
            if isinstance(value, (list, np.ndarray)):
                if len(np.shape(value)) == 2:
                    for v1, v2 in itertools.izip(value, correct_result[key]):
                        assert all(v1 == v2)
                elif isinstance(value[-1], float) and np.isnan(value[-1]):
                    assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                    assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                else:
                    assert value == correct_result[key]
            else:
                assert value == correct_result[key]

    def test_csv_single_multi(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_csv(folder=folder_name, file_name_filter='multisubject', split_by='subno',
                                 choices='response', feedbacks='resp_rew', stimuli=['cue1', 'cue2', 'cue3', 'cue4'])
        for i, result in enumerate(dat):
            loc = MULTI_DATA_PART_ORDER[i]
            correct_result = {k: v[loc * 9:loc * 9 + 9] for k, v in MULTI_DATA.iteritems()}
            correct_result[data.DATA_KEYWORDS['filename']] = file_names["csv_multi"]
            correct_result[data.DATA_KEYWORDS['ID']] = correct_result['subno'][0]
            correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
            correct_result['cues_combined'] = [[q1, q2, q3, q4] for q1, q2, q3, q4 in
                                               itertools.izip(correct_result['cue1'],
                                                              correct_result['cue2'],
                                                              correct_result['cue3'],
                                                              correct_result['cue4'])]

            for key, value in result.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.shape(value)) == 2:
                        for v1, v2 in itertools.izip(value, correct_result[key]):
                            assert all(v1 == v2)
                    elif isinstance(value[-1], float) and np.isnan(value[-1]):
                        assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                        assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                    else:
                        assert value == correct_result[key]
                else:
                    assert value == correct_result[key]

    def test_csv_multi_files(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_csv(folder=folder_name, file_name_filter='subj', participantID='subno', choices='response',
                                 feedbacks='resp_rew', stimuli=['cue1', 'cue2', 'cue3', 'cue4'])
        for i, (f, result) in enumerate(itertools.izip(file_names["csv"], dat)):
            loc = MULTI_DATA_PART_ORDER[i]
            correct_result = {k: v[loc * 9:loc * 9 + 9] for k, v in MULTI_DATA.iteritems()}
            correct_result[data.DATA_KEYWORDS['filename']] = f
            correct_result[data.DATA_KEYWORDS['ID']] = correct_result['subno'][0]
            correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
            correct_result['cues_combined'] = [[q1, q2, q3, q4] for q1, q2, q3, q4 in
                                               itertools.izip(correct_result['cue1'],
                                                              correct_result['cue2'],
                                                              correct_result['cue3'],
                                                              correct_result['cue4'])]

            for key, value in result.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.shape(value)) == 2:
                        for v1, v2 in itertools.izip(value, correct_result[key]):
                            assert all(v1 == v2)
                    elif isinstance(value[-1], float) and np.isnan(value[-1]):
                        assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                        assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                    else:
                        assert value == correct_result[key]
                else:
                    assert value == correct_result[key]


class TestClass_xlsx:
    def test_xlsx_single(self, single_files):
        folder_name, file_names = single_files
        dat = data.Data.from_xlsx(folder=folder_name, file_name_filter='subj1', participantID=None, feedbacks='corr_resp',
                                  stimuli=['left_symbol', 'right_symbol'], choices='Left_PRESSED_0no_1yes')
        result = dat[0]
        correct_result = SINGLE_DATA
        correct_result[data.DATA_KEYWORDS['filename']] = 'subj1.xlsx'
        correct_result[data.DATA_KEYWORDS['ID']] = 'all'
        correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
        correct_result['cues_combined'] = [[q1, q2] for q1, q2 in itertools.izip(SINGLE_DATA['left_symbol'], SINGLE_DATA['right_symbol'])]

        for key, value in result.iteritems():
            if isinstance(value, (list, np.ndarray)):
                if len(np.shape(value)) == 2:
                    for v1, v2 in itertools.izip(value, correct_result[key]):
                        assert all(v1 == v2)
                elif isinstance(value[-1], float) and np.isnan(value[-1]):
                    assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                    assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                else:
                    assert value == correct_result[key]
            else:
                assert value == correct_result[key]

    def test_xlsx_single_multi(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_xlsx(folder=folder_name, file_name_filter='multisubject', split_by='subno',
                                 choices='response', feedbacks='resp_rew', stimuli=['cue1', 'cue2', 'cue3', 'cue4'])
        for i, result in enumerate(dat):
            loc = MULTI_DATA_PART_ORDER[i]
            correct_result = {k: v[loc * 9:loc * 9 + 9] for k, v in MULTI_DATA.iteritems()}
            correct_result[data.DATA_KEYWORDS['filename']] = file_names["xlsx_multi"]
            correct_result[data.DATA_KEYWORDS['ID']] = correct_result['subno'][0]
            correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
            correct_result['cues_combined'] = [[q1, q2, q3, q4] for q1, q2, q3, q4 in
                                               itertools.izip(correct_result['cue1'],
                                                              correct_result['cue2'],
                                                              correct_result['cue3'],
                                                              correct_result['cue4'])]

            for key, value in result.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.shape(value)) == 2:
                        for v1, v2 in itertools.izip(value, correct_result[key]):
                            assert all(v1 == v2)
                    elif isinstance(value[-1], float) and np.isnan(value[-1]):
                        assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                        assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                    else:
                        assert value == correct_result[key]
                else:
                    assert value == correct_result[key]

    def test_xlsx_multi_files(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_xlsx(folder=folder_name, file_name_filter='subj', participantID='subno', choices='response',
                                 feedbacks='resp_rew', stimuli=['cue1', 'cue2', 'cue3', 'cue4'])
        for i, (f, result) in enumerate(itertools.izip(file_names["xlsx"], dat)):
            loc = MULTI_DATA_PART_ORDER[i]
            correct_result = {k: v[loc * 9:loc * 9 + 9] for k, v in MULTI_DATA.iteritems()}
            correct_result[data.DATA_KEYWORDS['filename']] = f
            correct_result[data.DATA_KEYWORDS['ID']] = correct_result['subno'][0]
            correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'
            correct_result['cues_combined'] = [[q1, q2, q3, q4] for q1, q2, q3, q4 in
                                               itertools.izip(correct_result['cue1'],
                                                              correct_result['cue2'],
                                                              correct_result['cue3'],
                                                              correct_result['cue4'])]

            for key, value in result.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.shape(value)) == 2:
                        for v1, v2 in itertools.izip(value, correct_result[key]):
                            assert all(v1 == v2)
                    elif isinstance(value[-1], float) and np.isnan(value[-1]):
                        assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                        assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                    else:
                        assert value == correct_result[key]
                else:
                    assert value == correct_result[key]


class TestClass_pkl:
    def test_pkl_single(self, single_files):
        folder_name, file_names = single_files
        dat = data.Data.from_pkl(folder=folder_name, file_name_filter=None, participantID='simID',
                                 feedbacks='Rewards', stimuli=None, choices='Decisions')
        result = dat[0]
        correct_result = SIM_DATA[0]
        correct_result[data.DATA_KEYWORDS['filename']] = 'QLearn_modelData_sim-0.pkl'
        correct_result[data.DATA_KEYWORDS['ID']] = 'all'
        correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'

        for key, value in result.iteritems():
            if isinstance(value, (list, np.ndarray)):
                if len(np.shape(value)) == 2:
                    for v1, v2 in itertools.izip(value, correct_result[key]):
                        assert all(v1 == v2)
                elif isinstance(value[-1], float) and np.isnan(value[-1]):
                    assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                    assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                else:
                    try:
                        assert value == correct_result[key]
                    except ValueError:
                        assert all(value == correct_result[key])
            else:
                assert value == correct_result[key]

    def test_pkl_multi(self, multi_files):
        folder_name, file_names = multi_files
        dat = data.Data.from_pkl(folder=folder_name, file_name_filter=None, participantID='simID',
                                 feedbacks='Rewards', stimuli=None, choices='Decisions')
        for f, result, correct_result in itertools.izip(file_names["pkl"], dat, SIM_DATA):
            correct_result[data.DATA_KEYWORDS['filename']] = f
            correct_result[data.DATA_KEYWORDS['ID']] = correct_result['simID']
            correct_result[data.DATA_KEYWORDS['folder']] = folder_name.replace('\\', '/') + '/'

            for key, value in result.iteritems():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.shape(value)) == 2:
                        for v1, v2 in itertools.izip(value, correct_result[key]):
                            assert all(v1 == v2)
                    elif isinstance(value[-1], float) and np.isnan(value[-1]):
                        assert all([i == j for i, j in itertools.izip(value, correct_result[key]) if not np.isnan(i)])
                        assert all([np.isnan(j) for i, j in itertools.izip(value, correct_result[key]) if np.isnan(i)])
                    else:
                        try:
                            assert value == correct_result[key]
                        except ValueError:
                            assert all(value == correct_result[key])
                else:
                    assert value == correct_result[key]