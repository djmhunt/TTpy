# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
import numpy as np

import pytest
import os
import itertools
import logging

import collections
import outputting


@pytest.fixture(scope="session")
def output_folder(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data", numbered=True)

    return folder_name


#%% For saving
class TestClass_saving:
    def test_S_none(self, caplog):
        caplog.set_level(logging.INFO)

        with outputting.Saving() as saving:
            captured = caplog.records.copy()
            assert saving is None

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Framework']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message for k in captured]
        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Beginning task labelled: Untitled']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_close(self, caplog):
        caplog.set_level(logging.INFO)

        with outputting.Saving() as saving:
            assert saving is None

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Framework', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message for k in captured]
        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Beginning task labelled: Untitled',
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_config(self, caplog):
        caplog.set_level(logging.INFO)
        config = {'label': None,
                  'config_file': None,
                  'output_path': None,
                  'pickle': False,
                  'min_log_level': 'INFO',
                  'numpy_error_level': 'log'}

        with outputting.Saving(config=config) as saving:
            assert saving is None

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Framework', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message for k in captured]
        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Beginning task labelled: Untitled',
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_label(self, caplog, tmpdir):
        caplog.set_level(logging.INFO)

        label = 'label'
        current_date = outputting.date()
        path_str = str(tmpdir)
        path_clean = path_str.replace('\\', '/')
        log_path = '{}/Outputs/{}_{}/log.txt'.format(path_clean, label, current_date)

        config = {'label': label,
                  'config_file': None,
                  'output_path': path_str,
                  'pickle': False,
                  'min_log_level': 'INFO',
                  'numpy_error_level': 'log'}

        with outputting.Saving(config=config) as saving:
            assert callable(saving)

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Setup', 'Framework', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message for k in captured]
        correct = ['{}'.format(current_date),
                   'Log initialised',
                   'The log you are reading was written to {}'.format(log_path),
                   'Beginning task labelled: {}'.format(label),
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        with open(log_path) as log:
            cleaned_log = [l[37:].strip() for l in log.readlines()]
        for correct_line, standard_captured_line in itertools.zip_longest(correct, cleaned_log):
            assert standard_captured_line == correct_line


#%% For folderSetup
class TestClass_folderSetup:
    def test_FS_basic(self, tmpdir):
        path_str = str(tmpdir)
        result = outputting.folder_setup('test', '20-2-2020', base_path=path_str)
        correct_result = path_str.replace('\\', '/') + '/Outputs/test_20-2-2020/'
        assert os.path.exists(result)
        assert result == correct_result
        assert os.path.exists(correct_result + 'data/')
        assert not os.path.exists(correct_result + 'Pickle/')

    def test_FS_double(self, tmpdir):
        path_str = str(tmpdir)
        result1 = outputting.folder_setup('test', '20-2-2020', base_path=path_str)
        result2 = outputting.folder_setup('test', '20-2-2020', base_path=path_str)
        correct_result = path_str.replace('\\', '/') + '/Outputs/test_20-2-2020_no_1/'
        assert os.path.exists(result2)
        assert result2 == correct_result

    def test_FS_pickle(self, tmpdir):
        path_str = str(tmpdir)
        result = outputting.folder_setup('test', '20-2-2020', base_path=path_str, pickle_data=True)
        correct_result = path_str.replace('\\', '/') + '/Outputs/test_20-2-2020/'
        assert os.path.exists(correct_result + 'Pickle/')


#%% For fileNameGenerator
class TestClass_fileNameGenerator:
    def test_FNG_none(self):

        file_name_generator = outputting.file_name_generator()
        assert callable(file_name_generator)

    def test_FNG_none2(self):

        file_name_generator = outputting.file_name_generator()
        result = file_name_generator("", "")
        correct_result = os.getcwd().replace('\\', '/') + '/'
        assert result == correct_result

    def test_FNG_none3(self, output_folder):
        output_path = str(output_folder)

        file_name_generator = outputting.file_name_generator(output_folder=output_path)
        result = file_name_generator("", "")
        correct_result = output_path.replace('\\', '/') + '/'
        assert result == correct_result

    def test_FNG_typeless(self, output_folder):
        output_path = str(output_folder)

        file_name_generator = outputting.file_name_generator(output_folder=output_path)
        result = file_name_generator("a", "")
        correct_result = output_path.replace('\\', '/') + '/a'

        assert result == correct_result

    def test_FNG_nameless(self, output_folder):
        output_path = str(output_folder)

        file_name_generator = outputting.file_name_generator(output_folder=output_path)
        result = file_name_generator("", "a")
        correct_result = output_path.replace('\\', '/') + '/.a'

        assert result == correct_result

    def test_FNG_basic(self, output_folder):
        output_path = str(output_folder)

        file_name_generator = outputting.file_name_generator(output_folder=output_path)
        result = file_name_generator("a", "b")
        correct_result = output_path.replace('\\', '/') + '/a.b'
        assert result == correct_result

    def test_FNG_basic2(self):
        file_name_generator = outputting.file_name_generator(output_folder='./')
        result = file_name_generator("a", "b")
        correct_result = './a.b'
        assert result == correct_result

        result = file_name_generator("a", "b")
        correct_result = './a_1.b'
        assert result == correct_result


#%% For fancy_logger
class TestClass_fancy_logger:
    def test_FL_none(self, caplog):
        caplog.set_level(logging.INFO)

        close_loggers = outputting.fancy_logger()

        captured = caplog.records
        standard_captured = [k.message for k in captured]

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        for logger in [k.name for k in captured]:
            assert logger == 'Setup'

        correct = ['{}'.format(outputting.date()),
                   'Log initialised']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line
        close_loggers()

    def test_FL_file(self, caplog, tmpdir):
        caplog.set_level(logging.INFO)

        path_str = str(tmpdir).replace('\\', '/')
        log_path = '{}/log.txt'.format(path_str)

        close_loggers = outputting.fancy_logger(log_file=log_path)

        captured = caplog.records
        standard_captured = [k.message for k in captured]

        date = outputting.date()

        correct = ['{}'.format(date),
                   'Log initialised',
                   'The log you are reading was written to {}'.format(log_path)]

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        with open(log_path) as log:
            cleaned_log = [l[37:].strip() for l in log.readlines()]
        for correct_line, standard_captured_line in itertools.zip_longest(correct, cleaned_log):
            assert standard_captured_line == correct_line

        close_loggers()

    def test_FL_close(self, caplog):
        caplog.set_level(logging.INFO)

        close_loggers = outputting.fancy_logger()
        close_loggers()

        captured = caplog.records
        standard_captured = [k.message for k in captured]

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        for logger in [k.name for k in captured]:
            assert logger == 'Setup'

        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line


#%% For flatDictKeySet
class TestClass_flatDictKeySet:
    def test_FDKS_none(self):
        store = []
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict()
        assert result == correct_result

    def test_FDKS_string(self):
        store = [{'string': 'string'}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('string', None)])
        assert result == correct_result

    def test_FDKS_list1(self):
        store = [{'list': [1, 2, 3, 4, 5, 6]}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('list', np.array([[0], [1], [2], [3], [4], [5]]))])
        assert (result['list'] == correct_result['list']).all()

    def test_FDKS_num(self):
        store = [{'num': 23.6}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('num', None)])
        assert result == correct_result

    def test_FDKS_array(self):
        store = [{'array': np.array([1, 2, 3])}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('array', np.array([[0], [1], [2]]))])
        assert (result['array'] == correct_result['array']).all()

    def test_FDKS_array2(self):
        store = [{'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4],
                                                                       [0, 5], [1, 5]]))])
        assert (result['array'] == correct_result['array']).all()

    def test_FDKS_array3(self):
        store = [{'array': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0, 1], [1, 1], [2, 1], [3, 1],
                                                                       [4, 1], [5, 1]]))])
        assert (result['array'] == correct_result['array']).all()

    def test_FDKS_dict(self):
        store = [{'dict': {1: "a", 2: "b"}}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('dict', collections.OrderedDict([(1, None), (2, None)]))])
        assert result['dict'] == correct_result['dict']

    def test_FDKS_dict2(self):
        store = [{'dict': {1: [1, 2, 3], 2: [[1, 2], [3, 4]]}}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('dict', collections.OrderedDict([(1, np.array([[0], [1], [2]])),
                                                                                     (2, [[0, 0], [1, 0],
                                                                                          [0, 1], [1, 1]])]))])
        assert (result['dict'][1] == correct_result['dict'][1]).all()
        assert (result['dict'][2] == correct_result['dict'][2]).all()

    def test_FDKS_dict3(self):
        store = [{'dict': {1: {3: "a"}, 2: "b"}}]
        result = outputting.flatDictKeySet(store)
        correct_result = collections.OrderedDict([('dict', collections.OrderedDict([(1, collections.OrderedDict([(3, None)])),
                                                                                     (2, None)]))])
        assert result['dict'] == correct_result['dict']

    # TODO: Upgrade flatDictKeySet to cope with this
    #def test_FDKS_list_dict(self):
    #    store = [{'list': [collections.OrderedDict([("A", 0.3), ("B", 0.7)]), collections.OrderedDict([(1, 0.7), (2, 0.3)])]}]
    #    result = outputting.flatDictKeySet(store)
    #    correct_result = collections.OrderedDict([('list', np.array([[0, "A"], [0, "B"], [1, 1], [1, 2]]))])
    #    assert (result['list'] == correct_result['list']).all()


#%% For newFlatDict
class TestClass_newFlatDict:
    def test_NFD_none(self):
        store = []
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict()
        assert result == correct_result

    def test_NFD_none2(self):
        store = [{}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict()
        assert result == correct_result

    def test_NFD_string(self):
        store = [{'string': 'string'}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('string', ["'string'"])])
        assert result == correct_result

    def test_NFD_list1(self):
        store = [{'list': [1, 2, 3, 4, 5, 6]}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('list_[0]', [1]), ('list_[1]', [2]), ('list_[2]', [3]),
                                                  ('list_[3]', [4]), ('list_[4]', [5]), ('list_[5]', [6])])
        assert result == correct_result

    def test_NFD_num(self):
        store = [{'num': 23.6}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('num', ['23.6'])])
        assert result == correct_result

    def test_NFD_num2(self):
        store = [{'num': 23.6}, {'num': 29}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('num', ['23.6', '29'])])
        assert result == correct_result

    def test_NFD_array(self):
        store = [{'array': np.array([1, 2, 3])}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('array_[0]', [1]),
                                                  ('array_[1]', [2]),
                                                  ('array_[2]', [3])])
        assert result == correct_result

    def test_NFD_array2(self):
        store = [{'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('array_[0 0]', [1]), ('array_[1 0]', [7]), ('array_[0 1]', [2]),
                                                  ('array_[1 1]', [8]), ('array_[0 2]', [3]), ('array_[1 2]', [9]),
                                                  ('array_[0 3]', [4]), ('array_[1 3]', [10]), ('array_[0 4]', [5]),
                                                  ('array_[1 4]', [11]), ('array_[0 5]', [6]), ('array_[1 5]', [12])])
        assert result == correct_result

    def test_NFD_array3(self):
        store = [{'array': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('array_[0 0]', [1]), ('array_[1 0]', [3]), ('array_[2 0]', [5]),
                                                  ('array_[3 0]', [7]), ('array_[4 0]', [9]), ('array_[5 0]', [11]),
                                                  ('array_[0 1]', [2]), ('array_[1 1]', [4]), ('array_[2 1]', [6]),
                                                  ('array_[3 1]', [8]), ('array_[4 1]', [10]), ('array_[5 1]', [12])])
        assert result == correct_result

    def test_NFD_dict(self):
        store = [{'dict': {1: "a", 2: "b"}}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('dict_1', ["'a'"]), ('dict_2', ["'b'"])])
        assert result == correct_result

    def test_NFD_dict2(self):
        store = [{'dict': {1: [1, 2, 3], 2: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]}}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('dict_1_[0]', [1]), ('dict_1_[1]', [2]), ('dict_1_[2]', [3]),
                                                  ('dict_2_[0 0]', [1]), ('dict_2_[1 0]', [12]), ('dict_2_[0 1]', [2]),
                                                  ('dict_2_[1 1]', [13]), ('dict_2_[0 2]', [3]), ('dict_2_[1 2]', [14]),
                                                  ('dict_2_[0 3]', [4]), ('dict_2_[1 3]', [15]), ('dict_2_[0 4]', [5]),
                                                  ('dict_2_[1 4]', [16]), ('dict_2_[0 5]', [6]), ('dict_2_[1 5]', [17]),
                                                  ('dict_2_[0 6]', [7]), ('dict_2_[1 6]', [18]), ('dict_2_[0 7]', [8]),
                                                  ('dict_2_[1 7]', [19]), ('dict_2_[0 8]', [9]), ('dict_2_[1 8]', [20]),
                                                  ('dict_2_[0 9]', [10]), ('dict_2_[1 9]', [21]), ('dict_2_[ 0 10]', [11]),
                                                  ('dict_2_[ 1 10]', [22])])
        assert result == correct_result

    def test_NFD_dict3(self):
        store = [{'dict': {1: {3: "a"}, 2: "b"}}]
        result = outputting.newFlatDict(store)
        correct_result = collections.OrderedDict([('dict_1_3', ["'a'"]), ('dict_2', ["'b'"])])
        assert result == correct_result

    # TODO: Upgrade newFlatDict to cope with this
    # def test_NFD_list_dict(self):
    #    store = {'listDict': [collections.OrderedDict([("A", 0.3), ("B", 0.7)]), collections.OrderedDict([(1, 0.7), (2, 0.3)])]}
    #    result = outputting.newFlatDict(store)
    #    correct_result = collections.OrderedDict([('listDict_[0]_"A"', [0.3]), ('listDict_[0]_"B"', [0.7]),
    #                                              ('listDict_[1]_1', [0.7]), ('listDict_[1]_1', [0.3])])
    #    assert result == correct_result


#%% For newListDict
class TestClass_newListDict:
    def test_NLD_none(self):
        store = {}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict()
        assert result == correct_result

    def test_NLD_string(self):
        store = {'string': 'string'}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('string', ['string'])])
        assert result == correct_result

    def test_NLD_list1(self):
        store = {'list': [1, 2, 3, 4, 5, 6]}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('list', [1, 2, 3, 4, 5, 6])])
        assert result == correct_result

    def test_NLD_num(self):
        store = {'num': 23.6}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('num', [23.6])])
        assert result == correct_result

    def test_NLD_array(self):
        store = {'array': np.array([1, 2, 3])}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('array', [1, 2, 3])])
        assert result == correct_result

    def test_NLD_array2(self):
        store = {'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('array_[0]', [1, 2, 3, 4, 5, 6]),
                                                  ('array_[1]', [7, 8, 9, 10, 11, 12])])
        assert result == correct_result

    def test_NLD_array3(self):
        store = {'array': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('array_[0]', [1, 2]), ('array_[1]', [3, 4]), ('array_[2]', [5, 6]),
                                                  ('array_[3]', [7, 8]), ('array_[4]', [9, 10]), ('array_[5]', [11, 12])])
        assert result == correct_result

    def test_NLD_dict(self):
        store = {'dict': {1: "a", 2: "b"}}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('dict_1', ['a']), ('dict_2', ['b'])])
        assert result == correct_result

    def test_NLD_dict2(self):
        store = {'dict': {1: [1, 2, 3], 2: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]}}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('dict_1', [1, 2, 3, None, None, None, None, None, None, None, None]),
                                                  ('dict_2_[0]', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                                                  ('dict_2_[1]', [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])])
        assert result == correct_result

    def test_NLD_dict3(self):
        store = {'dict': {1: {3: "a"}, 2: "b"}}
        result = outputting.newListDict(store)
        correct_result = collections.OrderedDict([('dict_1_3', ['a']), ('dict_2', ['b'])])
        assert result == correct_result

    def test_NLD_dict4(self):
        store = {'dict': {1: "a", 2: "b"}}
        result = outputting.newListDict(store, maxListLen=3)
        correct_result = collections.OrderedDict([('dict_1', ['a', None, None]), ('dict_2', ['b', None, None])])
        assert result == correct_result

    # TODO: Upgrade newListDict to cope with this
    #def test_NLD_list_dict(self):
    #    store = {'listDict': [collections.OrderedDict([("A", 0.3), ("B", 0.7)]), collections.OrderedDict([(1, 0.7), (2, 0.3)])]}
    #    result = outputting.newListDict(store)
    #    correct_result = collections.OrderedDict([('listDict_[0 "A"]', [0.3]), ('listDict_[0 "B"]', [0.7]),
    #                                              ('listDict_[1 1]', [0.7]), ('listDict_[1 1]', [0.3])])
    #    assert result == correct_result


#%% For pad
class TestClass_pad:
    def test_P_none(self):
        result = outputting.pad([1, 2, 3, 4], 4)
        correct_result = [1, 2, 3, 4]
        assert result == correct_result

    def test_P_more(self):
        result = outputting.pad([1, 2, 3, 4], 6)
        correct_result = [1, 2, 3, 4, None, None]
        assert result == correct_result

    def test_P_less(self):
        result = outputting.pad([1, 2, 3, 4], 3)
        correct_result = [1, 2, 3, 4]
        assert result == correct_result


#%% For listSelection
class TestClass_listSelection:
    def test_LS_simple(self):
        result = outputting.listSelection([1, 2, 3], (0,))
        correct_result = 1
        assert result == correct_result

    def test_LS_simple2(self):
        result = outputting.listSelection([[1, 2, 3], [4, 5, 6]], (0,))
        correct_result = [1, 2, 3]
        assert result == correct_result

    def test_LS_double(self):
        result = outputting.listSelection([[1, 2, 3], [4, 5, 6]], (0, 2))
        correct_result = 3
        assert result == correct_result

    def test_LS_string(self):
        result = outputting.listSelection([["A", "B", "C"], ["D", "E", "F"]], (0, 2))
        correct_result = "C"
        assert result == correct_result

    def test_LS_none(self):
        result = outputting.listSelection([[1, 2, 3], [4, 5, 6]], ())
        correct_result = None
        assert result == correct_result

    def test_LS_over(self):
        result = outputting.listSelection([[1, 2, 3], [4, 5, 6]], (0, 2, 1))
        correct_result = None
        assert result == correct_result


#%% For dictKeyGen
class TestClass_dictKeyGen:
    def test_DKG_none(self):
        store = {}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict(), None)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_DKG_string(self):
        store = {'string': 'string'}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('string', None)]), 1)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_DKG_list1(self):
        store = {'list': [1, 2, 3, 4, 5, 6]}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('list', np.array([[0], [1], [2], [3], [4], [5]]))]), 1)
        assert (result[0]['list'] == correct_result[0]['list']).all()
        assert result[1] == correct_result[1]

    def test_DKG_num(self):
        store = {'num': 23.6}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('num', None)]), 1)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_DKG_array(self):
        store = {'array': np.array([1, 2, 3])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0], [1], [2]]))]), 1)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

    def test_DKG_array2(self):
        store = {'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4],
                                                                       [0, 5], [1, 5]]))]),
                          1)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

    def test_DKG_array3(self):
        store = {'array': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0, 1], [1, 1], [2, 1], [3, 1],
                                                                       [4, 1], [5, 1]]))]),
                          1)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

    def test_DKG_dict(self):
        store = {'dict': {1: "a", 2: "b"}}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('dict', collections.OrderedDict([(1, None), (2, None)]))]), 1)
        assert result[0]['dict'] == correct_result[0]['dict']
        assert result[1] == correct_result[1]

    def test_DKG_dict2(self):
        store = {'dict': {1: [1, 2, 3], 2: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]}}
        result = outputting.dictKeyGen(store, abridge=True)
        correct_result = (collections.OrderedDict([('dict', collections.OrderedDict([(1, np.array([[0], [1], [2]])),
                                                                                     (2, None)]))]),
                          1)
        assert (result[0]['dict'][1] == correct_result[0]['dict'][1]).all()
        assert result[0]['dict'][2] == correct_result[0]['dict'][2]
        assert result[1] == correct_result[1]

    def test_DKG_dict3(self):
        store = {'dict': {1: {3: "a"}, 2: "b"}}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('dict', collections.OrderedDict([(1, collections.OrderedDict([(3, None)])),
                                                                                     (2, None)]))]),
                          1)
        assert result[0]['dict'] == correct_result[0]['dict']
        assert result[1] == correct_result[1]

    # TODO: Upgrade dictKeyGen to cope with this
    #def test_DKG_list_dict(self):
    #    store = {'list': [collections.OrderedDict([("A", 0.3), ("B", 0.7)]), collections.OrderedDict([(1, 0.7), (2, 0.3)])]}
    #    result = outputting.dictKeyGen(store)
    #    correct_result = (collections.OrderedDict([('list', np.array([[0, "A"], [0, "B"], [1, 1], [1, 2]]))]), None)
    #    assert (result[0]['list'] == correct_result[0]['list']).all()
    #    assert result[1] == correct_result[1]


#%% For listKeyGen
class TestClass_listKeyGen:
    def test_LKG_empty(self):
        store = []
        result = outputting.listKeyGen(store)
        correct_result = (None, None)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_LKG_list1(self):
        store = [1, 2, 3, 4, 5, 6]
        result = outputting.listKeyGen(store)
        correct_result = (np.array([[0], [1], [2], [3], [4], [5]]), 1)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list2(self):
        store = [1, 2, 3, 4, 5, 6]
        result = outputting.listKeyGen(store, maxListLen=10)
        correct_result = (np.array([[0], [1], [2], [3], [4], [5]]), 10)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list3(self):
        store = [1, 2, 3, 4, 5, 6]
        result = outputting.listKeyGen(store, returnList=True, maxListLen=10)
        correct_result = (None, 10)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_LKG_list4(self):
        store = [1, 2, 3, 4, 5, 6]
        result = outputting.listKeyGen(store, returnList=True)
        correct_result = (None, 6)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_LKG_list5(self):
        store = [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store)
        correct_result = (np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5]]),
                          1)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list6(self):
        store = [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store, returnList=True)
        correct_result = (np.array([[0], [1]]), 6)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list7(self):
        store = [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store, abridge=True)
        correct_result = (None, None)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_LKG_list8(self):
        store = [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store, returnList=True, abridge=True)
        correct_result = (np.array([[0], [1]]), 6)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list9(self):
        store = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store, returnList=True, abridge=True)
        correct_result = (np.array([[0], [1]]), 10)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list10(self):
        store = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        result = outputting.listKeyGen(store, returnList=True, abridge=True, maxListLen=11)
        correct_result = (np.array([[0], [1]]), 11)
        assert (result[0] == correct_result[0]).all()
        assert result[1] == correct_result[1]

    def test_LKG_list11(self):
        store = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]]
        result = outputting.listKeyGen(store, returnList=True, abridge=True)
        correct_result = (None, 2)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_LKG_list12(self):
        store = [[1, 2, 3, 4, 5, 6]]
        result = outputting.listKeyGen(store, returnList=True, abridge=True)
        correct_result = (np.array([[0]]), 6)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    # TODO: Upgrade listKeyGen to cope with this
    #def test_LKG_dict(self):
    #    store = [collections.OrderedDict([("A", 0.3), ("B", 0.7)]), collections.OrderedDict([(1, 0.7), (2, 0.3)])]
    #    result = outputting.listKeyGen(store)
    #    correct_result = (np.array([[0, "A"], [0, "B"], [1, 1], [1, 2]]), None)
    #    assert result[0] == correct_result[0]
    #    assert result[1] == correct_result[1]
