# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np

import pytest

import collections

import outputting


#%% For saving
class TestClass_saving:
    def test_S(self):
        assert 1 == 1


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
        correct_result = collections.OrderedDict([('string', ["u'string'"])])
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
        correct_result = collections.OrderedDict([('dict_1', ["u'a'"]), ('dict_2', ["u'b'"])])
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
        correct_result = collections.OrderedDict([('dict_1_3', ["u'a'"]), ('dict_2', ["u'b'"])])
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
        correct_result = collections.OrderedDict([(u'dict_1_3', ['a']), (u'dict_2', ['b'])])
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
