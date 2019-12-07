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


#%% For newListDict
#class TestClass_newListDict:
#    def test_NLD_none(self):
#        store = {}
#        result = outputting.newListDict([1, 2, 3, 4], 4, store)
#        correct_result = collections.OrderedDict()
#        assert result == correct_result


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
        correct_result = (collections.OrderedDict([('string', None)]), None)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_DKG_list1(self):
        store = {'list': [1, 2, 3, 4, 5, 6]}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('list', np.array([[0], [1], [2], [3], [4], [5]]))]), None)
        assert (result[0]['list'] == correct_result[0]['list']).all()
        assert result[1] == correct_result[1]

    def test_DKG_num(self):
        store = {'num': 23.6}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('num', None)]), None)
        assert result[0] == correct_result[0]
        assert result[1] == correct_result[1]

    def test_DKG_array(self):
        store = {'array': np.array([1, 2, 3])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0], [1], [2]]))]), None)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

    def test_DKG_array2(self):
        store = {'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4],
                                                                       [0, 5], [1, 5]]))]),
                          None)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

    def test_DKG_array3(self):
        store = {'array': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])}
        result = outputting.dictKeyGen(store)
        correct_result = (collections.OrderedDict([('array', np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0, 1], [1, 1], [2, 1], [3, 1],
                                                                       [4, 1], [5, 1]]))]),
                          None)
        assert (result[0]['array'] == correct_result[0]['array']).all()
        assert result[1] == correct_result[1]

#%% For listKeyGen
class TestClass_listKeyGen:
    def test_LKG_list1(self):
        store = [1, 2, 3, 4, 5, 6]
        result = outputting.listKeyGen(store)
        correct_result = (np.array([[0], [1], [2], [3], [4], [5]]), None)
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
                          None)
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
