# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
import pytest
import os
import itertools

import numpy as np

import utils


@pytest.fixture(scope="module")
def listMerTestData():
    
    v = 3  # type: int
    interSets = [[i for i in range(a-v, a)] for a in range(1, v)]
    
    answer = np.array([[-2, -1,  0, -2, -1,  0, -2, -1,  0],
                       [-1, -1, -1,  0,  0,  0,  1,  1,  1]])
    
    return interSets, answer

#%% For list_all_equal
class TestClass_list_all_equal:
    def test_LAE_empty(self):
        input = []
        output = utils.list_all_equal(input)
        correct_output = True
        assert output == correct_output

    def test_LAE_single(self):
        input = [1]
        output = utils.list_all_equal(input)
        correct_output = True
        assert output == correct_output

    def test_LAE_double(self):
        input = [1, 1]
        output = utils.list_all_equal(input)
        correct_output = True
        assert output == correct_output

    def test_LAE_double2(self):
        input = [1, 2]
        output = utils.list_all_equal(input)
        correct_output = False
        assert output == correct_output

    def test_LAE_nan(self):
        input = [np.nan]
        output = utils.list_all_equal(input)
        correct_output = True
        assert output == correct_output

    def test_LAE_nan2(self):
        input = [np.nan, np.nan]
        output = utils.list_all_equal(input)
        correct_output = True
        assert output == correct_output

    def test_LAE_nan3(self):
        input = [1, np.nan]
        output = utils.list_all_equal(input)
        correct_output = False
        assert output == correct_output

    def test_LAE_nan3(self):
        input = [np.nan, 1]
        output = utils.list_all_equal(input)
        correct_output = False
        assert output == correct_output


#%% For mergeDatasets
class TestClass_mergeDatasets:
    def test_mergeDatasets(self):
        data = [{'a': [1, 2, 3], 'b': [7, 8, 9]}, {'b': [4, 5, 6], 'c': 'string', 'd': 5}]
        results = utils.mergeDatasets(data)
        correct_result = {'a': [[1, 2, 3], None], 'd': [None, 5], 'c': [None, 'string'], 'b': [[7, 8, 9], [4, 5, 6]]}
        assert results == correct_result

    def test_mergeDatasets2(self):
        data = [{'a': [1, 2, 3], 'b': [7, 8, 9]}, {'b': [4, 5, 6], 'c': 'string', 'd': 5}]
        results = utils.mergeDatasets(data, extend=True)
        correct_result = {'a': [1, 2, 3, None], 'c': [None, 'string'], 'b': [7, 8, 9, 4, 5, 6], 'd': [None, 5]}
        assert results == correct_result

    def test_mergeDatasets3(self):
        data = [{'b': np.array([[7, 8, 9], [1, 2, 3]])}, {'b': np.array([[4, 5, 6], [2, 3, 4]])}]
        results = utils.mergeDatasets(data, extend=True)
        correct_result = {'b': [np.array([7, 8, 9]), np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([2, 3, 4])]}
        for r, c in itertools.zip_longest(results['b'], correct_result['b']):
            assert all(r == c)

    def test_mergeDatasets4(self):
        data = [{'b': np.array([[7, 8, 9], [1, 2, 3]])}, {'b': np.array([[4, 5, 6], [2, 3, 4]])}]
        results = utils.mergeDatasets(data)
        correct_result = {'b': [np.array([[7, 8, 9], [1, 2, 3]]), np.array([[4, 5, 6], [2, 3, 4]])]}
        for r, c in itertools.zip_longest(results['b'], correct_result['b']):
            for rp, cp in itertools.zip_longest(r, c):
                assert all(rp == cp)

#%% For listMergeGen
class TestClass_listMergeGen:
    def test_listMergeGen(self, listMerTestData):
        
        (interSets, answer) = listMerTestData
        listData = [i for i in utils.listMergeGen(*interSets)]
        assert (np.array(listData).T == answer).all()
        
    def test_listMergeGenArray(self, listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For numpy Arrays                    
        interSetsTuple = [np.array(i) for i in interSets]
        listData = [i for i in utils.listMergeGen(*interSetsTuple)]
        assert (np.array(listData).T == answer).all()
        
    def test_listMergeGenTuple(self, listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a tuple                    
        interSetsTuple = tuple(interSets) 
        listData = [i for i in utils.listMergeGen(*interSetsTuple)]
        assert (np.array(listData).T == answer).all()
        
    def test_listMergeGenGen(self, listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a generator    
        interSetsGen = (i for i in interSets)
        listData = [i for i in utils.listMergeGen(*interSetsGen)]
        assert (np.array(listData).T == answer).all()


#%% For movingaverage
class TestClass_movingaverage:
    def test_MA_1(self):
        result = utils.movingaverage([1, 1, 1, 1, 1], 3)
        correct_result = np.array([0.66666667, 1, 1, 1, 0.66666667])
        np.testing.assert_array_almost_equal(result, correct_result)

    def test_MA_2(self):
        result = utils.movingaverage([1, 1, 1, 1, 1, 1, 1, 1], 4)
        correct_result = np.array([0.5, 0.75, 1., 1., 1., 1., 1., 0.75])
        np.testing.assert_array_equal(result, correct_result)

    def test_MA_edge(self):
        result = utils.movingaverage([1, 1, 1, 1, 1], 3, edgeCorrection=True)
        correct_result = np.array([1., 1., 1., 1., 1.])
        np.testing.assert_array_equal(result, correct_result)

    def test_MA_edge2(self):
        result = utils.movingaverage([1, 2, 3, 4, 5], 3, edgeCorrection=True)
        correct_result = np.array([1.5, 2., 3., 4., 4.5])
        np.testing.assert_array_almost_equal(result, correct_result)

    def test_MA_edge3(self):
        result = utils.movingaverage([1, 1, 1, 1, 1, 1, 1, 1], 4, edgeCorrection=True)
        correct_result = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        np.testing.assert_array_equal(result, correct_result)

    def test_MA_edge4(self):
        result = utils.movingaverage([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 7, edgeCorrection=True)
        correct_result = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, correct_result)


#%% For errorResp
class TestClass_errorResp:
    def test_ER_zeroDiv(self):
        try:
            a = 1 / 0.0
        except:
            result = utils.errorResp()
        correct_result = '''A <class 'ZeroDivisionError'> : "float division by zero" in'''
        assert result[:59] == correct_result

    def test_ER_Name(self):
        try:
            a = b()
        except:
            result = utils.errorResp()
        correct_result = '''A <class 'NameError'> : "name 'b' is not defined" in'''
        assert result[:52] == correct_result


#%% For kendalwts
class TestClass_kendalwts:
    def test_wts_Random(self):
        data = np.array([[2., 0., 5., 1.],
                         [3., 3., 3., 4.],
                         [1., 5., 3., 5.],
                         [1., 1., 4., 2.],
                         [2., 4., 5., 1.],
                         [1., 0., 0., 2.]])
        assert utils.kendalwts(data) == 0.24615384615384617

    def test_wtsUniform(self):
        data = np.array([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3],
                         [4, 4, 4, 4],
                         [5, 5, 5, 5],
                         [6, 6, 6, 6]])
        assert utils.kendalwts(data) == 1


#%% For kldivergence
class TestClass_kldivergence:
    def test_klNormal(self):
        m0 = np.array([1, 1])
        m1 = np.array([1, 1])
        c0 = np.array([[0, 1], [1, 0]])
        c1 = np.array([[0, 1], [1, 0]])
        assert utils.kldivergence(m0, m1, c0, c1) == 0.0

    # TODO : Create more KL tests


if __name__ == '__main__':
    pytest.main()
