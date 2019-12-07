# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest

import numpy as np

import utils


def test_an_exception():
    with pytest.raises(IndexError):
        # Indexing the 30th item in a 3 item list
        [5, 10, 15][30]


@pytest.fixture(scope="module")
def listMerTestData():
    
    v = 3  # type: int
    interSets = [[i for i in range(a-v, a)] for a in range(1, v)]
    
    answer = np.array([[-2, -1,  0, -2, -1,  0, -2, -1,  0],
                       [-1, -1, -1,  0,  0,  0,  1,  1,  1]])
    
    return interSets, answer

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

#%% For kendalwts
class TestClass_kendalwts:
    def test_wtsRandom(self):
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
