# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest

from utils import listMergeGen

from numpy import array

def test_an_exception():
    with raises(IndexError):
        # Indexing the 30th item in a 3 item list
        [5, 10, 15][30]

@pytest.fixture(scope="module")
def listMerTestData():
    
    v = 3
    interSets = [[i for i in range(a-v,a)] for a in range(1,v)]
    
    answer = array([[-2, -1,  0, -2, -1,  0, -2, -1,  0],
                    [-1, -1, -1,  0,  0,  0,  1,  1,  1]])
    
    return interSets, answer

class TestClass: 
            
    def test_listMergeGen(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        l = [i for i in listMergeGen(*interSets)]
        assert (array(l).T == answer).all()
        
    def test_listMergeGenArray(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For numpy Arrays                    
        interSetsTuple = [array(i) for i in interSets] 
        l = [i for i in listMergeGen(*interSetsTuple)]
        assert (array(l).T == answer).all()
        
    def test_listMergeGenTuple(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a tuple                    
        interSetsTuple = tuple(interSets) 
        l = [i for i in listMergeGen(*interSetsTuple)]
        assert (array(l).T == answer).all()
        
    def test_listMergeGenGen(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a generator    
        interSetsGen = (i for i in interSets)
        l = [i for i in listMergeGen(*interSetsGen)]
        assert (array(l).T == answer).all() 
    
if __name__ == '__main__':
    pytest.main()