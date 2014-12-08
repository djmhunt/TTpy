# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

import pytest

from utils import listMerGen

from numpy import array
from pytest import raises, fixture

def test_an_exception():
    with raises(IndexError):
        # Indexing the 30th item in a 3 item list
        [5, 10, 15][30]

@fixture(scope="module")  
def listMerTestData():
    
    v = 3
    interSets = [[i for i in range(a-v,a)] for a in range(1,v)]
    
    answer = array([[-2, -1,  0, -2, -1,  0, -2, -1,  0],
                    [-1, -1, -1,  0,  0,  0,  1,  1,  1]])
    
    return interSets, answer

class TestClass: 
            
    def test_listMerGen(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        l = [i for i in listMerGen(*interSets)]
        assert (array(l).T == answer).all()
        
    def test_listMerGenArray(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For numpy Arrays                    
        interSetsTuple = [array(i) for i in interSets] 
        l = [i for i in listMerGen(*interSetsTuple)]
        assert (array(l).T == answer).all()
        
    def test_listMerGenTuple(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a tuple                    
        interSetsTuple = tuple(interSets) 
        l = [i for i in listMerGen(*interSetsTuple)]
        assert (array(l).T == answer).all()
        
    def test_listMerGenGen(self,listMerTestData):
        
        (interSets, answer) = listMerTestData
        # For a generator    
        interSetsGen = (i for i in interSets)
        l = [i for i in listMerGen(*interSetsGen)]
        assert (array(l).T == answer).all() 
    
if __name__ == '__main__':
    pytest.main()