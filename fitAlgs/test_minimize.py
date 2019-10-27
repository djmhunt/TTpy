# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
sys.path.append("../")

import pytest

from numpy import array

from models import models
from model.qLearn import qLearn
from fitAlgs.fitSims import fitSim
from fitAlgs.minimize import minimize

#def test_an_exception():
#    with raises(IndexError):
#        # Indexing the 30th item in a 3 item list
#        [5, 10, 15][30]
#        


@pytest.fixture(scope="function")
def modelSets():
    
    beta = 0
    alpha = 0.2
    theta = 1.5
    parameters = {'alpha': alpha,
                  'beta': beta,
                  'theta': theta}
    paramExtras = {'prior':array([0.5, 0.5])}
    
    modelSet = models((qLearn,parameters,paramExtras))
    modelInfos = [m for m in modelSet.iterFitting()]
    modelInfo = modelInfos[0]
    model = modelInfo[0]
    modelSetup = modelInfo[1:]
    
    return model, modelSetup


@pytest.fixture(scope="module")
def participant():
    
    data = {'subchoice': array([2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 
                                2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2,
                                2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1,
                                2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1]),
             'subreward': array([7,  2,  2, 10,  5, 10,  6,  1,  6, 10, 10, 10,  1,
                                 8,  4,  8, 10, 4,  9, 10,  8,  2,  1,  1,  3,  2,  
                                 6,  2,  8,  6,  1,  6, 10, 10, 10,  4,  7, 10,  5,  
                                 2,  1,  1,  5,  8,  5, 10,  4, 10, 10,  9,  2,  9,  
                                 8, 10,  7,  7,  1, 10, 10,  8,  3, 10,  2, 10,  7, 
                                 10,  8,  3,  6,  4,  4,  9, 10,  3,  7,  2,  6,  3,  
                                 1,  5]), 
             'cumpts': array([7,   9,  11,  21,  26,  36,  42,  43,  49,  59,  69,
                              79,  80,  88,  92, 100, 110, 114, 123, 133, 141, 143, 
                              144, 145, 148, 150, 156, 158, 166, 172, 173, 179, 189, 
                              199, 209, 213, 220, 230, 235, 237, 238, 239, 244, 252, 
                              257, 267, 271, 281, 291, 300, 302, 311, 319, 329, 336, 
                              343, 344, 354, 364, 372, 375, 385, 387, 397, 404, 414, 
                              422, 425, 431, 435, 439, 448, 458, 461, 468, 470, 476, 
                              479, 480, 485])}
    
#    data = {'subchoice': array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
#                                2, 2, 2, 2, 2, 2, 2,
#       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
#       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#       1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2]), 
#            'subreward': array([ 7, 10,  5, 10,  6,  6, 10, 10, 10,  8,  4,  
#                                 8, 10,  4,  9, 10,  8,
#        6, 10, 10, 10,  4,  7, 10,  5, 10,  4, 10, 10,  9,  2,  9,  8, 10,
#        7,  7,  1, 10,  2,  6,  4,  7,  2,  1,  6,  9,  4,  4,  4, 10,  6,
#        4, 10,  3, 10,  5, 10,  3, 10, 10,  5,  4,  6, 10,  7,  7, 10, 10,
#       10,  3,  1,  4,  1,  3,  1,  1,  1,  1,  1,  1])}
                              
    return data


@pytest.fixture(scope="function")
def fitting():
    
    def scaleFunc(x):
        return x - 1
        
    fitAlg = minimize(fitQualFunc="-2log", method='constrained', bounds={'alpha': (0, 1),'theta': (0, 5)})

    fitFunc = fitSim('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFunc)
    
    return fitFunc, fitAlg
    

class TestClass: 
              
    def test_startParamVals(self):
        
        fitAlg = minimize()
        
        ans1 = fitAlg.startParamVals(0.5)
        assert (abs(ans1 - [0.25, 0.5, 0.75]) < 0.01).all()
        
        ans2 = fitAlg.startParamVals(0.5, numPoints=4)
        assert (abs(ans2 - [0.2, 0.4, 0.6, 0.8]) < 0.01).all()
        
        ans3 = fitAlg.startParamVals(0.5, bMax=0.7, numPoints=4)
        assert (abs(ans3 - [0.38, 0.46, 0.54, 0.62]) < 0.01).all()
        
        ans4 = fitAlg.startParamVals(0.7, numPoints=4)
        assert (abs(ans4 - [0.52, 0.64, 0.76, 0.88]) < 0.01).all()
        
        ans5 = fitAlg.startParamVals(1.2, numPoints=4)
        assert (abs(ans5 - [0.48, 0.96, 1.44, 1.92]) < 0.01).all()
            
    def test_startParams(self):
        
        fitAlg = minimize()
        
        fitAlg.bounds = None
        
        starts1 = fitAlg.startParams([0.5,1.2], numPoints = 3)
        
        ans1 = array([[ 0.25,  0.6 ], [ 0.5 ,  0.6 ], [ 0.75,  0.6 ], 
                     [ 0.25,  1.2 ], [ 0.5 ,  1.2 ], [ 0.75,  1.2 ],
                     [ 0.25,  1.8 ], [ 0.5 ,  1.8 ], [ 0.75,  1.8 ]])
        
        assert (abs(starts1 - ans1) < 0.01).all()
                                
        fitAlg.bounds = [[0,0.7],[0,5]]
        
        starts2 = fitAlg.startParams([0.5,1.2], numPoints = 3)
        
        ans2 = array([[ 0.4,  0.6], [ 0.5,  0.6], [ 0.6,  0.6], [ 0.4,  1.2],
                      [ 0.5,  1.2], [ 0.6,  1.2], [ 0.4,  1.8], [ 0.5,  1.8],
                      [ 0.6,  1.8]])
        
        assert (abs(starts2 - ans2) < 0.01).all()
        
    def test_logprob(self):
        
        def sim(*params):
            
            a = [0.5, 1, 5, 10]
            
            return a
            
        fitAlg = minimize(fitQualFunc="-2log")
        
        fitAlg.sim = sim
        
        assert abs(fitAlg.fitness([]) - -9.28771238) < 0.1

    def test_methodFit(self, fitting, participant, modelSets):
        
        model, modelSetup = modelSets
        fitFunc, fitAlg = fitting
        
        fitFunc.model = model
        fitFunc.mInitialParams = modelSetup[0].values()
        fitFunc.mParamNames = modelSetup[0].keys()
        fitFunc.mOtherParams = modelSetup[1]

        fitFunc.partChoices = fitFunc.scaler(participant[fitFunc.partChoiceParam])

        fitFunc.partRewards = participant[fitFunc.partRewardParam]
        
        fitAlg.sim = fitFunc.fitness
        
        initParamSets = fitAlg.startParams([0.5,0.5], numPoints=30)
        
        result = fitAlg._methodFit(fitAlg.methodSet[0], initParamSets, fitAlg.bounds)
        
        pytest.set_trace()
        
        assert (abs(result.x - array([0.072, 1.358])) < 0.1).all()


if __name__ == '__main__':
    pytest.main()
    
#    pytest.set_trace()