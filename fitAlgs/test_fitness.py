# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
sys.path.append("../../")

import pytest

from numpy import array

from modelGenerator import ModelGen
from model.qLearn import QLearn
from fitAlgs.fitSims import fitSim
from fitAlgs.minimize import minimize

#def test_an_exception():
#    with raises(IndexError):
#        # Indexing the 30th item in a 3 item list
#        [5, 10, 15][30]
#        
      
@pytest.fixture(scope="function")
def modelSets():
    
    beta = 0.15
    alpha = 0.2
    theta = 1.5
    parameters = {  'alpha':alpha,
    #                'beta':beta,
                    'theta':theta}
    paramExtras = {'prior':array([0.5,0.5])}
    
    modelSet = ModelGen((QLearn, parameters, paramExtras))
    modelInfos = [m for m in modelSet.iterInitDetails()]
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
                              
    return data
    
@pytest.fixture(scope="module")
def probabilities():
    
    prob = array([5.000e-01,   3.543e-01,   1.824e-01,   9.020e-01,   9.734e-01,
                  9.610e-01,   9.877e-01,   1.577e-02,   9.934e-01,   9.919e-01,
                  9.971e-01,   9.987e-01,   6.491e-04,   9.997e-01,   9.997e-01,
                  9.988e-01,   9.990e-01,   9.995e-01,   9.985e-01,   9.991e-01,
                  9.996e-01,   4.335e-04,   3.341e-04,   2.009e-04,   1.338e-04,
                  1.760e-04,   1.625e-04,   5.056e-04,   3.778e-04,   9.982e-01,
                  3.320e-03,   1.427e-03,   9.968e-01,   9.984e-01,   9.991e-01,
                  9.994e-01,   9.976e-01,   9.969e-01,   9.985e-01,   3.886e-03,
                  2.265e-03,   1.089e-03,   6.058e-04,   1.257e-03,   5.524e-03,
                  9.926e-01,   9.965e-01,   9.885e-01,   9.950e-01,   9.974e-01,
                  9.980e-01,   9.865e-01,   9.923e-01,   9.934e-01,   9.968e-01,
                  9.956e-01,   9.944e-01,   4.040e-02,   1.919e-01,   4.866e-01,
                  6.115e-01,   6.549e-01,   8.503e-01,   4.466e-01,   7.331e-01,
                  7.484e-01,   8.863e-01,   9.025e-01,   2.970e-01,   2.932e-01,
                  8.168e-01,   6.691e-01,   8.280e-01,   9.286e-01,   2.208e-01,
                  2.989e-01,   8.835e-01,   8.678e-01,   2.959e-01,   8.914e-01])
                  
    return prob
                              
@pytest.fixture(scope="function")
def fitting():
    
    def scaleFunc(x):
        return x - 1
        
    fitAlg = minimize(fitQualFunc="-2log", method='constrained', bounds={'alpha': (0, 1), 'theta': (0, 40)})

    fitFunc = fitSim('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFunc)
    
    return fitFunc, fitAlg
    

class TestClass: 
    
    def test_complete(self, fitting, participant, modelSets):
        model, modelSetup = modelSets
        fitFunc, fitAlg = fitting
        
        modelFitted, fitQuality = fitFunc.participant(None, model, modelSetup, participant)
        params = modelFitted.params()
        
        assert abs(params['alpha'] - 0.038898802) < 0.01	
        assert abs(params['theta'] - 0.220813253) < 0.01
        
    def test_modelInputs(self, fitting, modelSets):
        
        fitFunc, fitAlg = fitting
        model, modelSetup = modelSets
        modelParams, modelOtherParams = modelSetup
        
        paramNames = ['alpha','theta']
        paramInputs = [modelParams[n] for n in paramNames]
        
        fitFunc.mParamNames = paramNames
        fitFunc.mOtherParams = modelOtherParams
        inputs = fitFunc._getModInput(*paramInputs)
        
        answer = {'alpha': 0.2, 'prior': array([ 0.5,  0.5]), 'theta': 1.5}
        
        for k,v in answer.iteritems():
            assert (abs(inputs[k] - v) < 0.01).all()
        
    def test_simSetup(self, fitting, participant, modelSets, probabilities):
        
        fitFunc, fitAlg = fitting
        model, modelSetup = modelSets
        modelParams, modelOtherParams = modelSetup
        
        paramNames = ['alpha','theta']
        paramInputs = [modelParams[n] for n in paramNames]
        
        fitFunc.model = model
        fitFunc.mParamNames = paramNames
        fitFunc.mOtherParams = modelOtherParams
        fitFunc.partChoices = fitFunc.scaler(participant['subchoice'])
        fitFunc.partRewards = participant['subreward']
        
        model = fitFunc._simSetup(*paramInputs)
        
        results = model.returnTaskState()

        probs = results["Probabilities"]
        act = results["Actions"]
        actProb = results['ActionProb']
        
        for i, a in enumerate(act):
            assert probs[i,a] == actProb[i]
            
        assert (abs(actProb - probabilities)<0.01).all()


if __name__ == '__main__':
    pytest.main()
    
#    pytest.set_trace()