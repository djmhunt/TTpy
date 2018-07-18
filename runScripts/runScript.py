# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each successful investigation and storing it
with the data.
"""
### Import useful functions
from __future__ import division, print_function, unicode_literals, absolute_import

import sys

sys.path.append("../")  # So code can be found from the main folder

# Other used function
from numpy import ones

### Import all experiments, models, outputting and interface functions
# The experiments and stimulus processors
from experiment.probSelect import probSelectStimDirect, probSelectRewDirect

# The model factory
from models import models
# The decision methods
#from model.decision.binary import decEta, decEtaSets, decSingle, decRandom
from model.decision.discrete import decWeightProb
# The model
from model.qLearn import qLearn

from outputting import outputting

### Set the outputting, model sets and experiment sets
alpha = 0.5
alphaBounds = (0, 1)
beta = 0.5
betaBounds = (0, 30)
numActions = 6
numCues = 1

parameters = {'alpha': sum(alphaBounds)/2,
              'beta': sum(betaBounds)/2}
paramExtras = {'numActions': numActions,
               'numCues': numCues,
               'actionCodes': {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
               'expect': ones((numActions, numCues)) / 2,
               'stimFunc': probSelectStimDirect(),
               'rewFunc': probSelectRewDirect(),
               'decFunc': decWeightProb(["A", "B", "C", "D", "E", "F"])}

modelSet = models((qLearn, parameters, paramExtras))

outputOptions = {'simLabel': 'qLearn_probSelect_fromSim',
                 'save': True,
                 'saveScript': True,
                 'pickleData': True,
                 'simRun': False,
                 'saveFittingProgress': True,
                 'saveOneFile': False,
                 'saveFigures': False,
                 'silent': True,
                 'npErrResp': 'log'}  # 'raise','log'
output = outputting(**outputOptions)

bounds = {'alpha': alphaBounds,
          'beta': betaBounds}

### For data fitting

from dataFitting import dataFitting

from data import data

#from fitting.fitAlgs.boundFunc import infBound, scalarBound

from fitting.actReactFitter import fitter
from fitting.fitAlgs.evolutionary import evolutionary

# Import data
dat = data("./Outputs/qLearn_probSelectSimSet_2018-4-19/Pickle/", 'pkl', validFiles=["qLearn_modelData_sim-"])

for d in dat:
    d["validActions"] = d["ValidActions"].T

# Create a scaling function to match up the actions understood by the model and
# those taken by the participant
def scaleFuncSingle():
    def scaleFunc(x):
        return x

    scaleFunc.Name = "Repeat"
    return scaleFunc


# Define the fitting algorithm
fitAlg = evolutionary(fitQualFunc="BIC2",
                      qualFuncArgs={"numParams": len(parameters),
                                    "numActions": numActions,
                                    "randActProb": 1/2},
                      bounds=bounds,
                      boundCostFunc=None,  # scalarBound(base=140),
                      tolerance=0.01,
                      polish=False)

# Set up the fitter
fit = fitter('Decisions',
             'Rewards',
             'ActionProb',
             fitAlg,
             scaleFuncSingle(),
             fitSubset=float('Nan'),  # float('Nan'), None, range(0,40)
             #stimuliParams=["stimCues"],
             actChoiceParams='validActions')

# Run the data fitter
dataFitting(modelSet, output, data=dat, fitter=fit, partLabel='simID')