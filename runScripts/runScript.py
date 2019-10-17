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
import os
filePath = os.path.realpath(__file__)
codePath = filePath.split('\\')[:-2]
sys.path.append("/".join(codePath))  # So code can be found from the main folder

# Other used function
from numpy import ones

### Import all experiments, models, outputting and interface functions
# The experiments and stimulus processors
from experiment.probSelect import probSelect, probSelectStimDirect, probSelectRewDirect

# The model factory
from models import models
# The decision methods
from model.decision.discrete import decWeightProb
# The model
from model.qLearn import qLearn

from outputting import outputting

### Set the outputting and model sets
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
               'prior': ones(numActions) / numActions,
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

from fitAlgs import infBound, scalarBound

from fitAlgs.simMethods.actReactFitter import fitter
from fitAlgsevolutionary import evolutionary

# Import data
dat = data("./Outputs/qLearn_probSelectSimSet_2018-4-19/Pickle/", 'pkl', validFiles=["qLearn_modelData_sim-"])

for d in dat:
    d["validActions"] = d["ValidActions"].T

# Set up the model simulation
modSim = fitter('Decisions',
                'Rewards',
                'ActionProb',
                fitSubset=float('Nan'),  # float('Nan'), None, range(0,40)
                #stimuliParams=["stimCues"],
                actChoiceParams='validActions'
                )

# Define the fitting algorithm
fitAlg = evolutionary(modSim,
                      fitQualFunc="BIC2norm",
                      qualFuncArgs={"numParams": len(parameters), "numActions": numActions, "qualityThreshold": 20},
                      # strategy="all",
                      boundCostFunc=None, # scalarBound(base=160),
                      polish=False,
                      bounds=bounds,
                      extraFitMeasures={"-2log": {},
                                        "BIC": {"numParams": len(parameters)},
                                        "r2": {"numParams": len(parameters), "randActProb": 1/numActions},
                                        "bayesFactor": {"numParams": len(parameters), "randActProb": 1/numActions}})

# Run the data fitter
dataFitting(modelSet, output, data=dat, fitter=fitAlg, partLabel='simID')