# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each successful investigation and storing it
with the data.
"""
#%% Import useful functions
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
import os
filePath = os.path.realpath(__file__)
codePath = filePath.split('\\')[:-2]
sys.path.append("/".join(codePath))  # So code can be found from the main folder

# Other used function
from numpy import ones

#%% Import all experiments, models and fitting functions
# The experiments and stimulus processors
from experiment.probSelect import probSelectStimDirect, probSelectRewDirect

# The model factory
from modelGenerator import ModelGen
# The decision methods
from model.decision.discrete import weightProb
# The model
from model.qLearn import QLearn

#%% For importing the data
from data import data

#For data fitting
from dataFitting import dataFitting
from fitAlgs.fitSims import FitSim
from fitAlgs.evolutionary import Evolutionary

#%% Set the model sets
alphaBounds = (0, 1)
betaBounds = (0, 30)
bounds = {'alpha': alphaBounds,
          'beta': betaBounds}

numActions = 6
numCues = 1

modelParameters = {'alpha': sum(alphaBounds)/2,
                   'beta': sum(betaBounds)/2}
modelStaticArgs = {'numActions': numActions,
                   'numCues': numCues,
                   'actionCodes': {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
                   'expect': ones((numActions, numCues)) / 2,
                   'prior': ones(numActions) / numActions,
                   'stimFunc': probSelectStimDirect(),
                   'rewFunc': probSelectRewDirect(),
                   'decFunc': weightProb(["A", "B", "C", "D", "E", "F"])}

modelSet = ModelGen(QLearn, modelParameters, modelStaticArgs)



#%% Import data
dat = data("./Outputs/qLearn_probSelectSimSet_2019-10-29/Pickle/", 'pkl', validFiles=["qLearn_modelData_sim-"])

for d in dat:
    d["validActions"] = d["ValidActions"].T

#%% Set up the fitting
modSim = FitSim('Decisions',
                'Rewards',
                'ActionProb',
                fitSubset=float('Nan'),  # float('Nan'), None, range(0,40)
                #stimuliParams=["stimCues"],
                actChoiceParams='validActions'
                )

# Define the fitting algorithm
fitAlg = Evolutionary(modSim,
                      fitQualityFunc="BIC2norm",
                      qualityFuncArgs={"numParams": len(modelParameters), "numActions": numActions, "qualityThreshold": 20},
                      # strategy="all",
                      boundCostFunc=None,  # scalarBound(base=160),
                      polish=False,
                      bounds=bounds,
                      extraFitMeasures={"-2log": {},
                                        "BIC": {"numParams": len(modelParameters)},
                                        "r2": {"numParams": len(modelParameters), "randActProb": 1/numActions},
                                        "bayesFactor": {"numParams": len(modelParameters), "randActProb": 1/numActions}})

#%% Run the data fitter
dataFitting(modelSet,
            dat,
            fitAlg,
            partLabel='simID',
            simLabel='qLearn_probSelect_fromSim',
            save=True,
            saveFittingProgress=True,
            saveScript=True,
            pickleData=True,
            npSetErr='log')  # 'raise','log'
