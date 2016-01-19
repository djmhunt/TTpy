# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each sucessful investigation and storing it
 with the data.
"""
### Import useful functions
# Make division floating point by default
from __future__ import division

import sys
sys.path.append("../") #So code can be found from the main folder

# Other used function
from numpy import array, concatenate, ones

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.decks import Decks, deckStimDualInfo, deckStimDualInfoLogistic, deckStimDirect, deckStimDirectNormal

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decEtaSets, decSingle
from model.decision.discrete import decMaxProb
# The model
from model.BPMS import BPMS

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks, expParams, expExtraParams))

eta = 0
beta = 0.5
betaBounds = (0, 10)
delta = 0.5
deltaBounds = (0.001, 0.999)
numCritics = 2

parameters = {'delta': sum(deltaBounds)/2,
              'beta': sum(betaBounds)/2}
paramExtras = {'eta': eta,
               'numCritics': numCritics,
               'stimFunc': deckStimDualInfo(10, 0.01),
               'decFunc': decSingle()}

modelSet = models((BPMS, parameters, paramExtras))

outputOptions = {'simLabel': 'BPMS_decksSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp': 'log'}  # 'raise','log'
output = outputting(**outputOptions)

bounds = {'delta': deltaBounds,
          'beta': betaBounds}
### For data fitting

from numpy import concatenate

from dataFitting import dataFitting

from data import data, datasets

from fitting.fitters.boundFunc import infBound, scalarBound

# from fitting.expfitter import fitter #Not sure this will ever be used, but I want to keep it here for now
from fitting.actReactFitter import fitter
from fitting.fitters.leastsq import leastsq
from fitting.fitters.minimize import minimize
from fitting.fitters.basinhopping import basinhopping
from fitting.fitters.evolutionary import evolutionary

# Import data
dataFolders = ["../../Shared folders/worthy models and data/jessdata/",
               "../../Shared folders/worthy models and data/carlosdata/",
               "../../Shared folders/worthy models and data/iant_studentdata/"]

dataSet = datasets(dataFolders, ['mat'] * len(dataFolders))  # "./testData/",'mat')#

# Add the reward for card. Not in the original data set
for i in xrange(len(dataSet)):
    partCumRewards = dataSet[i]["cumpts"]
    dataSet[i]["subreward"] = concatenate((partCumRewards[0:1], partCumRewards[1:] - partCumRewards[:-1]))


# Create a scaling function to match up the actions understood by the model and
# those taken by the participant
def scaleFuncSingle():
    def scaleFunc(x):
        return x - 1

    scaleFunc.Name = "subOne"
    return scaleFunc

# Define the fitting algorithm
# fitAlg = minimize(fitQualFunc = "-2log",
#                  method = 'constrained', #'unconstrained',
#                  bounds = bounds,
#                  boundCostFunc = scalarBound(base = 160),
#                  numStartPoints = 5,
#                  boundFit = True)
fitAlg = evolutionary(fitQualFunc="-2log",
                      bounds=bounds,
                      boundCostFunc=scalarBound(base=160))
#                      polish=False)
# fitAlg = leastsq(dataShaper = "-2log")

# Set up the fitter
fit = fitter('subchoice',
             'subreward',
             'ActionProb',
             fitAlg,
             scaleFuncSingle(),
             actChoiceParams=[0, 1])

# Run the data fitter
dataFitting(expSets, modelSet, output, data=dataSet, fitter=fit)
