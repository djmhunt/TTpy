# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

This is a script with all the components for running an investigation. I would
recommend making a copy of this for each sucessful investigation and storing it
 with the data.

Desired plots:
    :math:`\\alpha_N = 0.1, \\alpha_G \in ]0,0.2[`
    :math:`\\beta_N = 1, \\beta_G in ]0,2[`

    Plot Positive vs negative choice bias against :math:`prob(R|A) \in ]0.5,1[`
    with:
        :math:`\\alpha_G=\\alpha_N`, varying :math:`\\beta_G` relative to :math:`\\beta_N`
        :math:`\\beta_G=\\beta_N`, varying :math:`\\alpha_G` relative to :math:`\\alpha_N`

    Plot the range of
    :math:`\\alpha_G = 0.2 - \\alpha_N` for :math:`\\alpha_N \in ]0,0.2[` and
    :math:`\\beta_G = 2 - \\beta_N` for :math:`\\beta_N in ]0,2[` with the Y-axis being
    Choose(A) = prob(A) - prob(M),
    Avoid(B) = prob(M) - prob(B),
    Bias = choose(A) - avoid(B),
"""
### Import useful functions
# Make devision floating point by default
from __future__ import division

import sys
sys.path.append("../") #So code can be found from the main folder

# Other used function
from numpy import array, concatenate, arange

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.decks import Decks, deckStimDualInfo, deckStimDirect
from experiment.beads import Beads, beadStimDirect, beadStimDualDirect, beadStimDualInfo
from experiment.pavlov import Pavlov, pavlovStimTemporal
from experiment.probSelect import probSelect, probSelectStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta
from model.decision.discrete import decMaxProb
#The model
from model.OpAL import OpAL

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks,expParams,expExtraParams))

sR = arange(-0.1,0.35,0.05)

rhoSet = arange(-1,1.5,0.5)

parameters = {'alphaGo':alphaGoDiffSet,
              'alphaC': 0.1,
              'betaDiff':rhoSet}

paramExtras = {'alpha': 0.1,
#               'alphaGoDiff':0,
               'beta': 1,
               'betaDiff':0,
               'numActions':4,
               'stimFunc':deckStimDirect(),
               'decFunc':decMaxProb([0,1,2,3])} #For decks
modelSet = models((OpAL,parameters,paramExtras))

outputOptions = {'simLabel': 'OpAL_simSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

bounds = {'alphaC' : alphaCBounds,
          'beta' : betaBounds}

### For data fitting

from numpy import concatenate

from dataFitting import dataFitting

from data import data, datasets

#from fitting.expfitter import fitter #Not sure this will ever be used, but I want to keep it here for now
from fitting.actReactFitter import fitter
from fitting.fitters.leastsq import leastsq
from fitting.fitters.minimize import minimize

# Import data
dataFolders = ["../../Shared folders/worthy models and data/jessdata/",
               "../../Shared folders/worthy models and data/carlosdata/",
               "../../Shared folders/worthy models and data/iant_studentdata/"]


dataSet = datasets(dataFolders,['mat']*len(dataFolders))#"./testData/",'mat')#

# Add the reward for card. Not in the original data set
for i in xrange(len(dataSet)):
    partCumRewards = dataSet[i]["cumpts"]
    dataSet[i]["subreward"] = concatenate((partCumRewards[0:1],partCumRewards[1:]-partCumRewards[:-1]))

# Create a scaling function to match up the actions understood by the model and
# those taken by the participant
def scaleFuncSingle():
    def scaleFunc(x):
        return x - 1

    scaleFunc.Name = "subOne"
    return scaleFunc

# Define the fitting algorithm
fitAlg = minimize(fitQualFunc = "-2log",
                  method = 'constrained', #'unconstrained',
                  bounds = bounds,
                  numStartPoints = 5,
                  boundFit = True)
#fitAlg = leastsq(dataShaper = "-2log")

# Set up the fitter
fit = fitter('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFuncSingle())

# Run the data fitter
dataFitting(expSets, modelSet, output, data = dataSet, fitter = fit)