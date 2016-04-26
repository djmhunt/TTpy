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
sys.path.append("../")  # So code can be found from the main folder

# Other used function
from numpy import array, concatenate, arange, ones

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.weather import Weather, weatherStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decEtaSets, decSingle
from model.decision.discrete import decMaxProb
# The model
from model.qLearn import qLearn

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Weather, expParams, expExtraParams))

eta = 0.0
alpha = 0.5
alphaBounds = (0, 1)
beta = 0.5
betaBounds = (0, 10)
numActions = 2
numStimuli = 4
probActions = True

parameters = {'alpha': sum(alphaBounds)/2,
              'beta': sum(betaBounds)/2}
paramExtras = {'eta': eta,
               'numActions': numActions,
               'numStimuli': numStimuli,
               'stimFunc': weatherStimDirect(),
               'decFunc': decEta(eta=eta)}

modelSet = models((qLearn, parameters, paramExtras))

outputOptions = {'simLabel': 'qLearn_weatherSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

bounds = {'alpha': alphaBounds,
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
dataSet = data("../../Other peoples work/weather task/", 'xlsx', splitBy="subno")

# Add the stimulus. Not in the original data set and tidy up response and outcome to be in [0,1]
for i in xrange(len(dataSet)):
    d = dataSet[i]
    s = concatenate(([d['cue1']], [d['cue2']], [d['cue3']], [d['cue4']]))
    dataSet[i]["stimulus"] = s.T + 1

    dataSet[i]["response"] = array(dataSet[i]["response"]) - 1

    dataSet[i]["outcome"] = array(dataSet[i]["outcome"]) - 1

def scaleFuncSingle():
    """A blank scaling function"""
    def scaleFunc(x):
        return x

    scaleFunc.Name = "retVal"
    return scaleFunc

# Define the fitting algorithm
#fitAlg = minimize(fitQualFunc = "-2log",
#                  method = 'constrained', #'unconstrained',
#                  bounds = bounds,
#                  boundCostFunc = scalarBound(base = 160),
#                  numStartPoints = 5,
#                  boundFit = True)
fitAlg = evolutionary(fitQualFunc="-2log",
#                      strategy="all",
                      boundCostFunc=scalarBound(base=140),
#                      polish=False,
                      bounds=bounds)
#fitAlg = leastsq(dataShaper = "-2log")

# Set up the fitter
fit = fitter('response',
             'outcome',
             'ActionProb',
             fitAlg,
             scaleFuncSingle(),
             stimuliParams=["stimulus"],
             actChoiceParams=[0, 1],
             fpRespVal=1/1e100
             )

# Run the data fitter
dataFitting(expSets, modelSet, output, data=dataSet, fitter=fit)