# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

This is a script with all the components for running an investigation. I would 
recommend making a copy of this for each sucessful investigation and storing it
 with the data.
"""
### Import useful functions
# Make devision floating point by default
from __future__ import division

from numpy import array, concatenate

### Import all experiments, models, outputting and interface functions
from experiments import experiments
from experiment.decks import decks
from experiment.beads import beads
from experiment.pavlov import pavlov, pavlovStimTemporal

from models import models
from model.BP import BP
from model.EP import EP
from model.MS import MS
from model.MS_rev import MS_rev
from model.qLearn import qLearn
from model.RVPM import RVPM

from outputting import outputting

### Set the outputting, model sets and experiment sets
beta = 0.3#0#0.15
alpha = 0.5#0.2#0.5#0.2
theta = 0.5#0.7#0.5#0.7
simDur = 30
outputOptions = {'simLabel': 'qLearn_jessData',
                 'save': True,
                 'silent': False}
parameters = {  'alpha':alpha,
#                'beta':beta,
                'theta':theta}
paramExtras = {'prior':array([0.5,0.5])} #For qLearn
#paramExtras = {'activity':array([5,5])} # For EP

expSets = experiments((decks,{},{}))
modelSet = models((qLearn,parameters,paramExtras))
output = outputting(**outputOptions)

### For simulating experiments
#
#from simulation import simulation
#
#simulation(expSets, modelSet, output)

### For data fitting

from dataFitting import dataFitting

from data import data

#from fitting.expfitter import fitter #Not sure this will ever be used, but I want to 
from fitting.fitness import fitter
from fitting.fitters.leastsq import leastsq
from fitting.fitters.minimize import minimize

# Import data
jessData = data("../Shared folders/worthy models and data/jessdata/",'mat')

# Add the reward for card. Not in the original data set
for i in xrange(len(jessData)):
    partCumRewards = jessData[i]["cumpts"]
    jessData[i]["subreward"] = concatenate((partCumRewards[0:1],partCumRewards[1:]-partCumRewards[:-1]))

def scaleFuncSingle():
    def scaleFunc(x):
        return x - 1
        
    scaleFunc.Name = "subOne"
    return scaleFunc
    
scaler = lambda x : x - 1

#fitAlg = minimize(dataShaper = "-2log", method = 'unconstrained')
fitAlg = minimize(dataShaper = "-2log", method = 'constrained', bounds= [(0,1),(0,5)])
#fitAlg = leastsq(dataShaper = "-2log")
fit = fitter('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFuncSingle())

dataFitting(expSets, modelSet, output, data = jessData, fitter = fit)