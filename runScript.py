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

# Other used function
from numpy import array, concatenate

### Import all experiments, models, outputting and interface functions
#The experiment factory
from experiments import experiments
#The experiments and stimulus processors
from experiment.decks import Decks, deckStimDualInfo, deckStimDirect
from experiment.beads import Beads, beadStimDirect, beadStimDualDirect, beadStimDualInfo
from experiment.pavlov import Pavlov, pavlovStimTemporal

# The model factory
from models import models 
# The decision methods
from model.decision.binary import decBeta
#The models
from model.BP import BP
from model.EP import EP
from model.MS import MS
from model.MS_rev import MS_rev
from model.qLearn import qLearn
from model.RVPM import RVPM

from outputting import outputting

### Set the outputting, model sets and experiment sets
beta = 0.0
alpha = 0.5
gamma = 0.5
simDur = 30
outputOptions = {'simLabel': 'qLearn_jessData',
                 'save': True,
                 'silent': False}
parameters = {  'alpha':alpha,
#                'beta':beta,
                'gamma':gamma}
paramExtras = {'beta':beta,
               'stimFunc':deckStimDirect(),
               'decFunc':decBeta(beta = beta)} #For qLearn decks
#paramExtras = {'activity':array([5,5])} # For EP

expSets = experiments((Decks,{},{}))
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

#from fitting.expfitter import fitter #Not sure this will ever be used, but I want to keep it here for now
from fitting.fitness import fitter
from fitting.fitters.leastsq import leastsq
from fitting.fitters.minimize import minimize

# Import data
jessData = data("../Shared folders/worthy models and data/jessdata/",'mat')

# Add the reward for card. Not in the original data set
for i in xrange(len(jessData)):
    partCumRewards = jessData[i]["cumpts"]
    jessData[i]["subreward"] = concatenate((partCumRewards[0:1],partCumRewards[1:]-partCumRewards[:-1]))

# Create a scaling function to match up the actions understood by the model and
# those taken by the participant
def scaleFuncSingle():
    def scaleFunc(x):
        return x - 1
        
    scaleFunc.Name = "subOne"
    return scaleFunc
# Another way of setting up a scaler    
scaler = lambda x : x - 1

# Define the fitting algorithm
#fitAlg = minimize(dataShaper = "-2log", method = 'unconstrained')
fitAlg = minimize(fitQualFunc = "-2log", method = 'constrained', bounds= [(0,1),(0,5)])
#fitAlg = leastsq(dataShaper = "-2log")

# Set up the fitter
fit = fitter('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFuncSingle())

# Run the data fitter
dataFitting(expSets, modelSet, output, data = jessData, fitter = fit)