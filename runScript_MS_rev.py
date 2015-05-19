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
#The model
from model.MS_rev import MS_rev

from outputting import outputting

### Set the outputting, model sets and experiment sets
beta = 0#0.3#0.15
alpha = 0.5#0.2#0.5#0.2
gamma = 0.5#0.7#0.5#0.7
simDur = 30
outputOptions = {'simLabel': 'MS_rev_decksSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
parameters = {  'alpha':alpha,
                'gamma':gamma}
paramExtras = {'beta':beta,
               'stimFunc':deckStimDualInfo(10),
               'decFunc':decBeta(beta = beta)} #For decks

expSets = experiments((Decks,{},{}))
modelSet = models((MS_rev,parameters,paramExtras))
output = outputting(**outputOptions)

### For simulating experiments
#
#from simulation import simulation
#
#simulation(expSets, modelSet, output)

### For data fitting

from numpy import concatenate

from dataFitting import dataFitting

from data import data, datasets

#from fitting.expfitter import fitter #Not sure this will ever be used, but I want to keep it here for now
from fitting.fitness import fitter
from fitting.fitters.leastsq import leastsq
from fitting.fitters.minimize import minimize

# Import data
dataFolders = ["../Shared folders/worthy models and data/jessdata/", 
               "../Shared folders/worthy models and data/carlosdata/", 
               "../Shared folders/worthy models and data/iant_studentdata/"]
				

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
                  bounds = {'alpha' : (0,1),
                            'gamma' : (0,5)}, 
                  numStartPoints = 5,
                  boundFit = True)
#fitAlg = leastsq(dataShaper = "-2log")

# Set up the fitter
fit = fitter('subchoice', 'subreward', 'ActionProb', fitAlg, scaleFuncSingle())

# Run the data fitter
dataFitting(expSets, modelSet, output, data = dataSet, fitter = fit)