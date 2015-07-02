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
# Make devision floating point by default
from __future__ import division

import sys
sys.path.append("../") #So code can be found from the main folder

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
from model.decision.binary import decEta
#The model
from model.RVPM import RVPM

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {'rewMag':4,
             'rewProb':array([0.87,0.33]),
             'stimMag':1,
             'stimDur':20,#200) # Stimulus duration
             'rewDur':4,#40) #duration of reward
             'simDur':30,#300) # the length of the simulation
             'stimRepeats':7}
expExtraParams = {}
expSets = experiments((Pavlov,expParams,expExtraParams))

eta = 0.0
alpha = 0.005
beta = 0.1
w = array([0.01,0.01])
zeta = 2
tau = 160
z = 100
averaging = 3

parameters = {  'alpha':alpha,
                'beta':beta}
paramExtras = {'eta':eta,
               'w':w,
               'zeta':zeta,
               'tau':tau,
               'z':z,
               'averaging':averaging,
               'stimFunc':pavlovStimTemporal(),
               'decFunc':decEta(eta = eta)}

modelSet = models((RVPM,parameters,paramExtras))

outputOptions = {'simLabel': 'RVPM_sim',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

## For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)