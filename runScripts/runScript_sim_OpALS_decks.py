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
from experiment.decks import Decks, deckStimDirect, deckRewDirect, deckRewDirectNormal

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decEtaSets, decSingle
from model.decision.discrete import decMaxProb
# The model
from model.OpALS import OpALS

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks, expParams, expExtraParams))

eta = 0.0
numActions = 2
numStimuli = 1
probActions = True
saturateVal = 14

alphaSet = [0.1, 0.3, 0.5, 0.7, 0.9]
betaSet = [0.5, 1, 2, 4, 8, 16]
alphaGoDiffSet = arange(-0.2, 0.25, 0.05)
rhoSet = arange(-1, 1.5, 0.5)

parameters = {'alphaCrit': alphaSet,
              'alphaGoNogoDiff':alphaGoDiffSet,
              'beta': betaSet,
              'betaDiff':rhoSet}

paramExtras = {'numActions': numActions,
               'numStimuli': numStimuli,
               'probActions': probActions,
               'saturateVal': saturateVal,
               'expect': ones((numActions, numStimuli)) * 0.6,
               'stimFunc': deckStimDirect(),
               'rewFunc': deckRewDirectNormal(10),
               'decFunc': decEta(eta=eta)}
modelSet = models((OpALS,parameters,paramExtras))

outputOptions = {'simLabel': 'OpALS_decksSimSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': True,
                 'npErrResp': 'raise'}#'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)
