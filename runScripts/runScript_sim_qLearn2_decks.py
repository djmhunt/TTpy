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
from experiment.decks import Decks, deckRewDirect, deckStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta
# The model
from model.qLearn2 import qLearn2

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks, expParams, expExtraParams))

eta = 0.0
numActions = 2
numStimuli = 1
probActions = True

alphaSet = [0.1, 0.3, 0.5, 0.7, 0.9]
betaSet = [0.5, 1, 2, 4, 8, 16]

parameters = {'alphaPos': alphaSet,
              'alphaNeg': alphaSet,
              'beta': betaSet}
paramExtras = {'numActions': numActions,
               'numStimuli': numStimuli,
               'probActions': probActions,
               'stimFunc': deckStimDirect(),
               'rewFunc': deckRewDirect(),
               'decFunc': decEta(eta=eta)}

modelSet = models((qLearn2, parameters, paramExtras))

outputOptions = {'simLabel': 'qLearn2_decksSimSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': True,
                 'npErrResp': 'log'}  # 'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)
