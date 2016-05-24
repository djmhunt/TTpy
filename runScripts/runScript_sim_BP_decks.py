# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each successful investigation and storing it
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
# The experiment and stimulus processors
from experiment.decks import Decks, deckRewDualInfo, deckStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decEtaSets, decSingle
from model.decision.discrete import decMaxProb
# The model
from model.BP import BP

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks, expParams, expExtraParams))

eta = 0
numActions = 2
numStimuli = 1
probActions = False

betaSet = [0.5, 1, 2, 4, 8, 16]

parameters = {'beta': betaSet}
paramExtras = {'numActions': numActions,
               'numStimuli': numStimuli,
               'probActions': probActions,
               'stimFunc': deckStimDirect(),
               'rewFunc': deckRewDualInfo(10, 0.01),
               'decFunc': decEta(eta=eta)}

modelSet = models((BP, parameters, paramExtras))

outputOptions = {'simLabel': 'BP_decksSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': True,
                 'npErrResp': 'log'}  # 'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)
