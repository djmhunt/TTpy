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
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
sys.path.append("../")  # So code can be found from the main folder

# Other used function
from numpy import array, concatenate, arange

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.decks import Decks, deckRewDirect, deckStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decEtaSets, decSingle, decRandom
from model.decision.discrete import decMaxProb, decProbThresh
# The model
from model.qLearn import qLearn

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {'discard': False}
expSets = experiments((Decks, expParams, expExtraParams))


numActions = 2
numCues = 1
probActions = False

repetitions = 30
alphaSet = repeat(array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
betaSet = array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])

parameters = {'alpha': alphaSet,
              'beta': betaSet}
paramExtras = {'numActions': numActions,
               'numCues': numCues,
               'probActions': probActions,
               'expect': ones((numActions, numCues)) * 5,
               'prior': ones(numActions) / numActions,
               'stimFunc': deckStimDirect(),
               'rewFunc': deckRewDirect(),
               'decFunc': decRandom()}

modelSet = models((qLearn, parameters, paramExtras))

outputOptions = {'simLabel': 'qLearn_decksSimSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': True,
                 'simRun': True,
                 'saveFittingProgress': False,
                 'saveFigures': False,
                 'saveOneFile': False,
                 'silent': True,
                 'npErrResp': 'log'}  # 'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)

