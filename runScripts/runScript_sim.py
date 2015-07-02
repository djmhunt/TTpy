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
from numpy import array, concatenate, arange

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
#The models
from model.BP import BP
from model.EP import EP
from model.MS import MS
from model.MS_rev import MS_rev
from model.qLearn import qLearn
from model.qLearn2 import qLearn2
from model.OpAL import OpAL
from model.RVPM import RVPM

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {}
expSets = experiments((Decks,expParams,expExtraParams))

eta = 0.0
alphaSet = arange(0.1,0.5,0.1)
betaSet = arange(0.2,5,0.2)

parameters = {  'alpha':alphaSet,
                'beta':betaSet}
paramExtras = {'eta':eta,
               'stimFunc':deckStimDirect(),
               'decFunc':decEta(eta = eta)} #For qLearn decks
modelSet = models((qLearn,parameters,paramExtras))

outputOptions = {'simLabel': 'qLearn_decksSim',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)