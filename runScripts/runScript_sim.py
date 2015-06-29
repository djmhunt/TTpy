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
eta = 0.0
alpha = 0.5
beta = 0.5
simDur = 30
outputOptions = {'simLabel': 'qLearn_decksSim',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
parameters = {  'alpha':alpha,
                'beta':beta}
paramExtras = {'eta':eta,
               'stimFunc':deckStimDirect(),
               'decFunc':decEta(eta = eta)} #For qLearn decks

expSets = experiments((Decks,{},{}))
modelSet = models((qLearn,parameters,paramExtras))
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)