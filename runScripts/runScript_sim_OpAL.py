# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each sucessful investigation and storing it
 with the data.

Desired plots:
    :math:`\\alpha_N = 0.1, \\alpha_G \in ]0,0.2[`
    :math:`\\beta_N = 1, \\beta_G in ]0,2[`

    Plot Positive vs negative choice bias against :math:`prob(R|A) \in ]0.5,1[`
    with:
        :math:`\\alpha_G=\\alpha_N`, varying :math:`\\beta_G` relative to :math:`\\beta_N`
        :math:`\\beta_G=\\beta_N`, varying :math:`\\alpha_G` relative to :math:`\\alpha_N`

    Plot the range of
    :math:`\\alpha_G = 0.2 - \\alpha_N` for :math:`\\alpha_N \in ]0,0.2[` and
    :math:`\\beta_G = 2 - \\beta_N` for :math:`\\beta_N in ]0,2[` with the Y-axis being
    Choose(A) = prob(A) - prob(M),
    Avoid(B) = prob(M) - prob(B),
    Bias = choose(A) - avoid(B),
"""
### Import useful functions
# Make devision floating point by default
from __future__ import division

import sys
sys.path.append("../") #So code can be found from the main folder

# Other used function
from numpy import array, concatenate, arange, ones

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.decks import Decks, deckStimDualInfo, deckStimDirect
from experiment.beads import Beads, beadStimDirect, beadStimDualDirect, beadStimDualInfo
from experiment.pavlov import Pavlov, pavlovStimTemporal
from experiment.probSelect import probSelect, probSelectStimDirect

# The model factory
from models import models
# The decision methods
from model.decision.binary import decEta, decIntEtaReac, decSingle
from model.decision.discrete import decMaxProb
# The model
from model.OpAL import OpAL

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {'rewardProb': 0.7}
expExtraParams = {'learningLen':100}
expSets = experiments((probSelect,expParams,expExtraParams))

sA = arange(-0.1,0.15,0.05)
extended = ones((1000,len(sA)))
extended[:,:] = sA
alphaGoDiffSet = extended.flatten()

sR = arange(-1,1.5,0.5)
rhoSet = sR

numActions = 4

parameters = {'alphaGoDiff':alphaGoDiffSet,
              'betaDiff':rhoSet}
paramExtras = {'alpha': 0.1,
               'alphaC': 0.1,
               'beta': 1,
#               'betaDiff':0,
               'numActions':numActions,
               'expect': ones(numActions)/numActions,
               'expectGo': ones(numActions),
               'stimFunc':deckStimDirect(),
               'decFunc':decMaxProb(range(numActions))} #For decks
modelSet = models((OpAL,parameters,paramExtras))

outputOptions = {'simLabel': 'OpAL_simSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)