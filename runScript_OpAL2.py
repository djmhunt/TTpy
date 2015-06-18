# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

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

# Other used function
from numpy import array, concatenate, arange

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
from model.decision.binary import decEta
from model.decision.discrete import decMaxProb
#The model
from model.OpAL import OpAL

from outputting import outputting

### Set the outputting, model sets and experiment sets
outputOptions = {'simLabel': 'OpAL_simSet',
                 'save': True,
                 'saveScript': True,
                 'pickleData': False,
                 'silent': False,
                 'npErrResp' : 'log'}#'raise','log'
output = outputting(**outputOptions)

expSets = experiments((probSelect,{},{}))

alphaGSet = arange(0,0.225,0.025)
alphaNSet = 0.2 - alphaGSet

rhoSet = arange(-1,1.25,0.25)

parameters = {'alphaGo':alphaGSet,
              'alphaNogo':alphaNSet,
              'betaDiff':rhoSet}

paramExtras = {'alpha': 0.2,
               'beta': 1,
               'numActions':4,
               'stimFunc':deckStimDirect(),
               'decFunc':decMaxProb([0,1,2,3])} #For decks
modelSet = models((OpAL,parameters,paramExtras))

### For simulating experiments

from simulation import simulation

simulation(expSets, modelSet, output)