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
from numpy import array, ones, repeat
from collections import OrderedDict

### Import all experiments, models, outputting and interface functions
# The experiment factory
from experiments import experiments
# The experiments and stimulus processors
from experiment.probSelect import probSelect, probSelectStimDirect, probSelectRewDirect

# The model factory
from models import models
# The decision methods
from model.decision.discrete import decWeightProb
# The model
from model.qLearn import qLearn

from outputting import outputting

### Set the outputting, model sets and experiment sets
expParams = {}
expExtraParams = {'numActions': 6,
                  'learningLen': 200,
                  'testLen': 100,
                  'rewardSize': 1,
                  'actRewardProb': OrderedDict([("A", 0.80),
                                                ("B", 0.20),
                                                ("C", 0.70),
                                                ("D", 0.30),
                                                ("E", 0.60),
                                                ("F", 0.40)]),
                  'learnActPairs': [("A", "B"), ("C", "D"), ("E", "F")]}
expSets = experiments((probSelect, expParams, expExtraParams))

numActions = 6
numCues = 1
repetitions = 30
alphaSet = repeat(array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
betaSet = array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])

parameters = {'alpha': alphaSet,
              'beta': betaSet}
paramExtras = {'numActions': numActions,
               'numCues': numCues,
               'actionCodes': {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
               'expect': ones((numActions, numCues)) / 2,
               'prior': ones(numActions) / numActions,
               'stimFunc': probSelectStimDirect(),
               'rewFunc': probSelectRewDirect(),
               'decFunc': decWeightProb(["A", "B", "C", "D", "E", "F"])}

modelSet = models((qLearn, parameters, paramExtras))

outputOptions = {'simLabel': 'qLearn_probSelectSimSet',
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