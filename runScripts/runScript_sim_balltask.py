# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

Notes
-----
This is a script with all the components for running an investigation. I would
recommend making a copy of this for each successful investigation and storing it
with the data.
"""
#%% Import useful functions
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
sys.path.append("../")  # So code can be found from the main folder

# Other used function
from numpy import array, ones, repeat
from collections import OrderedDict

#%% Import all experiments, models and interface functions
# The experiment factory
from experimentGenerator import ExperimentGen
# The experiments and stimulus processors
from experiment.balltask import Balltask, balltaskStimulusDirect, balltaskRewardDirect

# The model factory
from modelGenerator import ModelGen
# The decision methods
from model.decision.discrete import decWeightProb
# The model
from model.qLearn import QLearn

#%% Set the outputting, model sets and experiment sets
expParams = {}
#expExtraParams = {'numActions': 6,
#                  'learningLen': 200,
#                  'testLen': 100,
#                  'rewardSize': 1,
#                  'actRewardProb': OrderedDict([("A", 0.80),
#                                                ("B", 0.20),
#                                                ("C", 0.70),
#                                                ("D", 0.30),
#                                                ("E", 0.60),
#                                                ("F", 0.40)]),
#                  'learnActPairs': [("A", "B"), ("C", "D"), ("E", "F")]}
expStaticArgs = {}
expSets = ExperimentGen(Balltask, expParams, expStaticArgs)

numActions = 3
numCues = 3
#repetitions = 30
#alphaSet = repeat(array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
#betaSet = array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])
repetitions = 1
alphaSet = repeat(array([0.5]), repetitions)
betaSet = array([0.7])

modelParameters = {'alpha': alphaSet,
                   'beta': betaSet}
modelStaticArgs = {'numActions': numActions,
                   'numCues': numCues,
                   'actionCodes': {0: 0, 1: 1, 2: 2},
                   'expect': ones((numActions, numCues)) / 2,
                   'prior': ones(numActions) / numActions,
                   'stimFunc': balltaskStimulusDirect(),
                   'rewFunc': balltaskRewardDirect(),
                   'decFunc': decWeightProb([0, 1, 2])}

modelSet = ModelGen(QLearn, modelParameters, modelStaticArgs)

#%% For simulating experiments

from simulation import simulation

simulation(expSets,
           modelSet,
           simLabel='qLearn_balltask_simulation',
           save=True,
           saveScript=True,
           pickleData=True,
           npSetErr="log") # 'raise','log'
