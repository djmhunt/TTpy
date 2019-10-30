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
import os
filePath = os.path.realpath(__file__)
codePath = filePath.split('\\')[:-2]
sys.path.append("/".join(codePath))  # So code can be found from the main folder

# Other used function
from numpy import array, ones, repeat
from collections import OrderedDict

#%% Import all experiments, models and interface functions
# The experiment factory
from experimentGenerator import ExperimentGen
# The experiments and stimulus processors
from experiment.probSelect import probSelect, probSelectStimDirect, probSelectRewDirect

# The model factory
from modelGenerator import ModelGen
# The decision methods
from model.decision.discrete import decWeightProb
# The model
from model.qLearn import qLearn

from simulation import simulation

#%% Set the model sets and experiment sets
numActions = 6
numCues = 1
repetitions = 2
alphaSet = repeat(array([0.1, 0.3, 0.5, 0.7, 0.9]), repetitions)
betaSet = array([0.1, 0.3, 0.5, 0.7, 1, 2, 4, 8, 16])

expParams = {}
expStaticArgs = {'numActions': numActions,
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

modelParameters = {'alpha': alphaSet,
                   'beta': betaSet}
modelStaticArgs = {'numActions': numActions,
                   'numCues': numCues,
                   'actionCodes': {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
                   'expect': ones((numActions, numCues)) / 2,
                   'prior': ones(numActions) / numActions,
                   'stimFunc': probSelectStimDirect(),
                   'rewFunc': probSelectRewDirect(),
                   'decFunc': decWeightProb(["A", "B", "C", "D", "E", "F"])}

modelSet = ModelGen(qLearn, modelParameters, modelStaticArgs)
expSets = ExperimentGen(probSelect, expParams, expStaticArgs)

#%% For simulating experiments
simulation(expSets,
           modelSet,
           simLabel='qLearn_probSelectSimSet',
           save=True,
           saveScript=True,
           pickleData=True,
           npSetErr="log") # 'raise','log'
