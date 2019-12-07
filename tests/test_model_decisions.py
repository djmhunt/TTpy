# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest
import collections

import model.decision.binary as binary
import model.decision.discrete as discrete

import numpy as np


#%% For binary.single
class TestClass_decSingle:
    def test_S_normal(self):
        np.random.seed(100)
        d = binary.single()
        result = d(0.23)
        correct_result = (0, collections.OrderedDict([(0, 0.77), (1, 0.23)]))
        assert result == correct_result

    def test_S_normal_2(self):
        last_action = 0
        np.random.seed(100)
        d = binary.single()
        result = d(0.23, last_action)
        correct_result = (0, collections.OrderedDict([(0, 0.77), (1, 0.23)]))
        assert result == correct_result

    def test_S_normal_3(self):
        last_action = 0
        np.random.seed(104)
        d = binary.single()
        result = d(0.23, last_action)
        correct_result = (1, collections.OrderedDict([(0, 0.77), (1, 0.23)]))
        assert result == correct_result

    def test_S_valid_1(self):
        np.random.seed(100)
        d = binary.single()
        result = d(0.23, validResponses=[1])
        correct_result = (1, collections.OrderedDict([(0, 0), (1, 1)]))
        assert result == correct_result

    def test_S_valid_2(self):
        np.random.seed(100)
        d = binary.single()
        result = d(0.23, validResponses=[])
        correct_result = (None, collections.OrderedDict([(0, 0.77), (1, 0.23)]))
        assert result == correct_result


#%% For discrete.weightProb
class TestClass_decWeightProb:
    def test_WP_normal(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.8, 0.5, 0.7])
        correct_result = (2, collections.OrderedDict([(1, 0.4), (2, 0.25), (3, 0.35)]))
        assert result == correct_result

    def test_WP_normal_2(self):
        np.random.seed(101)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5])
        correct_result = (3, collections.OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_WP_valid(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[1, 2])
        correct_result = (2, collections.OrderedDict([(1, 0.4), (2, 0.6), (3, 0)]))
        assert result == correct_result

    def test_WP_valid_2(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[1])
        correct_result = (1, collections.OrderedDict([(1, 1), (2, 0), (3, 0)]))
        assert result == correct_result

    def test_WP_no_valid(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[])
        correct_result = (None, collections.OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_WP_string(self):
        np.random.seed(100)
        d = discrete.weightProb(["A", "B", "C"])
        result = d([0.2, 0.3, 0.5], validResponses=["A", "B"])
        correct_result = ('B', collections.OrderedDict([('A', 0.4), ('B', 0.6), ('C', 0)]))
        assert result == correct_result

    def test_WP_err(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.6, 0.3, 0.5], validResponses=[0, 3])
        correct_result = (3, collections.OrderedDict([(1, 0), (2, 0), (3, 1)]))
        assert result == correct_result

    def test_WP_err_2(self):
        np.random.seed(100)
        d = discrete.weightProb(expResponses=[1, 2, 3])
        result = d([0.6, 0.3, 0.5], validResponses=[1, 1])
        correct_result = (1, collections.OrderedDict([(1, 1), (2, 0), (3, 0)]))
        assert result == correct_result


#%% For discrete.maxProb
class TestClass_decMaxProb:
    def test_MP_normal(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.6, 0.3, 0.5])
        correct_result = (1, collections.OrderedDict([(1, 0.6), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_normal_2(self):
        np.random.seed(101)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.5, 0.3, 0.5])
        correct_result = (3, collections.OrderedDict([(1, 0.5), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_valid(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[1, 2])
        correct_result = (2, collections.OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_valid_2(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[1])
        correct_result = (1, collections.OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_no_valid(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.2, 0.3, 0.5], validResponses=[])
        correct_result = (None, collections.OrderedDict([(1, 0.2), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_string(self):
        np.random.seed(100)
        d = discrete.maxProb(["A", "B", "C"])
        result = d([0.2, 0.3, 0.5], validResponses=["A", "B"])
        correct_result = ('B', collections.OrderedDict([('A', 0.2), ('B', 0.3), ('C', 0.5)]))
        assert result == correct_result

    def test_MP_err(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.6, 0.3, 0.5], validResponses=[0, 3])
        correct_result = (3, collections.OrderedDict([(1, 0.6), (2, 0.3), (3, 0.5)]))
        assert result == correct_result

    def test_MP_err_2(self):
        np.random.seed(100)
        d = discrete.maxProb(expResponses=[1, 2, 3])
        result = d([0.6, 0.3, 0.5], validResponses=[1, 1])
        correct_result = (1, collections.OrderedDict([(1, 0.6), (2, 0.3), (3, 0.5)]))
        assert result == correct_result


#%% For discrete.probThresh
class TestClass_decProbThresh:
    def test_PT_normal(self):
        np.random.seed(100)
        d = discrete.probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
        correct_result = (1, collections.OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
        result = d([0.2, 0.8, 0.3, 0.5])
        assert result == correct_result

    def test_PT_normal_2(self):
        np.random.seed(100)
        d = discrete.probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
        correct_result = (0, collections.OrderedDict([(0, 0.2), (1, 0.5), (2, 0.3), (3, 0.5)]))
        result = d([0.2, 0.5, 0.3, 0.5])
        assert result == correct_result

    def test_PT_normal_3(self):
        np.random.seed(101)
        d = discrete.probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
        correct_result = (3, collections.OrderedDict([(0, 0.2), (1, 0.5), (2, 0.3), (3, 0.5)]))
        result = d([0.2, 0.5, 0.3, 0.5])
        assert result == correct_result

    def test_PT_valid(self):
        np.random.seed(100)
        d = discrete.probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
        correct_result = (0, collections.OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
        result = d([0.2, 0.8, 0.3, 0.5], validResponses=[0, 2])
        assert result == correct_result

    def test_PT_no_valid(self):
        np.random.seed(100)
        d = discrete.probThresh(expResponses=[0, 1, 2, 3], eta=0.8)
        correct_result = (None, collections.OrderedDict([(0, 0.2), (1, 0.8), (2, 0.3), (3, 0.5)]))
        result = d([0.2, 0.8, 0.3, 0.5], validResponses=[])
        assert result == correct_result

    def test_PT_string(self):
        np.random.seed(100)
        d = discrete.probThresh(["A", "B", "C"])
        correct_result = ('A', collections.OrderedDict([('A', 0.2), ('B', 0.3), ('C', 0.8)]))
        result = d([0.2, 0.3, 0.8], validResponses=["A", "B"])
        assert result == correct_result

    def test_PT_err(self):
        np.random.seed(100)
        d = discrete.probThresh(["A", "B", "C"])
        correct_result = ('A', collections.OrderedDict([('A', 0.2), ('B', 0.3), ('C', 0.8)]))
        result = d([0.2, 0.3, 0.8], validResponses=["A", "D"])
        assert result == correct_result


#%% For discrete.validProbabilities
class TestClass_validProbabilities:
    def test_VP_reduced_int(self):
        correct_result = (np.array([0.1, 0.7]), np.array([2, 3]))
        result = discrete.validProbabilities([0.2, 0.1, 0.7], [1, 2, 3], [2, 3])
        assert (result[0] == correct_result[0]).all()
        assert (result[1] == correct_result[1]).all()

    def test_VP_reduced_str(self):
        correct_result = (np.array([0.1, 0.7]), np.array(['B', 'C']))
        result = discrete.validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["B", "C"])
        assert (result[0] == correct_result[0]).all()
        assert (result[1] == correct_result[1]).all()

    def test_VP_normal(self):
        correct_result = (np.array([0.2, 0.1, 0.7]), np.array(["A", "B", "C"]))
        result = discrete.validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["A", "B", "C"])
        assert (result[0] == correct_result[0]).all()
        assert (result[1] == correct_result[1]).all()

    def test_VP_err(self):
        correct_result = (np.array([0.2]), np.array(['A']))
        result = discrete.validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["A", "D"])
        assert (result[0] == correct_result[0]).all()
        assert (result[1] == correct_result[1]).all()

    def test_VP_err_2(self):
        correct_result = (np.array([0.2]), np.array(['A']))
        result = discrete.validProbabilities([0.2, 0.1, 0.7], ["A", "B", "C"], ["A", "A"])
        assert (result[0] == correct_result[0]).all()
        assert (result[1] == correct_result[1]).all()