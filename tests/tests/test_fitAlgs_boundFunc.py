# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import pytest
import collections

import fitAlgs.boundFunc as boundFunc

import numpy as np

#%% For infBound
class TestClass_infBound:
    def test_IB_base(self):
        cst = boundFunc.infBound(base=160)
        result = cst([0.5, 2], [(0, 1), (0, 5)])
        correct_result = 160
        assert result == correct_result

    def test_IB_inf(self):
        cst = boundFunc.infBound(base=160)
        result = cst([0.5, 7], [(0, 1), (0, 5)])
        correct_result = np.inf
        assert result == correct_result

    def test_IB_inf2(self):
        cst = boundFunc.infBound(base=160)
        result = cst([2, 7], [(0, 1), (0, 5)])
        correct_result = np.inf
        assert result == correct_result

    def test_IB_inf3(self):
        cst = boundFunc.infBound(base=160)
        result = cst([-1, 7], [(0, 1), (0, 5)])
        correct_result = np.inf
        assert result == correct_result

    def test_IB_inf3(self):
        cst = boundFunc.infBound(base=160)
        result = cst([-1, -2], [(0, 1), (0, 5)])
        correct_result = np.inf
        assert result == correct_result

#%% For scalarBound
class TestClass_scalarBound:
    def test_IB_base(self):
        cst = boundFunc.scalarBound(base=160)
        result = cst([0.5, 2], [(0, 1), (0, 5)])
        correct_result = 160
        assert result == correct_result

    def test_IB_grow(self):
        cst = boundFunc.scalarBound(base=160)
        result = cst([0.5, 7], [(0, 1), (0, 5)])
        correct_result = 162
        assert result == correct_result

    def test_IB_grow2(self):
        cst = boundFunc.scalarBound(base=160)
        result = cst([2, 7], [(0, 1), (0, 5)])
        correct_result = 163
        assert result == correct_result

    def test_IB_grow3(self):
        cst = boundFunc.scalarBound(base=160)
        result = cst([-1, 7], [(0, 1), (0, 5)])
        correct_result = 163
        assert result == correct_result

    def test_IB_grow4(self):
        cst = boundFunc.scalarBound(base=160)
        result = cst([-1, -2], [(0, 1), (0, 5)])
        correct_result = 163
        assert result == correct_result
