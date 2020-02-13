# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import sys
sys.path.append("../")

import pytest
import itertools
import os

import numpy as np

import simulation
import outputting

@pytest.fixture(scope="session")
def output_folder(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data", numbered=False)

    return folder_name

class TestClass_basic:

    def test_S_1(self, output_folder, capsys):
        output_path = str(output_folder)
        simulation.run(output_path=output_path)

        captured = capsys.readouterr()
        standard_captured = captured[1]

        correct = ['Setup        INFO     {}'.format(outputting.date()),
                   'Setup        INFO     Log initialised',
                   'Framework    INFO     Beginning task labelled: Untitled',
                   "Simulation   INFO     Simulation 0 contains the task 'Basic: Trials = 100'.The model used is 'QLearn: number_cues = 1, reward_shaper = model.modelTemplate.Rewards with Name : model.modelTemplate.Rewards, stimulus_shaper = model.modelTemplate.Stimulus with Name : model.modelTemplate.Stimulus, number_critics = 2, prior = 0.5 0.5, expectation = 1.]",
                   " [1., decision_function = discrete.weightProb with task_responses : 0, 1, beta = 4, actionCode = {0: 0, 1: 1}, non_action = None, alpha = 0.3, number_actions = 2'.",
                   'Setup        INFO     Task completed. Shutting down']

        for correct_line, standard_captured_line in itertools.izip(correct, standard_captured.splitlines()):
            assert correct_line == standard_captured_line

        assert os.listdir(output_path) == []

        #file = output_folder.join('output.txt')
        #file.read()





