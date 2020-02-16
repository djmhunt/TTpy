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

    def test_S_none(self, capsys):
        simulation.run()

        captured = capsys.readouterr()
        standard_captured = [c[6:] for c in captured[0].splitlines()]
        error_captured = captured[1]

        correct = ['Setup        INFO     {}'.format(outputting.date()),
                   'Setup        INFO     Log initialised',
                   'Framework    INFO     Beginning task labelled: Untitled',
                   "Simulation   INFO     Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_cues = 1, reward_shaper = u'model.modelTemplate.Rewards with Name : model.modelTemplate.Rewards', stimulus_shaper = u'model.modelTemplate.Stimulus with Name : model.modelTemplate.Stimulus', number_critics = 2, prior = array([0.5, 0.5]), expectation = array([[0.5],",
                   " [0.5]]), decision_function = u'discrete.weightProb with task_responses : 0, 1', beta = 4, actionCode = {0: 0, 1: 1}, non_action = u'None', alpha = 0.3, number_actions = 2.",
                   'Setup        INFO     Shutting down program']

        for correct_line, standard_captured_line in itertools.izip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_folder(self, output_folder, capsys):
        output_path = str(output_folder).replace('\\', '/')
        simulation.run(output_path=output_path)

        captured = capsys.readouterr()
        standard_captured = [c[6:] for c in captured[0].splitlines()]
        error_captured = captured[1]

        correct = ['Setup        INFO     {}'.format(outputting.date()),
                   'Setup        INFO     Log initialised',
                   'Framework    INFO     Beginning task labelled: Untitled',
                   "Simulation   INFO     Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_cues = 1, reward_shaper = u'model.modelTemplate.Rewards with Name : model.modelTemplate.Rewards', stimulus_shaper = u'model.modelTemplate.Stimulus with Name : model.modelTemplate.Stimulus', number_critics = 2, prior = array([0.5, 0.5]), expectation = array([[0.5],",
                   " [0.5]]), decision_function = u'discrete.weightProb with task_responses : 0, 1', beta = 4, actionCode = {0: 0, 1: 1}, non_action = u'None', alpha = 0.3, number_actions = 2.",
                   'Setup        INFO     Shutting down program']

        for correct_line, standard_captured_line in itertools.izip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert os.listdir(output_path) == []

    def test_S_label(self, output_folder, capsys):
        output_path = str(output_folder).replace('\\', '/')
        simulation.run(label='test', output_path=output_path)

        captured = capsys.readouterr()
        standard_captured = [c[6:].strip() for c in captured[0].splitlines()]
        error_captured = captured[1]
        date = outputting.date()

        correct = ['Setup        INFO     {}'.format(date),
                   'Setup        INFO     Log initialised',
                   'Setup        INFO     The log you are reading was written to {}/Outputs/test_{}/log.txt'.format(output_path, date),
                   'Framework    INFO     Beginning task labelled: test',
                   "Simulation   INFO     Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_cues = 1, reward_shaper = u'model.modelTemplate.Rewards with Name : model.modelTemplate.Rewards', stimulus_shaper = u'model.modelTemplate.Stimulus with Name : model.modelTemplate.Stimulus', number_critics = 2, prior = array([0.5, 0.5]), expectation = array([[0.5],",
                   "[0.5]]), decision_function = u'discrete.weightProb with task_responses : 0, 1', beta = 4, actionCode = {0: 0, 1: 1}, non_action = u'None', alpha = 0.3, number_actions = 2.",
                   'Framework    INFO     Beginning simulation output processing',
                   'Framework    INFO     Store data for simulation 0',
                   'Setup        INFO     Shutting down program']

        for correct_line, standard_captured_line in itertools.izip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert os.path.exists(output_path)
        assert os.path.exists(output_path + '/Outputs')
        folder_path = output_path + '/Outputs/test_{}/'.format(date)
        assert os.path.exists(folder_path)
        assert os.path.exists(folder_path + 'data')
        assert not os.path.exists(folder_path + 'Pickle')

        with open(folder_path + 'log.txt') as log:
            cleaned_log = [l[6:].strip() if l[0] == ' ' else l[15:].strip() for l in log.readlines()]
        for correct_line, standard_captured_line in itertools.izip_longest(correct, cleaned_log):
            assert standard_captured_line == correct_line






