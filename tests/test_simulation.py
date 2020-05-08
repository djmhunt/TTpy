# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import pytest
import os
import itertools
import logging
import shutil
import subprocess

import simulation
import outputting


@pytest.fixture(scope="session")
def output_folder(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data", numbered=True)

    return folder_name


class TestClass_basic:

    def test_S_none(self, caplog):
        caplog.set_level(logging.INFO)
        simulation.run()

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Framework', 'Simulation', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message.replace('\n', '') for k in captured]
        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Beginning task labelled: Untitled',
                   "Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', actionCode = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_folder(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = str(output_folder).replace('\\', '/')
        simulation.run(output_path=output_path)

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Framework', 'Simulation', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message.replace('\n', '') for k in captured]
        correct = ['{}'.format(outputting.date()),
                   'Log initialised',
                   'Beginning task labelled: Untitled',
                   "Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', actionCode = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert os.listdir(output_path) == []

    def test_S_label(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = str(output_folder).replace('\\', '/')
        date = outputting.date()

        simulation.run(label='test', output_path=output_path)

        captured = caplog.records

        for level in [k.levelname for k in captured]:
            assert level == 'INFO'

        captured_loggers = [k.name for k in captured]
        correct_loggers = ['Setup', 'Setup', 'Setup', 'Framework', 'Simulation', 'Setup']
        for capt, corr in itertools.zip_longest(captured_loggers, correct_loggers):
            assert capt == corr

        standard_captured = [k.message.replace('\n', '') for k in captured]
        correct = ['{}'.format(date),
                   'Log initialised',
                   'The log you are reading was written to {}/Outputs/test_{}/log.txt'.format(output_path, date),
                   'Beginning task labelled: test',
                   "Simulation 0 contains the task Basic: Trials = 100.The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', actionCode = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert os.path.exists(output_path)
        assert os.path.exists(output_path + '/Outputs')
        folder_path = output_path + '/Outputs/test_{}/'.format(date)
        assert os.path.exists(folder_path)
        assert os.path.exists(folder_path + 'data')
        assert not os.path.exists(folder_path + 'Pickle')

        with open(folder_path + 'log.txt') as log:
            cleaned_log = [l.split('    ')[-1].strip() for l in log.readlines()]
            correct[-2] = correct[-2][:-15]
            final_correct = correct[-1]
            correct[-1] = '[0.5]]).'
            correct.append(final_correct)
        for correct_line, standard_captured_line in itertools.zip_longest(correct, cleaned_log):
            assert standard_captured_line == correct_line


class TestClass_example:
    def test_R_sim(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = str(output_folder).replace('\\', '/')
        test_file_path = output_path + '/runScript.py'
        date = outputting.date()

        shutil.copyfile('../runScripts/runScript_sim.py', test_file_path)
        completedProcess = subprocess.run('python ' + test_file_path)

        assert os.path.exists(output_path)
        assert os.path.exists(output_path + '/Outputs')
        folder_path = output_path + '/Outputs/qLearn_probSelectSimSet_{}/'.format(date)
        assert os.path.exists(folder_path)
        assert os.path.exists(folder_path + 'data')
        assert os.path.exists(folder_path + 'Pickle')
        assert completedProcess.returncode == 0
        assert os.path.exists(folder_path + 'log.txt')
        assert os.path.exists(folder_path + 'config.yaml')

        # TODO: extend this to validate the data somewhat




