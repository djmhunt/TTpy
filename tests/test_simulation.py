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
import pathlib

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
                   "Simulation 0 contains the task Basic: nbr_of_trials = 100, number_actions = 2. The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', action_code = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

    def test_S_folder(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = pathlib.Path(output_folder)
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
                   "Simulation 0 contains the task Basic: nbr_of_trials = 100, number_actions = 2. The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', action_code = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert os.listdir(output_path) == []

    def test_S_label(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = pathlib.Path(output_folder)
        date = outputting.date()
        folder_path = output_path / 'Outputs' / 'test_{}'.format(date)

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
                   'The log you are reading was written to {}/log.txt'.format(folder_path.as_posix()),
                   'Beginning task labelled: test',
                   "Simulation 0 contains the task Basic: nbr_of_trials = 100, number_actions = 2. The model used is QLearn: number_actions = 2, number_cues = 1, number_critics = 2, prior = array([0.5, 0.5]), non_action = 'None', action_code = {0: 0, 1: 1}, stimulus_shaper = 'model.modelTemplate.Stimulus with ', reward_shaper = 'model.modelTemplate.Rewards with ', decision_function = 'discrete.weightProb with task_responses : 0, 1', alpha = 0.3, beta = 4, expectation = array([[0.5],       [0.5]]).",
                   'Shutting down program']

        for correct_line, standard_captured_line in itertools.zip_longest(correct, standard_captured):
            assert standard_captured_line == correct_line

        assert output_path.exists()
        assert (output_path / 'Outputs').exists()
        assert folder_path.exists()
        assert (folder_path / 'data').exists()
        assert not (folder_path / 'Pickle').exists()

        with open(folder_path / 'log.txt') as log:
            cleaned_log = [l.split('    ')[-1].strip() for l in log.readlines()]
            correct[-2] = correct[-2][:-15]
            final_correct = correct[-1]
            correct[-1] = '[0.5]]).'
            correct.append(final_correct)
        for correct_line, standard_captured_line in itertools.zip_longest(correct, cleaned_log):
            assert standard_captured_line == correct_line


class TestClass_simulation_overview:
    def test_tasks(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = pathlib.Path(output_folder)
        date = outputting.date()

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            main_folder = working_path.parent
        elif working_path.stem == 'TTpy':
            main_folder = working_path
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')
        task_folder = main_folder / 'tasks'

        task_list = [el.stem for el in task_folder.iterdir()
                     if el.is_file() and el.suffix == '.py' and el.stem[0] != '_' and el.stem != 'taskTemplate']
        for task in task_list:
            task_label = f'{task}_test'
            folder_path = output_path / 'Outputs' / f'{task_label}_{date}'
            simulation.run(task_name=task[0].upper() + task[1:], label=task_label, output_path=output_path)

            captured = caplog.records
            for level in [k.levelname for k in captured]:
                assert level in ['INFO']

            data_path = captured[2].message.split('The log you are reading was written to ')[1]

            assert output_path.exists()
            assert (output_path / 'Outputs').exists()
            assert folder_path.exists()
            assert data_path == (folder_path / 'log.txt').as_posix()
            assert (folder_path / 'data').exists()
            assert (folder_path / 'data' / 'modelSim_0.csv').exists()
            assert (folder_path / 'data' / 'modelSim_0.csv').stat().st_size > 0
            assert not (folder_path / 'Pickle').exists()

            caplog.clear()


#class TestClass_example:
#    def test_R_sim(self, output_folder, caplog):
#        caplog.set_level(logging.INFO)
#        output_path = pathlib.Path(output_folder)
#        date = outputting.date()
#        folder_path = output_path / 'Outputs' / 'qLearn_probSelectSimSet_{}'.format(date)
#        test_file_path = output_path / 'runScript.py'
#
#        working_path = pathlib.Path.cwd()
#        if working_path.stem == 'tests':
#            main_path = working_path.parent
#        elif working_path.stem == 'TTpy':
#            main_path = working_path
#        else:
#            raise NotImplementedError(f'Unexpected cwd {working_path}')
#        script_file = main_path / 'runScripts' / 'runScript_sim.py'
#
#        shutil.copyfile(script_file, test_file_path)
#        completed_process = subprocess.run('python ' + test_file_path.as_posix())
#
#        assert output_path.exists()
#        assert (output_path / 'Outputs').exists()
#        assert folder_path.exists()
#        assert (folder_path / 'data').exists()
#        assert (folder_path / 'Pickle').exists()
#        assert completed_process.returncode == 0
#        assert (folder_path / 'log.txt').exists()
#        assert (folder_path / 'config.yaml').exists()
#
#        # TODO: extend this to validate the data somewhat




