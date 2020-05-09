# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
import pytest
import os
import itertools
import logging
import shutil
import pathlib
import yaml

import start
import outputting

@pytest.fixture(scope="session")
def output_folder(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data", numbered=True)

    return folder_name


def nested_assessment(original, final):
    for key, val in original.items():
        assert key in final
        if isinstance(val, dict):
            nested_assessment(val, final[key])
        else:
            assert final[key] == val


#%% For read in and write out
class TestClass_IO:
    def test_IO_sim(self, caplog):
        caplog.set_level(logging.INFO)

        script_file = '../runScripts/runScripts_sim.yaml'
        with open(script_file) as file_stream:
            input_script = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        input = start.generate_run_properties(input_script.copy(), script_file)
        config_file = input.pop('config_file_path', None)
        output_script = start.generate_config(input)

        nested_assessment(input_script, output_script)

    def test_IO_fit(self, caplog):
        caplog.set_level(logging.INFO)

        script_file = '../runScripts/runScripts_fit.yaml'
        with open(script_file) as file_stream:
            input_script = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        input = start.generate_run_properties(input_script.copy(), script_file)
        config_file = input.pop('config_file_path', None)
        output_script = start.generate_config(input)

        nested_assessment(input_script, output_script)


#%% For yaml files found in runScripts
class TestClass_example:
    def test_RC_sim(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = str(output_folder).replace('\\', '/')
        test_file_path = output_path + '/runScript.yaml'
        date = outputting.date()

        shutil.copyfile('../runScripts/runScripts_sim.yaml', test_file_path)
        start.run_config(test_file_path, trusted_file=True)

        assert os.path.exists(output_path)
        assert os.path.exists(output_path + '/Outputs')
        folder_path = output_path + '/Outputs/qLearn_probSelectSimSet_{}/'.format(date)
        assert os.path.exists(folder_path)
        assert os.path.exists(folder_path + 'data')
        assert os.path.exists(folder_path + 'Pickle')
        assert os.path.exists(folder_path + 'log.txt')
        yaml_file = folder_path + 'config.yaml'
        assert os.path.exists(yaml_file)

        with open(test_file_path) as file_stream:
            original = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        with open(yaml_file) as file_stream:
            final = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        nested_assessment(original, final)

    def test_RC_fit(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = str(output_folder).replace('\\', '/')
        test_file_path = output_path + '/runScript.yaml'
        date = outputting.date()

        shutil.copyfile('../runScripts/runScripts_fit.yaml', test_file_path)
        shutil.copytree('./test_sim', output_path + '/tests/test_sim')
        start.run_config(test_file_path, trusted_file=True)

        assert os.path.exists(output_path)
        assert os.path.exists(output_path + '/Outputs')
        folder_path = output_path + '/Outputs/qLearn_probSelect_fromSim_{}/'.format(date)
        assert os.path.exists(folder_path)
        assert os.path.exists(folder_path + 'data')
        assert os.path.exists(folder_path + 'log.txt')
        yaml_file = folder_path + 'config.yaml'
        assert os.path.exists(yaml_file)

        with open(test_file_path) as file_stream:
            original = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        with open(yaml_file) as file_stream:
            final = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        nested_assessment(original, final)




