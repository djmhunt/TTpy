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

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            main_path = working_path.parent
        elif working_path.stem == 'TTpy':
            main_path = working_path
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')
        script_file = main_path / 'runScripts' / 'runScripts_sim.yaml'

        with open(script_file) as file_stream:
            input_script = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        input_properties = start.generate_run_properties(input_script.copy(), script_file)
        config_file = input_properties.pop('config_file_path', None)
        output_script = start.generate_config(input_properties)

        nested_assessment(input_script, output_script)

    def test_IO_fit(self, caplog):
        caplog.set_level(logging.INFO)

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            main_path = working_path.parent
        elif working_path.stem == 'TTpy':
            main_path = working_path
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')
        script_file = main_path / 'runScripts' / 'runScripts_sim.yaml'

        with open(script_file) as file_stream:
            input_script = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        input_properties = start.generate_run_properties(input_script.copy(), script_file)
        config_file = input_properties.pop('config_file_path', None)
        output_script = start.generate_config(input_properties)

        nested_assessment(input_script, output_script)


#%% For yaml files found in runScripts
class TestClass_example:
    def test_RC_sim(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = pathlib.Path(output_folder)
        date = outputting.date()
        folder_path = output_path / 'Outputs' / 'qLearn_probSelectSimSet_{}'.format(date)
        test_file_path = output_path / 'runScript.yaml'

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            main_path = working_path.parent
        elif working_path.stem == 'TTpy':
            main_path = working_path
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')
        script_file = main_path / 'runScripts' / 'runScripts_sim.yaml'

        shutil.copyfile(script_file, test_file_path)
        start.run_config(test_file_path, trusted_file=True)

        assert output_path.exists()
        assert (output_path / 'Outputs').exists()
        assert folder_path.exists()
        assert (folder_path / 'data').exists()
        assert (folder_path / 'Pickle').exists()
        assert (folder_path / 'log.txt').exists()
        yaml_file = folder_path / 'config.yaml'
        assert yaml_file.exists()

        with open(test_file_path) as file_stream:
            original = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        with open(yaml_file) as file_stream:
            final = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        nested_assessment(original, final)

    def test_RC_fit(self, output_folder, caplog):
        caplog.set_level(logging.INFO)
        output_path = pathlib.Path(output_folder)
        date = outputting.date()
        folder_path = output_path / 'Outputs' / 'qLearn_probSelect_fromSim_{}'.format(date)
        test_file_path = output_path / 'runScript.yaml'

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            main_path = working_path.parent
        elif working_path.stem == 'TTpy':
            main_path = working_path
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')
        script_file = main_path / 'runScripts' / 'runScripts_fit.yaml'
        data_folder_source = main_path / 'tests' / 'test_sim'
        data_folder_path = output_path / 'tests' / 'test_sim'

        shutil.copyfile(script_file, test_file_path)
        shutil.copytree(data_folder_source, data_folder_path)
        start.run_config(test_file_path, trusted_file=True)

        assert output_path.exists()
        assert (output_path / 'Outputs').exists()
        assert folder_path.exists()
        assert (folder_path / 'data').exists()
        assert not (folder_path / 'Pickle').exists()
        assert (folder_path / 'log.txt').exists()
        yaml_file = folder_path / 'config.yaml'
        assert yaml_file.exists()

        with open(test_file_path) as file_stream:
            original = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        with open(yaml_file) as file_stream:
            final = yaml.load(file_stream, Loader=yaml.UnsafeLoader)

        nested_assessment(original, final)




