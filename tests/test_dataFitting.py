# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import sys
sys.path.append("../")

import pytest
import pathlib
import itertools
import os

import numpy as np

import dataFitting
import data
import outputting

@pytest.fixture(scope="session")
def output_folder(tmpdir_factory):

    folder_name = tmpdir_factory.mktemp("data", numbered=False)

    return folder_name


class TestClass_basic:
    def test_DF_none(self):
        with pytest.raises(data.FileError, match='No data files found'):
            dataFitting.run()

    def test_DF_1(self, output_folder, capsys):
        output_path = str(output_folder)

        with pytest.raises(data.FileError, match='No data files found'):
            dataFitting.run(output_path=output_path)

    def test_DF_2(self, output_folder, capsys):
        output_path = str(output_folder)

        working_path = pathlib.Path.cwd()
        if working_path.stem == 'tests':
            data_folder = './test_sim/data/'
        elif working_path.stem == 'TTpy':
            data_folder = './tests/test_sim/data/'
        else:
            raise NotImplementedError(f'Unexpected cwd {working_path}')

        with pytest.raises(NameError, match='Please specify bounds for your parameters'):
            dataFitting.run(data_folder=data_folder, output_path=output_path, participantID='simID')


