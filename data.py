# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
import pickle
import scipy.io as io
import numpy as np
import pandas as pd

import os
import re
import collections

import utils


class LengthError(Exception):
    pass


class IDError(Exception):
    pass


class DimentionError(Exception):
    pass


class FileTypeError(Exception):
    pass


class FileError(Exception):
    pass


class FoldersError(Exception):
    pass


class ProcessingError(Exception):
    pass


class FileFilterError(Exception):
    pass


DATA_KEYWORDS = {"filename": "filename",
                 "ID": "participant_ID",
                 "folder": "folder"}


class Data(list):

    @classmethod
    def load_data(cls,
                  file_type='csv',
                  folders='./',
                  file_name_filter=None,
                  terminal_ID=True,
                  split_by=None,
                  participantID=None,
                  choices='actions',
                  feedbacks='feedbacks',
                  stimuli=None,
                  action_options=None,
                  group_by=None,
                  extra_processing=None,
                  data_read_options=None):
        """
        Import data from a folder. This is a wrapper function for the other import methods

        Parameters
        ----------
        file_type : string, optional
            The file type of the data, from ``mat``, ``csv``, ``xlsx`` and ``pkl``. Default is ``csv``
        folders : string or list of strings, optional
            The folder or folders where the data can be found. Default is the current folder.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``
        split_by : string or list of strings, optional
            If multiple participant datasets are in one file sheet, this specifies the column or columns that can
            distinguish and identify the rows for each participant. Default ``None``
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``None``, which results in the file name being
            used.
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or None or one element list with a list, optional
            If a string or list of strings these are treated as dict keys where the valid actions for each trial can
            be found. If None then all trials will use all available actions. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``
        group_by : list of strings, optional
            A list of parts of filenames that are repeated across participants, identifying all the files that should
            be grouped together to form one participants data. The rest of the filename is assumed to identify the
            participant. Default is ``None``
        extra_processing : callable, optional
            A function that modifies the dictionary of data read for each participant in such that it is appropriate
            for fitting. Default is ``None``
        data_read_options : dict, optional
            The keyword arguments for the data importing method chosen

        Returns
        -------
        Data : Data class instance
        """
        if isinstance(folders, str):
            folder_list = [folders]
        elif isinstance(folders, list):
            folder_list = folders
        else:
            raise FoldersError('``folders`` must be a string or a list of strings. Found {}'.format(type(folders)))

        dat = None
        for folder in folder_list:
            if file_type == 'mat':
                subdat = cls.from_mat(folder=folder,
                                      file_name_filter=file_name_filter,
                                      terminal_ID=terminal_ID,
                                      participantID=participantID,
                                      choices=choices,
                                      feedbacks=feedbacks,
                                      stimuli=stimuli,
                                      action_options=action_options,
                                      group_by=group_by,
                                      extra_processing=extra_processing)
            elif file_type == 'csv':
                subdat = cls.from_csv(folder=folder,
                                      file_name_filter=file_name_filter,
                                      terminal_ID=terminal_ID,
                                      split_by=split_by,
                                      participantID=participantID,
                                      choices=choices,
                                      feedbacks=feedbacks,
                                      stimuli=stimuli,
                                      action_options=action_options,
                                      group_by=group_by,
                                      extra_processing=extra_processing,
                                      csv_read_options=data_read_options)
            elif file_type == 'xlsx':
                subdat = cls.from_xlsx(folder=folder,
                                       file_name_filter=file_name_filter,
                                       terminal_ID=terminal_ID,
                                       split_by=split_by,
                                       participantID=participantID,
                                       choices=choices,
                                       feedbacks=feedbacks,
                                       stimuli=stimuli,
                                       action_options=action_options,
                                       group_by=group_by,
                                       extra_processing=extra_processing,
                                       xlsx_read_options=data_read_options)
            elif file_type == 'pkl':
                subdat = cls.from_pkl(folder=folder,
                                      file_name_filter=file_name_filter,
                                      terminal_ID=terminal_ID,
                                      participantID=participantID,
                                      choices=choices,
                                      feedbacks=feedbacks,
                                      stimuli=stimuli,
                                      action_options=action_options,
                                      group_by=group_by,
                                      extra_processing=extra_processing)
            else:
                raise FileTypeError('{} is not a supported file type. Please use ``mat``, ``csv``, ``xlsx`` or ``pkl``'.format(file_type))

            if dat is None:
                dat = subdat
            else:
                dat.extend(subdat)

        return dat

    @classmethod
    def from_mat(cls,
                 folder='./',
                 file_name_filter=None,
                 terminal_ID=True,
                 participantID=None,
                 choices='actions',
                 feedbacks='feedbacks',
                 stimuli=None,
                 action_options=None,
                 group_by=None,
                 extra_processing=None):
        """
        Import data from a folder full of .mat files, where each file contains the information of one participant

        Parameters
        ----------
        folder : string, optional
            The folder where the data can be found. Default is the current folder.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``None``, which results in the file name being
            used.
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or None or one element list with a list, optional
            If a string or list of strings these are treated as dict keys where the valid actions for each trial can
            be found. If None then all trials will use all available actions. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``
        group_by : list of strings, optional
            A list of parts of filenames that are repeated across participants, identifying all the files that should
            be grouped together to form one participants data. The rest of the filename is assumed to identify the
            participant. Default is ``None``
        extra_processing : callable, optional
            A function that modifies the dictionary of data read for each participant in such that it is appropriate
            for fitting. Default is ``None``

        Returns
        -------
        Data : Data class instance

        See Also
        --------
        scipy.io.loadmat
        """
        folder_path = cls.__folder_path_cleaning(folder)

        files, file_IDs = cls.__locate_files(folder_path, "mat", file_name_filter=file_name_filter, terminal_ID=terminal_ID)

        participant_data = []
        for f, i in zip(files, file_IDs):

            file_data = {DATA_KEYWORDS['filename']: f,
                         DATA_KEYWORDS['folder']: folder_path}
            if participantID is None:
                file_data[DATA_KEYWORDS['ID']] = i

            mat = io.loadmat(folder_path + f, struct_as_record=False, squeeze_me=True)
            for key, value in mat.items():
                if key[0:2] == "__":
                    continue
                elif type(value) == io.matlab.mio5_params.mat_struct:
                    data_structure = {s: getattr(value, s) for s in value._fieldnames}
                    file_data.update(data_structure)
                else:
                    file_data[key] = value

            participant_data.append(file_data)

        if participantID is None:
            participantID = DATA_KEYWORDS['ID']

        participant_processed_data = cls.__clean_data(participant_data,
                                                      extra_processing=extra_processing,
                                                      group_by=group_by)

        return cls(participant_processed_data,
                   participantID=participantID,
                   choices=choices,
                   feedbacks=feedbacks,
                   stimuli=stimuli,
                   action_options=action_options)

    @classmethod
    def from_csv(cls,
                 folder='./',
                 file_name_filter=None,
                 terminal_ID=True,
                 split_by=None,
                 participantID=None,
                 choices='actions',
                 feedbacks='feedbacks',
                 stimuli=None,
                 action_options=None,
                 group_by=None,
                 extra_processing=None,
                 csv_read_options=None):
        """
        Import data from a folder full of .csv files, where each file contains the information of one participant

        Parameters
        ----------
        folder : string, optional
            The folder where the data can be found. Default is the current folder.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``
        split_by : string or list of strings, optional
            If multiple participants datasets are in one file sheet, this specifies the column or columns that can
            distinguish and identify the rows for each participant. Default ``None``
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``None``, which results in the file name being
            used.
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or None or one element list with a list, optional
            If a string or list of strings these are treated as dict keys where the valid actions for each trial can
            be found. If None then all trials will use all available actions. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``
        group_by : list of strings, optional
            A list of parts of filenames that are repeated across participants, identifying all the files that should
            be grouped together to form one participants data. The rest of the filename is assumed to identify the
            participant. Default is ``None``
        extra_processing : callable, optional
            A function that modifies the dictionary of data read for each participant in such that it is appropriate
            for fitting. Default is ``None``
        csv_read_options : dict, optional
            The keyword arguments for pandas.read_csv. Default ``{}``

        Returns
        -------
        Data : Data class instance

        See Also
        --------
        pandas.read_csv
        """
        folder_path = cls.__folder_path_cleaning(folder)

        files, file_IDs = cls.__locate_files(folder_path, "csv", file_name_filter=file_name_filter,
                                             terminal_ID=terminal_ID)

        if split_by is None:
            split_by = []
        elif isinstance(split_by, str):
            split_by = [split_by]
        elif isinstance(split_by, (list, np.ndarray)):
            for s in split_by:
                if not isinstance(s, str):
                    raise TypeError('A split_by list should only contain strings. Found {}'.format(type(s)))
        else:
            raise TypeError('split_by should be a string or a list of strings. Found {}'.format(type(split_by)))

        if csv_read_options is None:
            csv_read_options = {}

        participant_data = []

        participantID_changed = False
        for filename, fileID in zip(files, file_IDs):

            dat = pd.read_csv(folder_path + filename, **csv_read_options)

            if split_by:
                classifier_list = []
                for s in split_by:
                    try:
                        dat[s].fillna(method='ffill', inplace=True)
                    except KeyError as err:
                        raise KeyError('Data split by contains a column that does not exist: ``{}``'.format(s))
                    if dat[s].dtype in [np.dtype('int64'), np.dtype('float64')]:
                        sSorted = sorted(list(set(dat[s])))
                        classifier_list.append(sSorted)
                    else:
                        classifier_list.append(cls.__sort_strings(list(set(dat[s])), ''))
                participants = utils.listMerge(*classifier_list)

                for p in participants:
                    sub_dat = dat[(dat[split_by] == p).all(axis=1)]
                    sub_dat_dict = sub_dat.to_dict(orient='list')
                    sub_dat_dict[DATA_KEYWORDS['filename']] = filename
                    sub_dat_dict[DATA_KEYWORDS['folder']] = folder_path
                    if participantID is None or participantID == split_by[0]:
                        participantID_changed = True
                        if len(p) > 1:
                            sub_dat_dict[DATA_KEYWORDS['ID']] = "-".join([str(pi) for pi in p])
                        else:
                            sub_dat_dict[DATA_KEYWORDS['ID']] = p[0]

                    participant_data.append(sub_dat_dict)
            else:
                dat_dict = dat.to_dict(orient='list')
                dat_dict[DATA_KEYWORDS['filename']] = filename
                dat_dict[DATA_KEYWORDS['folder']] = folder_path
                if participantID is None:
                    dat_dict[DATA_KEYWORDS['ID']] = fileID
                    participantID_changed = True
                elif participantID in dat_dict and isinstance(dat_dict[participantID], (list, np.ndarray)):
                    if utils.list_all_equal(dat_dict[participantID][1:]):
                        dat_dict[DATA_KEYWORDS['ID']] = dat_dict[participantID][0]
                        participantID_changed = True
                    else:
                        raise TypeError("participantID's column, {}, had more than one value".format(participantID))

                participant_data.append(dat_dict)

        if participantID_changed:
            participantID = DATA_KEYWORDS['ID']

        participant_processed_data = cls.__clean_data(participant_data,
                                                      extra_processing=extra_processing,
                                                      group_by=group_by)

        return cls(participant_processed_data,
                   participantID=participantID,
                   choices=choices,
                   feedbacks=feedbacks,
                   stimuli=stimuli,
                   action_options=action_options)

    @classmethod
    def from_xlsx(cls,
                  folder='./',
                  file_name_filter=None,
                  terminal_ID=True,
                  split_by=None,
                  participantID=None,
                  choices='actions',
                  feedbacks='feedbacks',
                  stimuli=None,
                  action_options=None,
                  group_by=None,
                  extra_processing=None,
                  xlsx_read_options=None):
        """
        Import data from a folder full of .xlsx files, where each file contains the information of one participant

        Parameters
        ----------
        folder : string, optional
            The folder where the data can be found. Default is the current folder.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``
        split_by : string or list of strings, optional
            If multiple participants datasets are in one file sheet, this specifies the column or columns that can
            distinguish and identify the rows for each participant. Default ``None``
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``None``, which results in the file name being
            used.
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or None or one element list with a list, optional
            If a string or list of strings these are treated as dict keys where the valid actions for each trial can
            be found. If None then all trials will use all available actions. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``
        group_by : list of strings, optional
            A list of parts of filenames that are repeated across participants, identifying all the files that should
            be grouped together to form one participants data. The rest of the filename is assumed to identify the
            participant. Default is ``None``
        extra_processing : callable, optional
            A function that modifies the dictionary of data read for each participant in such that it is appropriate
            for fitting. Default is ``None``
        xlsx_read_options : dict, optional
            The keyword arguments for pandas.read_excel

        Returns
        -------
        Data : Data class instance

        See Also
        --------
        pandas.read_excel
        """
        folder_path = cls.__folder_path_cleaning(folder)

        files, file_IDs = cls.__locate_files(folder_path, "xlsx", file_name_filter=file_name_filter,
                                             terminal_ID=terminal_ID)

        if split_by is None:
            split_by = []
        elif isinstance(split_by, str):
            split_by = [split_by]
        elif isinstance(split_by, (list, np.ndarray)):
            for s in split_by:
                if not isinstance(s, str):
                    raise TypeError('A split_by list should only contain strings. Found {}'.format(type(s)))
        else:
            raise TypeError('split_by should be a string or a list of strings. Found {}'.format(type(split_by)))

        if xlsx_read_options is None:
            xlsx_read_options = {}

        participant_data = []

        participantID_changed = False
        for filename, fileID in zip(files, file_IDs):

            # In case the file is open, this will in fact be a temporary file and not a valid file.
            if filename.startswith('~$'):
                continue

            dat = pd.read_excel(folder_path + filename, **xlsx_read_options)

            if split_by:
                classifier_list = []
                for s in split_by:
                    try:
                        dat[s].fillna(method='ffill', inplace=True)
                    except KeyError as err:
                        raise KeyError('Data split by contains a column that does not exist: ``{}``'.format(s))
                    if dat[s].dtype in [np.dtype('int64'), np.dtype('float64')]:
                        sSorted = sorted(list(set(dat[s])))
                        classifier_list.append(sSorted)
                    else:
                        classifier_list.append(cls.__sort_strings(list(set(dat[s])), ''))

                participants = utils.listMerge(*classifier_list)

                for p in participants:
                    sub_dat = dat[(dat[split_by] == p).all(axis=1)]
                    sub_dat_dict = sub_dat.to_dict(orient='list')
                    sub_dat_dict[DATA_KEYWORDS['filename']] = filename
                    sub_dat_dict[DATA_KEYWORDS['folder']] = folder_path
                    if participantID is None or participantID == split_by[0]:
                        participantID_changed = True
                        if len(p) > 1:
                            sub_dat_dict[DATA_KEYWORDS['ID']] = "-".join([str(pi) for pi in p])
                        else:
                            sub_dat_dict[DATA_KEYWORDS['ID']] = p[0]
                    participant_data.append(sub_dat_dict)
            else:
                dat_dict = dat.to_dict(orient='list')
                dat_dict[DATA_KEYWORDS['filename']] = filename
                dat_dict[DATA_KEYWORDS['folder']] = folder_path
                if participantID is None:
                    dat_dict[DATA_KEYWORDS['ID']] = fileID
                    participantID_changed = True
                elif participantID in dat_dict and isinstance(dat_dict[participantID], (list, np.ndarray)):
                    if utils.list_all_equal(dat_dict[participantID][1:]):
                        dat_dict[DATA_KEYWORDS['ID']] = dat_dict[participantID][0]
                        participantID_changed = True
                    else:
                        raise TypeError("participantID's column, {}, had more than one value".format(participantID))

                participant_data.append(dat_dict)

        if participantID_changed:
            participantID = DATA_KEYWORDS['ID']

        participant_processed_data = cls.__clean_data(participant_data,
                                                      extra_processing=extra_processing,
                                                      group_by=group_by)

        return cls(participant_processed_data,
                   participantID=participantID,
                   choices=choices,
                   feedbacks=feedbacks,
                   stimuli=stimuli,
                   action_options=action_options)

    @classmethod
    def from_pkl(cls,
                 folder='./',
                 file_name_filter=None,
                 terminal_ID=True,
                 participantID=None,
                 choices='actions',
                 feedbacks='feedbacks',
                 stimuli=None,
                 action_options=None,
                 group_by=None,
                 extra_processing=None):
        """
        Import data from a folder full of .pkl files, where each file contains the information of one participant.
        This will principally be used to import data stored by task simulations

        Parameters
        ----------
        folder : string, optional
            The folder where the data can be found. Default is the current folder.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``None``, which results in the file name being
            used.
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or None or one element list with a list, optional
            If a string or list of strings these are treated as dict keys where the valid actions for each trial can
            be found. If None then all trials will use all available actions. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``
        group_by : list of strings, optional
            A list of parts of filenames that are repeated across participants, identifying all the files that should
            be grouped together to form one participants data. The rest of the filename is assumed to identify the
            participant. Default is ``None``
        extra_processing : callable, optional
            A function that modifies the dictionary of data read for each participant in such that it is appropriate
            for fitting. Default is ``None``

        Returns
        -------
        Data : Data class instance
        """
        folder_path = cls.__folder_path_cleaning(folder)

        files, file_IDs = cls.__locate_files(folder_path, "pkl", file_name_filter=file_name_filter,
                                             terminal_ID=terminal_ID)

        participant_data = []
        for filename, fileID in zip(files, file_IDs):

            with open(folder_path + filename, 'rb') as o:
                dat = pickle.load(o)

                if not isinstance(dat, dict):
                    raise TypeError("Data coming from ``.pkl`` files expected to be dictionaries. Found {}".format(type(dat)))

                dat[DATA_KEYWORDS['filename']] = filename
                dat[DATA_KEYWORDS['folder']] = folder_path

                file_data = {k: v for k, v in dat.items()}

            if participantID is None:
                file_data[DATA_KEYWORDS['ID']] = fileID

            participant_data.append(file_data)

        if participantID is None:
            participantID = DATA_KEYWORDS['ID']

        participant_processed_data = cls.__clean_data(participant_data,
                                                      extra_processing=extra_processing,
                                                      group_by=group_by)

        return cls(participant_processed_data,
                   participantID=participantID,
                   choices=choices,
                   feedbacks=feedbacks,
                   stimuli=stimuli,
                   action_options=action_options)

    def __init__(self,
                 participants,
                 participantID='ID',
                 choices='actions',
                 feedbacks='feedbacks',
                 stimuli=None,
                 action_options=None,
                 process_data_function=None):
        """

        Parameters
        ----------
        participants : list of dict
            Each dictionary contains the information for one participant
        participantID : string, optional
            The dict key where the participant ID can be found. Default ``ID``
        choices : string, optional
            The dict key where the participant choices can be found. Default ``'actions'``
        feedbacks : string, optional
            The dict key where the feedbacks the participant received can be found. Default ``'feedbacks'``
        stimuli : string or list of strings, optional
            The dict keys where the stimulus cues for each trial can be found. Default ``'None'``
        action_options : string or list of strings or one element list with a list, optional
            The dict keys where the valid actions for each trial can be found as a single key or list of keys.
            If ``None`` then the action list is considered to stay constant. If the list contains one list then it will
            be treated as a list of valid actions for each trialstep. Default ``'None'``

        """
        self.process_function = process_data_function
        if callable(process_data_function):
            participant_data = process_data_function(participants)
        elif isinstance(process_data_function, str):
            pass
        else:
            participant_data = participants

        if not isinstance(participantID, str):
            raise TypeError('participantID should be a string not a {}'.format(type(participantID)))
        if not isinstance(choices, str):
            raise TypeError('choices should be a string not a {}'.format(type(choices)))
        if not isinstance(feedbacks, str):
            raise TypeError('feedbacks should be a string not a {}'.format(type(feedbacks)))

        if stimuli is None or isinstance(stimuli, str):
            combining_stimuli = False
        elif isinstance(stimuli, list):
            combining_stimuli = True
            if not all(isinstance(s, str) for s in stimuli):
                raise TypeError('stimuli in the list should be strings: {}'.format(stimuli))
        else:
            raise TypeError('stimuli should be a string or list of strings not a {}'.format(type(stimuli)))

        if action_options is None or isinstance(action_options, str):
            combining_action_options = False
        elif isinstance(action_options, (list, np.ndarray)):
            if all(isinstance(s, str) for s in action_options):
                combining_action_options = True
            elif len(action_options) == 1:
                combining_action_options = False
            else:
                raise TypeError('The list of action_options should contain strings or one example of trial valid action choices: {}'.format(action_options))
        else:
            raise TypeError('action_options should be a string, a list of strings or a list containing one example of trial valid action choices, not a {}'.format(type(action_options)))

        self.IDs = {}
        for loc, p in enumerate(participant_data):
            if not isinstance(p, dict):
                raise TypeError("participants must be in the form of a dict, not {}".format(type(p)))

            keys = list(p.keys())

            if participantID not in keys:
                raise KeyError("participantID key not found in participant data: `{}`".format(participantID))
            participantID_value = p[participantID]

            if not isinstance(participantID_value, str):
                if isinstance(participantID_value, (int, float)):
                    participantID_value = str(participantID_value)
                else:
                    raise TypeError("participantID value must be a string. Found {}".format(type(p[participantID])))

            if participantID_value in self.IDs:
                raise IDError("participantID must be unique. Found more than one instance of `{}`".format(p[participantID]))

            self.participantID = participantID
            self.IDs[participantID_value] = loc

            if choices not in keys:
                raise KeyError("choices key not found in participant {} data: `{}`".format(p[participantID], choices))
            elif not isinstance(p[choices], (list, np.ndarray)):
                raise TypeError("choices value must be a list or numpy array. Found {} in {}".format(type(p[choices]), p[participantID]))
            else:
                self.choices = choices

            if feedbacks not in keys:
                raise KeyError("feedbacks key not found in participant {} data: `{}`".format(p[participantID], feedbacks))
            elif not isinstance(p[feedbacks], (list, np.ndarray)):
                raise TypeError("feedbacks value must be a list or numpy array. Found {} in {}".format(type(p[feedbacks]), p[participantID]))
            else:
                self.feedbacks = feedbacks

            if len(p[choices]) != len(p[feedbacks]):
                raise LengthError('The number of values for choices and feedbacks must be the same: {} choices and {} feedbacks for participant `{}`'.format(len(p[choices]), len(p[feedbacks]), p[participantID]))

            if not combining_stimuli:
                if stimuli is None:
                    self.stimuli = None
                elif stimuli not in keys:
                    raise KeyError("stimuli key not found in participant {} data: `{}`".format(p[participantID], stimuli))
                elif not isinstance(p[stimuli], (list, np.ndarray)):
                    raise TypeError("stimuli value must be a list or numpy array. Found {} in {}".format(type(p[stimuli]), p[participantID]))
                else:
                    self.stimuli = stimuli
            else:
                if not set(stimuli).issubset(set(keys)):
                    raise KeyError("stimuli keys not found in participant {} data: `{}`".format(p[participantID], stimuli))
                cues_list = [np.array(p[s])[:, np.newaxis] for s in stimuli]
                try:
                    cues_array = np.hstack(cues_list)
                except ValueError as error:
                    if all([True if len(a.shape) == 2 else False for a in cues_list]):
                        # I did not expect this
                        raise error
                    else:
                        raise DimentionError("If you are using separate keys for each stimulus cue, they must all be 1D lists")
                stimuli_combined_name = "cues_combined"
                if stimuli_combined_name in keys:
                    raise KeyError("Unexpected use of key `{}`. Use other name".format(stimuli_combined_name))
                p[stimuli_combined_name] = cues_array
                self.stimuli = stimuli_combined_name

            if stimuli and len(p[choices]) != len(p[self.stimuli]):
                raise LengthError('The number of values for choices and stimuli must be the same: {} choices and {} stimuli for participant `{}`'.format(len(p[choices]), len(p[self.stimuli]), p[participantID]))

            if not combining_action_options:
                if action_options is None:
                    self.action_options = None
                elif isinstance(action_options, (list, np.ndarray)) and len(action_options) == 1:
                    action_options_constant_name = 'constant_valid_actions'
                    participant_data[loc][action_options_constant_name] = [action_options[0]] * len(p[choices])
                    self.action_options = action_options_constant_name
                elif action_options not in keys:
                    raise KeyError("action_options key not found in participant {} data: `{}`".format(p[participantID], action_options))
                elif not isinstance(p[action_options], (list, np.ndarray)):
                    raise TypeError("action_options value must be a list or numpy array. Found {} in {}".format(type(p[action_options]), p[participantID]))
                else:
                    self.action_options = action_options
            else:
                if not set(action_options).issubset(set(keys)):
                    raise KeyError("action_options keys not found in participant {} data: {}".format(p[participantID], action_options))
                options_list = [np.array(p[a])[:, np.newaxis] for a in action_options]
                try:
                    options_array = np.hstack(options_list)
                except ValueError as error:
                    if all([True if len(a.shape) == 2 else False for a in options_list]):
                        # I did not expect this
                        raise error
                    else:
                        raise DimentionError(
                            "If you are using separate keys for each action option, they must all be 1D lists")
                action_options_combined_name = "valid_actions_combined"
                if action_options_combined_name in keys:
                    raise KeyError("Unexpected use of key `{}`. Use other name".format(action_options_combined_name))
                participant_data[loc][action_options_combined_name] = options_array
                self.action_options = action_options_combined_name

            if action_options and len(p[choices]) != len(p[self.action_options]) and len(action_options) > 1:
                raise LengthError('The number of values for choices and valid actions must be the same: {} choices and {} action_options for participant `{}`'.format(len(p[choices]), len(p[self.action_options]), p[participantID]))

        super(Data, self).__init__(participant_data)

    def extend(self, iterable):
        """Combines two Data instances into one

        Parameters
        ----------
        iterable : Data instance or list of participant dicts

        """

        if isinstance(iterable, Data):
            if self.participantID != iterable.participantID:
                raise AttributeError('participantID ``{}`` cannot be extended with ``{}``'.format(self.participantID, iterable.participantID))
            if self.choices != iterable.choices:
                raise AttributeError('choices ``{}`` cannot be extended with ``{}``'.format(self.choices, iterable.choices))
            if self.feedbacks != iterable.feedbacks:
                raise AttributeError('feedbacks ``{}`` cannot be extended with ``{}``'.format(self.feedbacks, iterable.feedbacks))
            if self.stimuli != iterable.stimuli:
                raise AttributeError('stimuli ``{}`` cannot be extended with ``{}``'.format(self.stimuli, iterable.stimuli))
            if self.action_options != iterable.action_options:
                raise AttributeError('action_options ``{}`` cannot be extended with ``{}``'.format(self.action_options, iterable.action_options))
            if self.process_function != iterable.process_function:
                raise AttributeError('process_function ``{}`` cannot be extended with ``{}``'.format(self.process_function, iterable.process_function))

            IDs = self.IDs.copy()
            number_IDs = len(IDs)
            for i, (id_key, id_val) in enumerate(iterable.IDs.items()):
                if id_key in IDs:
                    raise IDError("participantID must be unique. Found more than one instance of `{}`".format(id_key))
                else:
                    self.IDs[id_key] = number_IDs + id_val

            super(Data, self).extend(iterable)
        else:
            dat = Data(iterable,
                       participantID=self.participantID,
                       choices=self.choices,
                       feedbacks=self.feedbacks,
                       stimuli=self.stimuli,
                       action_options=self.action_options,
                       process_data_function=self.process_function
                       )
            self.extend(dat)

    def __add__(self, y):

        self.extend(y)

    def __eq__(self, other):

        if not isinstance(other, Data):
            return False

        eq_list = []
        for item1, item2 in zip(self, other):
            if any(item1.keys() != item2.keys()):
                eq_list.append(False)
            elif any(item1.values() != item2.values()):
                eq_list.append(False)
            else:
                eq_list.append(True)

        if len(eq_list) == 0:
            return True
        else:
            return eq_list

    def __ne__(self, other):

        return not self.__eq__(other)

    @staticmethod
    def __folder_path_cleaning(folder):

        folder_path = os.path.abspath(folder).replace('\\', '/')
        if folder_path[-1] != '/':
            folder_path += '/'
        return folder_path

    @classmethod
    def __locate_files(cls, folder, file_type, file_name_filter=None, terminal_ID=True):
        """
        Produces the list of valid input files

        Parameters
        ----------
        folder : string
            The folder string should end in a ``/``
        file_type : string
            The file extension found after the ``.``.
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed
        terminal_ID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed.
            Default ``True``

        Returns
        -------
        dataFiles : list
            A sorted list of the the files
        fileIDs : list of strings
            A list of unique parts of the filenames, in the order of dataFiles

        See Also
        --------
        sortStrings : sorts the files found
        """

        files = os.listdir(folder)

        data_files = [f for f in files if f.endswith(file_type)]

        valid_file_names = cls.__valid_files(data_files, file_name_filter=file_name_filter)

        if not valid_file_names:
            raise FileError('No data files found')

        sorted_files, file_IDs = cls.__sort_strings(valid_file_names,
                                                    "." + file_type,
                                                    terminalID=terminal_ID,
                                                    return_IDs=True)

        return sorted_files, file_IDs

    @classmethod
    def __valid_files(cls, data_files, file_name_filter=None):
        """
        Take a list of file names in the folder and a filter function and returns the filtered list

        Parameters
        ----------
        data_files : list of strings
            The list of file names without paths
        file_name_filter : callable, string, list of strings or None, optional
            A function to process the file names or a list of possible prefixes as strings or a single string.
            Default ``None``, no file names removed

        Returns
        -------
        valid_file_list : list of strings
            A subset of the data_files
        """

        if file_name_filter is None:
            valid_file_list = data_files
        elif callable(file_name_filter):
            valid_file_list = file_name_filter(data_files)
        elif isinstance(file_name_filter, str):
            valid_file_list = cls.__file_prefix_filter(data_files, [file_name_filter])
        elif isinstance(file_name_filter, (list, np.ndarray)):
            valid_file_list = cls.__file_prefix_filter(data_files, file_name_filter)
        else:
            raise FileFilterError('Unrecognised data file filter {}', file_name_filter)

        return valid_file_list

    @classmethod
    def __sort_strings(cls, unordered_list, suffix, terminalID=True, return_IDs=False):
        """
        Takes an unordered list of strings and sorts them if possible and necessary

        Parameters
        ----------
        unordered_list : list of strings
            A list of valid strings
        suffix : string
            A known suffix for the string
        terminalID : bool, optional
            Is there an ID number at the end of the filename? If not then a more general search will be performed. Default ``True``
        return_IDs : bool, optional
            Specify if the fileIDs should be returned. Default ``False``

        Returns
        -------
        sortedList : list of strings
            A sorted list of the the strings
        fileIDs : list of strings
            A list of unique parts of the filenames, in the order of dataFiles. Only returned if ``return_IDs=True``

        See Also
        --------
        int_core : sorts the strings with the prefix and suffix removed if they are a number
        get_unique_prefix : identifies prefixes all strings have
        """
        if len(unordered_list) <= 1:
            return unordered_list, ["all"]

        suffixLen = len(suffix)
        if not terminalID:
            suffix = cls.__get_unique_suffix(unordered_list, suffixLen)
            suffixLen = len(suffix)

        prefix = cls.__get_unique_prefix(unordered_list, suffixLen)

        sortedList, fileIDs = cls.__int_core(unordered_list, prefix, suffix)
        if not sortedList:
            sortedList, fileIDs = cls.__str_core(unordered_list, len(prefix), suffixLen)

        if return_IDs:
            return sortedList, fileIDs
        else:
            return sortedList

    @staticmethod
    def __get_unique_suffix(unorderedList, knownSuffixLen):
        """

        Parameters
        ----------
        unorderedList : list of strings
            A list of strings to be ordered
        knownSuffixLen : int
            The length of the suffix identified so far

        Returns
        -------
        suffixLen : int
            The length of the discovered suffix
        """

        for i in range(knownSuffixLen, len(unorderedList[0])):  # Starting with the known string-suffix
            sec = unorderedList[0][-i:]
            if all((sec == d[-i:] for d in unorderedList)):
                continue
            else:
                break

        return unorderedList[0][-i + 1:]

    @staticmethod
    def __get_unique_prefix(unorderedList, suffixLen):
        """
        Identifies any initial part of strings that are identical
        for all string

        Parameters
        ----------
        unorderedList : list of strings
            A list of strings to be ordered
        suffixLen : int
            The length of the identified suffix

        Returns
        -------
        prefix : string
            The initial part of the strings that is identical for all strings in
            the list
        """

        for i in range(1, len(unorderedList[0]) - suffixLen + 2):  # Assuming the prefix might be the string-suffix
            sec = unorderedList[0][:i]
            if all((sec == d[:i] for d in unorderedList)):
                continue
            else:
                break
        return unorderedList[0][:i - 1]

    @staticmethod
    def __str_core(unorderedList, prefixLen, suffixLen):
        """
        Takes the *core* part of a string and, assuming it is a string,
        sorts them. Returns the list sorted

        Parameters
        ----------
        unorderedList : list of strings
            The list of strings to be sorted
        prefixLen : int
            The length of the unchanging start of each filename
        suffixLen : int
            The length of the unchanging end of each filename

        Returns
        -------
        orderedList : list of strings
            The strings now sorted

        """

        sortingList = ((f, f[prefixLen:-suffixLen]) for f in unorderedList)
        sortedList = sorted(sortingList, key=lambda s: s[1])
        orderedList = [s[0] for s in sortedList]
        fileIDs = [s[1] for s in sortedList]

        return orderedList, fileIDs

    @staticmethod
    def __int_core(unorderedList, prefix, suffix):
        """Takes the *core* part of a string and, assuming it is an integer, sorts them.

        Parameters
        ----------
        unorderedList : list of strings
            The list of strings to be sorted
        prefix : string
            The unchanging part of the start each string
        suffix : string
            The unchanging known end of each string

        Returns
        -------
        sortedStrings : list of strings
            The strings now sorted
        """

        try:
            if suffix:
                testItem = int(unorderedList[0][len(prefix):-len(suffix)])
            else:
                testItem = int(unorderedList[0][len(prefix):])
        except ValueError:
            return [], []

        if suffix:
            core = [(d[len(prefix):-(len(suffix))], i) for i, d in enumerate(unorderedList)]
        else:
            core = [(d[len(prefix):], i) for i, d in enumerate(unorderedList)]
        coreInt = [(int(c), i) for c, i in core]

        coreSorted = sorted(coreInt)
        coreStr = [(str(c), i) for c, i in coreSorted]

        sortedStrings = [''.join([prefix, '0' * (len(core[i][0]) - len(s)), s, suffix]) for s, i in coreStr]

        return sortedStrings, [c for c, i in coreStr]

    @staticmethod
    def __file_prefix_filter(data_files, file_filter):
        """
        Takes a list of file names and a list of strings and returns the file names that start with any of the file_name_filter

        Parameters
        ----------
        data_files : list of strings
            The list of file names without paths
        file_filter : list of strings
            The list of possible prefixes

        Returns
        -------
        valid_file_list : list of strings
            A subset of the data_files
        """

        valid_file_list = []
        for f in data_files:
            for v in file_filter:
                if f.startswith(v):
                    valid_file_list.append(f)

        return valid_file_list

    @staticmethod
    def __clean_data(participant_data, extra_processing=None, group_by=None):

        if isinstance(group_by, list):
            grouped_data = {}
            for dat in participant_data:
                filename = dat[DATA_KEYWORDS['filename']]
                for group in group_by:
                    if group in filename:
                        id_label = filename.replace(group, '')
                        if id_label not in grouped_data:
                            grouped_data[id_label] = {}
                        #grouped_data[id_label].update({'{}_{}'.format(k, group): v for k, v in dat.items()})
                        grouped_data[id_label][group] = dat

            merged_data = []
            for id_label, group_data in grouped_data.items():
                group_merged_data = {'merge_id': id_label}
                keyset = set().union(*[list(v.keys()) for v in grouped_data[id_label].values()])
                for key in keyset:
                    key_values = [group_data[group][key] for group in group_by if group in group_data
                                                                                  and key in group_data[group]]
                    if utils.list_all_equal(key_values):
                        group_merged_data[key] = key_values[0]
                    else:
                        for group in group_by:
                            if group in group_data and key in group_data[group]:
                                group_merged_data['{}_{}'.format(group, key)] = key_values.pop(0)

                merged_data.append(group_merged_data)
        else:
            merged_data = participant_data

        if extra_processing:
            processed_data = []
            for file_data in merged_data:
                dat_dict = extra_processing(file_data)
                if dat_dict is None:
                    raise ProcessingError('The extra_processing function must return the data')
                else:
                    processed_data.append(dat_dict)
        else:
            processed_data = merged_data

        return processed_data

# TODO work out how you want to integrate this into getFiles
def sort_by_last_number(dataFiles):

    # sort by the last number on the filename
    footSplit = [re.search(r"\.(?:[a-zA-Z]+)$", f).start() for f in dataFiles]
    numsplit = [re.search(r"\d+(\.\d+|$)?$", f[:n]).start() for n, f in zip(footSplit, dataFiles)]

    # check if number part is a float or an int (assuming the same for all) and use the appropriate conversion
    if "." in dataFiles[0][numsplit[0]:footSplit[0]]:
        numRepr = float
    else:
        numRepr = int

    fileNameSections = [(f[:n], numRepr(f[n:d]), f[d:]) for n, d, f in zip(numsplit, footSplit, dataFiles)]

    # Sort the keys for groupFiles
    sortedFileNames = sorted(fileNameSections, key=lambda fileGroup: fileGroup[1])

    dataSortedFiles = [head + str(num) + foot for head, num, foot in sortedFileNames]

    return dataSortedFiles
