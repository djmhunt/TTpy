# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
import logging
import sys
import os
import inspect
import collections

import pickle
import pandas as pd
import datetime as dt
import shutil as shu
import numpy as np

import utils
import start


#%% Folder management
class LoggerWriter(object):
    """
    Fake file-like stream object that redirects writes to a logger instance. Taken from
    https://stackoverflow.com/a/51612402

    Parameters
    ----------
    writer : logging function
    """
    #
    def __init__(self, writer):
        self._writer = writer
        self._message = ''

    def write(self, message):
        self._message = self._message + message
        while '\n' in self._message:
            pos = self._message.find('\n')
            self._writer(self._message[:pos])
            self._message = self._message[pos + 1:]

    def flush(self):
        if self._message != '':
            self._writer(self._message)
            self._message = ''


class Saving(object):
    """
    Creates the folder structure for the saved data and created the log file as ``log.txt``

    Parameters
    ----------
    label : string, optional
        The label for the simulation. Default ``None`` will mean no data is saved to files.
    output_path : string, optional
        The path that will be used for the run output. Default ``None``
    config : dict, optional
        The parameters of the running simulation/fitting. This is used to create a YAML configuration file.
        Default ``None``
    config_file : string, optional
        The file name and path of a ``.yaml`` configuration file. Default ``None``
    pickle_store : bool, optional
        If true the data for each model, task and participant is recorded.
        Default is ``False``
    min_log_level : str, optional
        Defines the level of the log from (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``). Default ``INFO``
        See https://docs.python.org/3/library/logging.html#levels
    numpy_error_level : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    Returns
    -------
    file_name_gen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    See Also
    --------
    folderSetup : creates the folders
    """

    def __init__(self,
                 label=None,
                 output_path=None,
                 config=None,
                 config_file=None,
                 pickle_store=False,
                 min_log_level='INFO',
                 numpy_error_level="log"):

        if config is not None:
            label = config['label']
            output_path = config['output_path']
            config_file = config['config_file']
            pickle_store = config['pickle']
            min_log_level = config['min_log_level']
            numpy_error_level = config['numpy_error_level']

        self.date_string = date()
        self.label = label
        self.config = config
        self.config_file = config_file
        self.pickle_store = pickle_store
        self.numpy_error_level = numpy_error_level
        if label:
            self.save_label = label
            if output_path:
                self.base_path = output_path
            elif config_file:
                self.base_path = folder_path_cleaning(os.path.dirname(os.path.abspath(config_file)))
            else:
                self.base_path = None
        else:
            self.save_label = 'Untitled'

        possible_log_levels = {'DEBUG': logging.DEBUG,
                               'INFO': logging.INFO,
                               'WARNING': logging.WARNING,
                               'ERROR': logging.ERROR,
                               'CRITICAL': logging.CRITICAL}
        self.log_level = possible_log_levels[min_log_level]

    def __enter__(self):
        if self.label:
            output_folder = folder_setup(self.save_label,
                                         self.date_string,
                                         pickle_data=self.pickle_store,
                                         base_path=self.base_path)
            file_name_gen = file_name_generator(output_folder)
            log_file = file_name_gen('log', 'txt')

            if self.config_file:
                shu.copy(self.config_file, output_folder)

            if self.config is not None:
                config_file = file_name_gen('config', 'yaml')
                start.write_script(config_file, self.config)
        else:
            output_folder = None
            log_file = None
            file_name_gen = None

        self.close_loggers = fancy_logger(log_file=log_file,
                                          log_level=self.log_level,
                                          numpy_error_level=self.numpy_error_level)

        logger = logging.getLogger('Framework')

        message = 'Beginning task labelled: {}'.format(self.save_label)
        logger.info(message)

        return file_name_gen

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None and issubclass(exc_type, Exception):
            logger = logging.getLogger('Fatal')
            logger.error("Logging an uncaught fatal exception", exc_info=(exc_type, exc_value, exc_traceback))
        self.close_loggers()


def folder_setup(label, date_string, pickle_data=False, base_path=None):
    """
    Identifies and creates the folder the data will be stored in

    Folder will be created as "./Outputs/<sim_label>_<date>/". If that had
    previously been created then it is created as
    "./Outputs/<sim_label>_<date>_no_<#>/", where "<#>" is the first
    available integer.

    A subfolder is also created with the name ``Pickle`` if  pickle is
    true.

    Parameters
    ----------
    label : str
        The label for the simulation
    date_string : str
        The date identifier
    pickle_data : bool, optional
        If true the data for each model, task and participant is recorded.
        Default is ``False``
    base_path : str, optional
        The path into which the new folder will be placed. Default is current working directory

    Returns
    -------
    folder_name : string
        The folder path that has just been created

    See Also
    --------
    newFile : Creates a new file
    saving : Creates the log system

    """
    if not base_path:
        base_path = folder_path_cleaning(os.getcwd())
    else:
        base_path = folder_path_cleaning(base_path)

    # While the folders have already been created, check for the next one
    folder_name = "{}Outputs/{}_{}".format(base_path, label, date_string)
    if os.path.exists(folder_name):
        i = 1
        folder_name += '_no_'
        while os.path.exists(folder_name + str(i)):
            i += 1
        folder_name += str(i)

    folder_name += "/"
    os.makedirs(folder_name)

    os.makedirs(folder_name + 'data/')

    if pickle_data:
        os.makedirs(folder_name + 'Pickle/')

    return folder_name


#%% File management
def file_name_generator(output_folder=None):
    """
    Keeps track of filenames that have been used and generates the next unused one

    Parameters
    ----------
    output_folder : string, optional
        The folder into which the new file will be placed. Default is the current working directory

    Returns
    -------
    new_file_name : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    Examples
    --------
    >>> file_name_gen = file_name_generator("./")
    >>> file_name_gen("a", "b")
    './a.b'
    >>> file_name_gen("a", "b")
    './a_1.b'
    >>> file_name_gen("", "")
    './'
    >>> file_name_gen = file_name_generator()
    >>> fileName = file_name_gen("", "")
    >>> fileName == os.getcwd()
    False
    """
    if not output_folder:
        output_path = folder_path_cleaning(os.getcwd())
    else:
        output_path = folder_path_cleaning(output_folder)

    output_file_counts = collections.defaultdict(int)

    def new_file_name(handle, extension):
        """
        Creates a new unused file name with the <handle> and the extension <extension>

        Parameters
        ----------
        handle : string
            The file name
        extension : string
            The extension of the file

        Returns
        -------
        file_name : string
            The file name allowed for the file
        """

        if extension == '':
            end = ''
        else:
            end = "." + extension

        file_name = output_path + handle
        file_name_form = file_name + end

        last_count = output_file_counts[file_name_form]
        output_file_counts[file_name_form] += 1
        if last_count > 0:
            file_name += "_" + str(last_count)
        # if os.path.exists(fileName + end):
        #     i = 1
        #     while os.path.exists(fileName + "_" + str(i) + end):
        #         i += 1
        #     fileName += "_" + str(i)
        file_name += end

        return file_name

    return new_file_name


def folder_path_cleaning(folder):
    """
    Modifies string file names from Windows format to Unix format if necessary
    and makes sure there is a ``/`` at the end.

    Parameters
    ----------
    folder : string
        The folder path

    Returns
    -------
    folder_path : str
        The folder path
    """

    folder_path = folder.replace('\\', '/')
    if folder_path[-1] != '/':
        folder_path += '/'
    return folder_path


#%% Logging
def fancy_logger(log_file=None, log_level=logging.INFO, numpy_error_level='log'):
    """
    Sets up the style of logging for all the simulations

    Parameters
    ----------
    date_string : str
        The date the log will start at
    log_file : string, optional
        Provides the path the log will be written to. Default "./log.txt"
    log_level : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default logging.INFO
    numpy_error_level : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    Returns
    -------
    close_loggers : function
        Closes the logging systems that have been set up

    See Also
    --------
    logging : The Python standard logging library
    numpy.seterr : The function npErrResp is passed to for defining the response to numpy errors
    """

    old_stdout = sys.stdout
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%H:%M',
                        level=log_level)
    core_logger = logging.getLogger('')

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%y-%m-%d %H:%M')
        file_handler.setFormatter(file_format)
        core_logger.addHandler(file_handler)

    logging.captureWarnings(True)

    np.seterr(all=numpy_error_level)
    old_np_error_call = np.seterrcall(LoggerWriter(logging.getLogger('NPSTDERR').error))
    old_stderr = sys.stderr
    sys.stderr = LoggerWriter(logging.getLogger('STDERR').error)

    setup_logger = logging.getLogger('Setup')
    setup_logger.info(date())
    setup_logger.info('Log initialised')
    if log_file:
        setup_logger.info("The log you are reading was written to " + str(log_file))

    def close_loggers():
        """
        To run once everything has been completed.
        """

        message = "Shutting down program"
        setup_logger.info(message)
        logging.shutdown()
        np.seterrcall(old_np_error_call)
        sys.stderr = old_stderr
        sys.stdout = old_stdout

        for h in core_logger.handlers[:]:
            h.close()
            core_logger.removeHandler(h)

    return close_loggers


#%% Pickle
def pickle_write(data, handle, file_name_gen):
    """
    Writes the data to a pickle file

    Parameters
    ----------
    data : object
        Data to be written to the file
    handle : string
        The name of the file
    file_name_gen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    """
    output_file = file_name_gen(handle, 'pkl')

    with open(output_file, 'wb') as w:
        pickle.dump(data, w)


def pickleLog(results, file_name_gen, label=""):
    """
    Stores the data in the appropriate pickle file in a Pickle subfolder of the outputting folder

    Parameters
    ----------
    results : dict
        The data to be stored
    file_name_gen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    label : string, optional
        A label for the results file
    """

    if not file_name_gen:
        return

    # TODO: remove the pulling out of ``Name`` from inside this method and make it more explicit higher up
    name = results["Name"]
    if isinstance(name, str):
        handle = 'Pickle/{}'.format(name)
    else:
        raise TypeError("The ``Name`` in the participant data is of type {} and not str".format(type(name)))

    if label:
        handle += label

    pickle_write(results, handle, file_name_gen)


#%% Utils
def flatDictKeySet(store, selectKeys=None):
    """
    Generates a dictionary of keys and identifiers for the new dictionary,
    including only the keys in the keys list. Any keys with lists  will
    be split into a set of keys, one for each element in the original key.

    These are named <key><location>

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys.
        Any dictionary keys containing lists in the input have been split
        into multiple numbered keys
    selectKeys : list of strings, optional
        The keys whose data will be included in the return dictionary. Default ``None``, which results in all keys being returned

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted

    See Also
    --------
    reframeListDicts, newFlatDict
    """

    keySet = collections.OrderedDict()

    for s in store:
        if selectKeys:
            sKeys = (k for k in s.keys() if k in selectKeys)
            abridge = True
        else:
            sKeys = s.keys()
            abridge = False
        for k in sKeys:
            if k in keySet:
                continue
            v = s[k]
            if isinstance(v, (list, np.ndarray)):
                listSet, maxListLen = listKeyGen(v, maxListLen=None, returnList=False, abridge=abridge)
                if listSet is not None:
                    keySet[k] = listSet
            elif isinstance(v, dict):
                dictKeySet, maxListLen = dictKeyGen(v, maxListLen=None, returnList=False, abridge=abridge)
                keySet[k] = dictKeySet
            else:
                 keySet[k] = None

    return keySet


def newFlatDict(store, selectKeys=None, labelPrefix=''):
    """
    Takes a list of dictionaries and returns a dictionary of 1D lists.

    If a dictionary did not have that key or list element, then 'None' is put in its place

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys.
        Any dictionary keys containing lists in the input have been split into multiple numbered keys
    selectKeys : list of strings, optional
        The keys whose data will be included in the return dictionary. Default ``None``, which results in all keys being returned
    labelPrefix : string
        An identifier to be added to the beginning of each key string.

    Returns
    -------
    newStore : dict
        The new dictionary with the keys from the keySet and the values as
        1D lists with 'None' if the keys, value pair was not found in the
        store.

    Examples
    --------
    >>> store = [{'list': [1, 2, 3, 4, 5, 6]}]
    >>> newFlatDict(store)
    OrderedDict([('list_[0]', [1]), ('list_[1]', [2]), ('list_[2]', [3]), ('list_[3]', [4]), ('list_[4]', [5]), ('list_[5]', [6])])
    >>> store = [{'string': 'string'}]
    >>> newFlatDict(store)
    OrderedDict([(u'string', ["u'string'"])])
    >>> store = [{'dict': {1: {3: "a"}, 2: "b"}}]
    >>> newFlatDict(store)
    OrderedDict([(u'dict_1_3', ["u'a'"]), (u'dict_2', ["u'b'"])])
    """
    keySet = flatDictKeySet(store, selectKeys=selectKeys)

    newStore = collections.OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.items():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            subStore = [s[key] for s in store]
            keyStoreSet = newFlatDict(subStore, labelPrefix=newKey)
            newStore.update(keyStoreSet)
        elif isinstance(loc, (list, np.ndarray)):
            for locCo in loc:
                tempList = []
                for s in store:
                    rawVal = s.get(key, None)
                    if rawVal is None:
                        tempList.append(None)
                    else:
                        tempList.append(listSelection(rawVal, locCo))
                newStore.setdefault(newKey + "_" + str(locCo), tempList)
        else:
            vals = [repr(s.get(key, None)) for s in store]
            newStore.setdefault(newKey, vals)

    return newStore


def newListDict(store, labelPrefix='', maxListLen=0):
    """
    Takes a dictionary of numbers, strings, lists and arrays and returns a dictionary of 1D arrays.

    If there is a single value, then a list is created with that value repeated

    Parameters
    ----------
    store : dict
        A dictionary of numbers, strings, lists, dictionaries and arrays
    labelPrefix : string
        An identifier to be added to the beginning of each key string. Default empty string

    Returns
    -------
    newStore : dict
        The new dictionary with the keys from the keySet and the values as
        1D lists.

    Examples
    --------
    >>> store = {'list': [1, 2, 3, 4, 5, 6]}
    >>> newListDict(store)
    OrderedDict([('list', [1, 2, 3, 4, 5, 6])])
    >>> store = {'string': 'string'}
    >>> newListDict(store)
    OrderedDict([('string', ['string'])])
    >>> store = {'dict': {1: {3: "a"}, 2: "b"}}
    >>> newListDict(store)
    OrderedDict([(u'dict_1_3', ['a']), (u'dict_2', ['b'])])
    """

    keySet, maxListLen = dictKeyGen(store, maxListLen=maxListLen, returnList=True, abridge=False)

    newStore = collections.OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.items():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            keyStoreSet = newListDict(store[key], labelPrefix=newKey, maxListLen=maxListLen)
            newStore.update(keyStoreSet)

        elif isinstance(loc, (list, np.ndarray)):
            for locCo in loc:
                vals = list(listSelection(store[key], locCo))
                vals = pad(vals, maxListLen)
                newStore[newKey + "_" + str(locCo)] = vals

        else:
            v = store[key]
            if isinstance(v, (list, np.ndarray)):
                vals = pad(list(v), maxListLen)
            else:
                # We assume the object is a single value or string
                vals = pad([v], maxListLen)
            newStore[newKey] = vals

    return newStore


def pad(values, maxListLen):
    """
    Pads a list with None

    Parameters
    ----------
    values : list
        The list to be extended
    maxListLen : int
        The number of elements the list needs to have
    """

    vLen = np.size(values)
    if vLen < maxListLen:
        values.extend([None for i in range(maxListLen - vLen)])
    return values


def listSelection(data, loc):
    """
    Allows numpy array-like referencing of lists

    Parameters
    ----------
    data : list
        The data to be referenced
    loc : tuple of integers
        The location to be referenced

    Returns
    -------
    selection : list
        The referenced subset

    Examples
    --------
    >>> listSelection([1, 2, 3], (0,))
    1
    >>> listSelection([[1, 2, 3], [4, 5, 6]], (0,))
    [1, 2, 3]
    >>> listSelection([[1, 2, 3], [4, 5, 6]], (0, 2))
    3
    """
    if len(loc) == 0:
        return None
    elif len(loc) == 1:
        return data[loc[0]]
    else:
        subData = listSelection(data, loc[:-1])
        if len(np.shape(subData)) > 0:
            return listSelection(data, loc[:-1])[loc[-1]]
        else:
            return None


def dictKeyGen(store, maxListLen=None, returnList=False, abridge=False):
    """
    Identifies the columns necessary to convert a dictionary into a table

    Parameters
    ----------
    store : dict
        The dictionary to be broken down into keys
    maxListLen : int or float with no decimal places or None, optional
        The length of the longest  expected list. Only useful if returnList is ``True``. Default ``None``
    returnList : bool, optional
        Defines if the lists will be broken into 1D lists or values. Default ``False``, lists will be broken into values
    abridge : bool, optional
        Defines if the final dataset will be a summary or the whole lot. If it is a summary, lists of more than 10 elements are removed.
        Default ``False``, not abridged

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted
    maxListLen : int or float with no decimal places or None, optional
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should return its original value

    Examples
    --------
    >>> store = {'string': 'string'}
    >>> dictKeyGen(store)
    (OrderedDict([('string', None)]), 1)
    >>> store = {'num': 23.6}
    >>> dictKeyGen(store)
    (OrderedDict([('num', None)]), 1)
    >>> store = {'array': np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])}
    >>> dictKeyGen(store, returnList=True, abridge=True)
    (OrderedDict([(u'array', array([[0],
           [1]]))]), 6)
    >>> store = {'dict': {1: "a", 2: "b"}}
    >>> dictKeyGen(store, maxListLen=7, returnList=True, abridge=True)
    (OrderedDict([('dict', OrderedDict([(1, None), (2, None)]))]), 7)
    """
    keySet = collections.OrderedDict()

    for k in store.keys():
        v = store[k]
        if isinstance(v, (list, np.ndarray)):
            listSet, maxListLen = listKeyGen(v, maxListLen=maxListLen, returnList=returnList, abridge=abridge)
            if listSet is not None:
                keySet.setdefault(k, listSet)
            else:
                keySet.setdefault(k, None)
        elif isinstance(v, dict):
            dictKeySet, maxListLen = dictKeyGen(v, maxListLen=maxListLen, returnList=returnList, abridge=abridge)
            keySet.setdefault(k, dictKeySet)
        else:
            keySet.setdefault(k, None)

    if maxListLen is None and len(keySet) > 0:
        maxListLen = 1

    return keySet, maxListLen


def listKeyGen(data, maxListLen=None, returnList=False, abridge=False):
    """
    Identifies the columns necessary to convert a list into a table

    Parameters
    ----------
    data : numpy.ndarray or list
        The list to be broken down
    maxListLen : int or float with no decimal places or None, optional
        The length of the longest  expected list. Only useful if returnList is ``True``. Default ``None``
    returnList : bool, optional
        Defines if the lists will be broken into 1D lists or values. Default ``False``, lists will be broken into values
    abridge : bool, optional
        Defines if the final dataset will be a summary or the whole lot. If it is a summary, lists of more than 10 elements are removed.
        Default ``False``, not abridged

    Returns
    -------
    returnList : None or list of tuples of ints or ints
        The list of co-ordinates for the elements to be extracted from the data. If None the list is used as-is.
    maxListLen : int or float with no decimal places or None, optional
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should return its original value

    Examples
    --------
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=False, abridge=False)
    (array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5]]), 1)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=False, abridge=True)
    (None, None)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=True, abridge=True)
    (array([[0],
           [1]]), 6)
    """
    dataShape = np.shape(data)
    if dataShape[-1] == 0:
        return None, maxListLen

    dataShapeList = list(dataShape)
    if returnList:
        dataShapeFirst = dataShapeList.pop(-1)
        numberColumns = np.prod(dataShapeList)
        if maxListLen is None:
            maxListLen = dataShapeFirst
        elif dataShapeFirst > maxListLen:
            maxListLen = dataShapeFirst
    else:
        numberColumns = np.prod(dataShape)

    # If we are creating an abridged dataset and the length is too long, skip it. It will just clutter up the document
    if abridge and numberColumns > 10:
        return None, maxListLen

    # We need to calculate every combination of co-ordinates in the array
    arrSets = [list(range(0, i)) for i in dataShapeList]
    # Now record each one
    locList = np.array([tuple(loc) for loc in utils.listMergeGen(*arrSets)])
    listItemLen = len(locList[0])
    if listItemLen == 1:
        finalList = locList #.flatten()
    elif listItemLen == 0:
        return None, maxListLen
    else:
        finalList = locList

    if maxListLen is None and len(finalList) > 0:
        maxListLen = 1

    return finalList, maxListLen


def date():
    """
    Calculate today's date as a string in the form <year>-<month>-<day>
    and returns it

    Returns
    -------
    todayDate : str
        The current date in the format <year>-<month>-<day>

    """
    d = dt.datetime(2000, 1, 1)
    d = d.today()
    todayDate = "{}-{}-{}".format(d.year, d.month, d.day)

    return todayDate
