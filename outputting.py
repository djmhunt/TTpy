# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import sys
import os
import inspect
import collections

import cPickle as pickle
import pandas as pd
import datetime as dt
import shutil as shu
import numpy as np

import utils

from types import NoneType


#%% Folder management
def saving(label, save=True, pickleData=False, saveScript=True, logLevel=logging.INFO, npSetErr="log"):
    """
    Creates the folder structure for the saved data and created the log file as ``log.txt``

    Parameters
    ----------
    label : string, optional
        The label for the simulation
    save : bool, optional
        If true the data will be saved to files. Default ``True``
    pickleData : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    saveScript : bool, optional
        If true a copy of the top level script running the current function
        will be copied to the log folder. Only works if save is set to ``True``
        Default ``True``
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default ``logging.INFO``
    npSetErr : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    Returns
    -------
    outputFolder : basestring
        The folder into which the data will be saved
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    closeLoggers : function
        Closes the logging systems that have been set up

    See Also
    --------
    folderSetup : creates the folders
    """
    dateStr = date()
    if save:
        outputFolder = folderSetup(label, dateStr, pickleData=pickleData, basePath=None)
        fileNameGen = fileNameGenerator(outputFolder)
        logFile = fileNameGen('log', 'txt')

        if saveScript:
            cwd = os.getcwd().replace("\\", "/")
            for s in inspect.stack():
                p = s[1].replace("\\", "/")
                if ("outputting(" in s[4][0]) or (cwd in p and "outputting.py" not in p):
                    shu.copy(p, outputFolder)
                    break
    else:
        outputFolder = None
        logFile = None
        fileNameGen = None

    closeLoggers = fancyLogger(dateStr, logFile=logFile, logLevel=logLevel, npErrResp=npSetErr)

    logger = logging.getLogger('Framework')

    message = "Beginning experiment labelled: " + label
    logger.info(message)

    return outputFolder, fileNameGen, closeLoggers


def folderSetup(label, dateStr, pickleData=False, basePath=None):
    """
    Identifies and creates the folder the data will be stored in

    Folder will be created as "./Outputs/<simLabel>_<date>/". If that had
    previously been created then it is created as
    "./Outputs/<simLabel>_<date>_no_<#>/", where "<#>" is the first
    available integer.

    A subfolder is also created with the name ``Pickle`` if  pickleData is
    true.

    Parameters
    ----------
    label : basestring
        The label for the simulation
    dateStr : basestring
        The date identifier
    pickleData : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    basePath : basestring, optional
        The path into which the new folder will be placed. Default is current working directory

    Returns
    -------
    folderName : string
        The folder path that has just been created

    Examples
    --------
    >>> newFolder = outputting.folderSetup("a","today", basePath=".")
    >>> newFolder
    './Outputs/a_today/'

    See Also
    --------
    newFile : Creates a new file
    saving : Creates the log system

    """
    if not basePath:
        basePath = os.getcwd().replace("\\", "/")

    # While the folders have already been created, check for the next one
    folderName = "{}/Outputs/{}_{}".format(basePath, label, dateStr)
    if os.path.exists(folderName):
        i = 1
        folderName += '_no_'
        while os.path.exists(folderName + str(i)):
            i += 1
        folderName += str(i)

    folderName += "/"
    os.makedirs(folderName)

    os.makedirs(folderName + 'data/')

    if pickleData:
        os.makedirs(folderName + 'Pickle/')

    return folderName


#%% File management
def fileNameGenerator(outputFolder=None):
    """
    Keeps track of filenames that have been used and generates the next unused one

    Parameters
    ----------
    outputFolder : string, optional
        The folder into which the new file will be placed. Default is the current working directory

    Returns
    -------
    newFileName : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string

    Examples
    --------
    >>> fileNameGen = outputting.fileNameGenerator("./")
    >>> fileNameGen("a", "b")
    './a.b'
    >>> fileNameGen("a", "b")
    './a_1.b'
    >>> fileNameGen("", "")
    './'
    >>> fileNameGen = outputting.fileNameGenerator()
    >>> fileName = fileNameGen("", "")
    >>> fileName == os.getcwd()
    True
    """

    if not outputFolder:
        outputFolder = os.getcwd()

    outputFileCounts = collections.defaultdict(int)

    def newFileName(handle, extension):
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
        fileName : string
            The filename allowed for the file
        """

        if not outputFolder:
            return ''

        if extension == '':
            end = ''
        else:
            end = "." + extension

        fileName = outputFolder + handle
        fileNameForm = fileName + end

        lastCount = outputFileCounts[fileNameForm]
        outputFileCounts[fileNameForm] += 1
        if lastCount > 0:
            fileName += "_" + str(lastCount)
        # if os.path.exists(fileName + end):
        #     i = 1
        #     while os.path.exists(fileName + "_" + str(i) + end):
        #         i += 1
        #     fileName += "_" + str(i)
        fileName += end

        return fileName

    return newFileName


#%% Logging
def fancyLogger(dateStr, logFile="./log.txt", logLevel=logging.INFO, npErrResp='log'):
    """
    Sets up the style of logging for all the simulations

    Parameters
    ----------
    dateStr : basestring
        The date the log will start at
    logFile : string, optional
        Provides the path the log will be written to. Default "./log.txt"
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default logging.INFO
    npErrResp : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr

    Returns
    -------
    closeLoggers : function
        Closes the logging systems that have been set up

    See Also
    --------
    logging : The Python standard logging library
    numpy.seterr : The function npErrResp is passed to for defining the response to numpy errors
    """

    class streamLoggerSim(object):
        """
        Fake file-like stream object that redirects writes to a logger instance.
        Based on one found at:
            http://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
        """
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        # See for why this next bit is needed http://stackoverflow.com/questions/20525587/python-logging-in-multiprocessing-attributeerror-logger-object-has-no-attrib
        def flush(self):
            try:
                self.logger.flush()
            except AttributeError:
                pass

    if logFile:
        logging.basicConfig(filename=logFile,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            level=logLevel,
                            filemode='w')

        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logLevel)
        console.setFormatter(consoleFormat)
        consoleHandler = console
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(datefmt='%m-%d %H:%M',
                            format='%(name)-12s %(levelname)-8s %(message)s',
                            level=logLevel)

    # Set the standard error output
    sys.stderr = streamLoggerSim(logging.getLogger('STDERR'), logging.ERROR)
    # Set the numpy error output and save the old one
    oldnperrcall = np.seterrcall(streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR))
    np.seterr(all=npErrResp)

    logger = logging.getLogger("Setup")
    logger.info(dateStr)
    logger.info("Log initialised")
    if logFile:
        logger.info("The log you are reading was written to " + str(logFile))

    # Finally, return a function that closes the loggers when necessary
    def closeLoggers():
        """
        To run once everything has been completed.
        """

        message = "Experiment completed. Shutting down"
        logger.info(message)

        if logFile:
            logger.removeHandler(consoleHandler)

            logging.shutdown()
            sys.stderr = sys.__stdout__
            np.seterrcall(oldnperrcall)

            root = logging.getLogger()
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            for f in root.filters[:]:
                f.close()
                root.removeFilter(f)

    return closeLoggers


#%% Pickle
def pickleRec(data, handle, fileNameGen):
    """
    Writes the data to a pickle file

    Parameters
    ----------
    data : object
        Data to be written to the file
    handle : string
        The name of the file
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    """

    outputFile = fileNameGen(handle, 'pkl')

    with open(outputFile, 'w') as w:
        pickle.dump(data, w)


def pickleLog(results, fileNameGen, label="", save=True):
    """
    Stores the data in the appropriate pickle file in a Pickle subfolder of the outputting folder

    Parameters
    ----------
    results : dict
        The data to be stored
    fileNameGen : function
        Creates a new file with the name <handle> and the extension <extension>. It takes two string parameters: (``handle``, ``extension``) and
        returns one ``fileName`` string
    label : string, optional
        A label for the results file
    """

    if not fileNameGen:
        return

    # TODO: remove the pulling out of ``Name`` from inside this method and make it more explicit higher up
    name = results["Name"]
    if isinstance(name, basestring):
        handle = 'Pickle/{}'.format(name)
    else:
        raise TypeError("The ``Name`` in the participant data is of type {} and not str".format(type(name)))

    if label:
        handle += label

    pickleRec(results, handle, fileNameGen)


#%% Utils
def reframeListDicts(store, keySet=None, storeLabel=''):
    """Take a list of dictionaries and turn it into a dictionary of lists
    containing only the useful keys

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys
    keySet : list of strings, optional
        The keys whose data will be included in the return dictionary. Default ``None``, which results in all keys being included
    storeLabel : string, optional
        An identifier to be added to the beginning of each key string.
        Default is ''.

    Returns
    -------
    newStore : dict of 1D lists
        Any dictionary keys containing lists in the input have been split
        into multiple numbered keys

    See Also
    --------
    flatDictKeySet, newFlatDict

    """

    keySet = flatDictKeySet(store, keySet=keySet)

    # For every key now found
    newStore = newFlatDict(keySet, store, storeLabel)

    return newStore


def dictData2Lists(store, storeLabel=''):
    """
    Take a dictionary of arrays, values and lists and turn it into a dictionary of lists

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys
    storeLabel : string, optional
        An identifier to be added to the beginning of each key string.
        Default is ''.

    Returns
    -------
    newStore : dict of 1D lists
        Any dictionary keys containing arrays in the input have been split
        into multiple numbered keys

    See Also
    --------
    flatDictKeySet, newFlatDict
    """

    keySet, maxListLen = listDictKeySet(store)

    # For every key now found
    newStore = newListDict(keySet, maxListLen, store, storeLabel)

    return newStore


def flatDictKeySet(store, keys=None):
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
    keys : list of strings, optional
        The keys whose data will be included in the return dictionary. Default ``None``, which   results in all keys being returned

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
        if keys:
            sKeys = (k for k in s.iterkeys() if k in keys)
            abridge = True
        else:
            sKeys = s.iterkeys()
            abridge = False
        for k in sKeys:
            if k in keySet:
                continue
            v = s[k]
            if isinstance(v, (list, np.ndarray)):
                listSet, maxListLen = listKeyGen(v, maxListLen=None, returnList=False, abridge=abridge)
                if listSet is not NoneType:
                    keySet[k] = listSet
            elif isinstance(v, dict):
                dictKeySet, maxListLen = dictKeyGen(v, maxListLen=None, returnList=False, abridge=abridge)
                keySet[k] = dictKeySet
            else:
                 keySet[k] = None

    return keySet


def listDictKeySet(store):
    """
    Generates a dictionary of keys and identifiers for the new dictionary,
    splitting any keys with arrays into a set of keys, one for each column
    in the original key.

    These are named <key><location>

    Parameters
    ----------
    store : dict
        The dictionary to be split up

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted
    maxListLen : int
        The longest list in the dictionary

    See Also
    --------
    reframeListDicts, newFlatDict
    """

    keySet, maxListLen = dictKeyGen(store, maxListLen=0, returnList=True, abridge=False)

    return keySet, maxListLen


def newFlatDict(keySet, store, labelPrefix):
    """
    Takes a list of dictionaries and returns a dictionary of 1D lists.

    If a dictionary did not have that key or list element, then 'None' is
    put in its place

    Parameters
    ----------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted
    store : list of dicts
        The dictionaries would be expected to have many of the same keys.
        Any dictionary keys containing lists in the input have been split
        into multiple numbered keys
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
    >>> from numpy import array
    >>> from collections import OrderedDict
    >>> from outputting import flatDictKeySet
    >>> store = [{'test': 'string',
                 'dict': {'test': 1, },
                 'OrderedDict': OrderedDict([(0, 0.5), (1, 0.5)]),
                 'list': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                 'num': 23.6,
                 'array': array([1, 2, 3])}]
    >>> keySet = flatDictKeySet(store)
    >>> newFlatDict(keySet, store, '')
    OrderedDict([('OrderedDict_0', ['0.5']), ('OrderedDict_1', ['0.5']),
                 ('list_(0, 0)', [1]), (list_(1, 0)', [7]), ('list_(0, 1)', [2]), ('list_(1, 1)', [8]),
                 ('list_(0, 2)', [3]), ('list_(1, 2)', [9]), ('list_(0, 3)', [4]), ('list_(1, 3)', [10]),
                 ('list_(0, 4)', [5]), ('list_(1, 4)', [11]), ('list_(0, 5)', [6]), ('list_(1, 5)', [12]),
                 ('num', ['23.6']),
                 ('dict_test', ['1']),
                 ('test', ["'string'"]),
                 ('array_[0]', [array([1])]), ('array_[1]', [array([2])]), ('array_[2]', [array([3])])])
    """
    newStore = collections.OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.iteritems():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            subStore = [s[key] for s in store]
            keyStoreSet = newFlatDict(loc, subStore, newKey)
            newStore.update(keyStoreSet)

        elif isinstance(loc, (list, np.ndarray)):
            for locCo in loc:
                tempList = []
                for s in store:
                    rawVal = s.get(key, None)
                    if type(rawVal) is NoneType:
                        tempList.append(None)
                    else:
                        try:
                            tempList.append(rawVal[locCo])
                        except TypeError:
                            tempList.append(listSelection(rawVal, locCo))
                newStore.setdefault(newKey + "_" + str(locCo), tempList)

        else:
            vals = [repr(s.get(key, None)) for s in store]
            newStore.setdefault(newKey, vals)

    return newStore


def newListDict(keySet, maxListLen, store, labelPrefix):
    """
    Takes a dictionary of numbers, strings, lists and arrays and returns a dictionary of 1D arrays.

    If there is a single value, then a list is created with that value repeated

    Parameters
    ----------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted
    maxListLen : int
        The longest list in the dictionary
    store : dict
        A dictionary of numbers, strings, lists and arrays
    labelPrefix : string
        An identifier to be added to the beginning of each key string.

    Returns
    -------
    newStore : dict
        The new dictionary with the keys from the keySet and the values as
        1D lists.

    Examples
    --------
    >>> from numpy import array
    >>> from collections import OrderedDict
    >>> from outputting import listDictKeySet
    >>> store = {'test': 'string',
                 'dict': {'test': 1, },
                 'OrderedDict': OrderedDict([(0, 0.5), (1, 0.5)]),
                 'list': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                 'num': 23.6,
                 'array': array([1, 2, 3])}
    >>> keySet, maxListLen = listDictKeySet(store)
    >>> newListDict(keySet, maxListLen, store, '')
    OrderedDict([('OrderedDict_0', [0.5, None, None, None, None, None]),
                 ('OrderedDict_1', [0.5, None, None, None, None, None]),
                 ('list_[0]', [1, 2, 3, 4, 5, 6]),
                 ('list_[1]', [7, 8, 9, 10, 11, 12]),
                 ('num', [23.6, None, None, None, None, None]),
                 ('dict_test', [1, None, None, None, None, None]),
                 ('test', ['string', None, None, None, None, None]),
                 ('array', [1, 2, 3, None, None, None])])
    """

    newStore = collections.OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.iteritems():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            keyStoreSet = newListDict(loc, maxListLen, store[key], newKey)
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


def pad(vals, maxListLen):
    vLen = np.size(vals)
    if vLen < maxListLen:
        vals.extend([None for i in range(maxListLen - vLen)])
    return vals


def listSelection(data, loc):
    """

    Parameters
    ----------
    data
    loc

    Returns
    -------
    selection :

    Examples
    --------
    >>> listSelection([1,2,3], [0])
    1
    >>> listSelection([[1, 2, 3], [4, 5, 6]], [0])
    [1, 2, 3]
    >>> listSelection([[1, 2, 3], [4, 5, 6]], (0,2))
    3
    """
    if len(loc) == 1:
        return data[loc[0]]
    else:
        return listSelection(data, loc[:-1])[loc[-1]]


def dictKeyGen(store, maxListLen=None, returnList=False, abridge=False):
    """

    Parameters
    ----------
    store : dict
        The dictionary to be broken down into keys
    maxListLen : int or float with no decimal places or None, optional
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should stay ``None``. Default ``None``
    returnList : bool, optional
        Defines if the lists will be broken into 1D lists or values. Default ``False``
    abridge : bool, optional
        Defines if the final dataset will be a summary or the whole lot. Default ``False``

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted
    maxListLen : int or None
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should stay ``None``.

    Examples
    --------
    >>> from numpy import array
    >>> store = {'test': 'string',
                 'dict': {'test': 1},
                 'list': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                 'num': 23.6,
                 'array': array([1, 2, 3])}
    >>> dictKeyGen(store, maxListLen=None, returnList=False, abridge=False)
    (OrderedDict([('test', None),
                  ('list', [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5)]),
                  ('array', array([[0], [1], [2]])),
                  ('num', None),
                  ('dict', OrderedDict([('test', None)]))]),
    None)
    >>> dictKeyGen(store, maxListLen=None, returnList=False, abridge=True)
    (OrderedDict([('test', None),
                  ('list', None),
                  ('array', array([[0], [1], [2]])),
                  ('num', None),
                  ('dict', OrderedDict([('test', None)]))]),
    None)
    >>> dictKeyGen(store, maxListLen=None, returnList=True, abridge=True)
    (OrderedDict([('test', None),
                  ('list', array([[0], [1]])),
                  ('array', None),
                  ('num', None),
                  ('dict', OrderedDict([('test', None)]))]),
    6L)
    >>> dictKeyGen(store, maxListLen=7, returnList=True, abridge=True)
    (OrderedDict([('test', None),
                  ('list', array([[0], [1]])),
                  ('array', None),
                  ('num', None),
                  ('dict', OrderedDict([('test', None)]))]),
    7)
    """
    keySet = collections.OrderedDict()

    for k in store.keys():
        v = store[k]
        if isinstance(v, (list, np.ndarray)):
            listSet, maxListLen = listKeyGen(v, maxListLen=maxListLen, returnList=returnList, abridge=abridge)
            if listSet is not NoneType:
                keySet.setdefault(k, listSet)
        elif isinstance(v, dict):
            dictKeySet, maxListLen = dictKeyGen(v, maxListLen=maxListLen, returnList=returnList, abridge=abridge)
            keySet.setdefault(k, dictKeySet)
        else:
            keySet.setdefault(k, None)

    return keySet, maxListLen


def listKeyGen(data, maxListLen=None, returnList=False, abridge=False):
    """

    Parameters
    ----------
    data : numpy.ndarray or list
        The list to be broken down
    maxListLen : int or float with no decimal places or None, optional
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should stay ``None``. Default ``None``
    returnList : bool, optional
        Defines if the lists will be broken into 1D lists or values. Default ``False``
    abridge : bool, optional
        Defines if the final dataset will be a summary or the whole lot. Default ``False``

    Returns
    -------
    returnList : None or list of tuples of ints or ints
        The list of co-ordinates for the elements to be extracted from the data. If None the list is used as-is.
    maxListLen : int or None
        If returnList is ``True`` this should be the length of the longest list. If returnList is ``False``
        this should stay ``None``.

    Examples
    --------
    >>> listKeyGen([1, 2, 3], maxListLen=None, returnList=False, abridge=False)
    (array([[0], [1], [2]]), None)
    >>> listKeyGen([[1, 2, 3], [4,5,6]], maxListLen=None, returnList=False, abridge=False)
    ([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)], None)
    >>> listKeyGen([[1, 2, 3], [4,5,6]], maxListLen=None, returnList=True, abridge=False)
    (array([[0], [1]]), 3L)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=False, abridge=False)
    ([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 5), (1, 5)], None)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=False, abridge=True)
    (None, None)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=None, returnList=True, abridge=True)
    (array([[0], [1]]), 6L)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=2, returnList=True, abridge=True)
    (array([[0], [1]]), 6L)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], maxListLen=7, returnList=True, abridge=True)
    (array([[0], [1]]), 7)
    >>> listKeyGen([[1, 2, 3, 4, 5, 6]], maxListLen=None, returnList=True, abridge=True)
    (array([[0]]), 6L)
    >>> listKeyGen([1, 2, 3, 4, 5, 6], maxListLen=None, returnList=True, abridge=True)
    (None, 6L)
    """

    if returnList:
        dataShape = list(np.shape(data))
        dataShapeFirst = dataShape.pop(-1)
        if type(maxListLen) is NoneType:
            maxListLen = dataShapeFirst
        elif dataShapeFirst > maxListLen:
            maxListLen = dataShapeFirst

    else:
        dataShape = np.shape(data)

    # If we are creating an abridged dataset and the length is too long, skip it. It will just clutter up the document
    if abridge and np.prod(dataShape) > 10:
        return None, maxListLen

    # We need to calculate every combination of co-ordinates in the array
    arrSets = [range(0, i) for i in dataShape]
    # Now record each one
    locList = [tuple(loc) for loc in utils.listMergeGen(*arrSets)]
    listItemLen = len(locList[0])
    if listItemLen == 1:
        returnList = np.array(locList)#.flatten()
    elif listItemLen == 0:
        return None, maxListLen
    else:
        returnList = locList

    return returnList, maxListLen


def date():
    """
    Calculate today's date as a string in the form <year>-<month>-<day>
    and returns it

    Returns
    -------
    todayDate : basestring
        The current date in the format <year>-<month>-<day>

    """
    d = dt.datetime(1987, 1, 1)
    d = d.today()
    todayDate = "{}-{}-{}".format(d.year, d.month, d.day)

    return todayDate
