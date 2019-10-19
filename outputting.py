# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import sys

import cPickle as pickle
import pandas as pd
import datetime as dt

import shutil as shu
from os import getcwd, makedirs
from os.path import isfile, exists
from inspect import stack
from numpy import seterr, seterrcall, array, ndarray, shape, prod, log10, around, size, amax, amin
from itertools import izip
from collections import OrderedDict, Callable, defaultdict
from types import NoneType
from copy import copy

from utils import listMerGen, callableDetailsString

class outputting(object):

    """An class which manages the outputting to the screen and to files of all
    data in any form for the simulation

    Parameters
    ----------
    save : bool, optional
        If true the data will be saved to files. Default ``True``
    saveScript : bool, optional
        If true a copy of the top level script running the current function
        will be copied to the log folder. Only works if save is set to ``True``
        Default ``True``
    pickleData : bool, optional
        If true the data for each model, experiment and participant is recorded.
        Default is ``False``
    simLabel : string, optional
        The label for the simulation
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default ``logging.INFO``
    maxLabelLength : int, optional
        The maximum length of a label to be used as a reference for an
        individual model-experiment combination. Default 18
    npErrResp : {'log', 'raise'}
        Defines the response to numpy errors. Default ``log``. See numpy.seterr
    saveFittingProgress : bool, optional
        Specifies if the results from each iteration of the fitting process should be returned. Default ``False``


    See Also
    --------
    date : Identifies today's date
    saving : Sets up the log file and folder to save results
    fancyLogger : Log creator
    numpy.seterr : The function npErrResp is passed to for defining the response to numpy errors
    """

    def __init__(self, **kwargs):

        self.save = kwargs.get('save', True)
        self.saveScript = kwargs.get('saveScript', True)
        self.pickleData = kwargs.get('pickleData', False)
        self.simRun = kwargs.get('simRun', False)
        self.saveFittingProgress = kwargs.pop("saveFittingProgress", False)
        self.label = kwargs.pop("simLabel", "Untitled")
        self.logLevel = kwargs.pop("logLevel", logging.INFO)  # logging.DEBUG
        self.maxLabelLength = kwargs.pop("maxLabelLength", 18)
        self.npErrResp = kwargs.pop("npErrResp", 'log')

        # Initialise the stores of information
        self.participantFit = defaultdict(list)

        self.fitInfo = None
        self.outputFileCounts = defaultdict(int)

        self.lastExpLabelID = 0
        self.lastModelLabelID = 0

        self.date = date()

        self.saving()

        self.fancyLogger(logFile=self.logFile, logLevel=self.logLevel, npErrResp=self.npErrResp)

        self.logger = logging.getLogger('Framework')
        self.loggerSim = logging.getLogger('Simulation')

        message = "Beginning experiment labelled: " + self.label
        self.logger.info(message)

    def end(self):
        """
        To run once everything has been completed.
        """

        if len(self.participantFit) > 0:
            participantFit = pd.DataFrame.from_dict(self.participantFit)
            outputFile = self.newFile("participantFits", 'csv')
            participantFit.to_csv(outputFile)

        message = "Experiment completed. Shutting down"
        self.logger.info(message)

        if self.save:
            self.logger.removeHandler(self.consoleHandler)

            logging.shutdown()
            sys.stderr = sys.__stdout__
            seterrcall(self.oldnperrcall)

            root = logging.getLogger()
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            for f in root.filters[:]:
                f.close()
                root.removeFilter(f)

    ### Folder management

    def folderSetup(self):
        """
        Identifies and creates the folder the data will be stored in

        Folder will be created as "./Outputs/<simLabel>_<date>/". If that had
        previously been created then it is created as
        "./Outputs/<simLabel>_<date>_no_<#>/", where "<#>" is the first
        available integer.

        A subfolder is also created with the name ``Pickle`` if  pickleData is
        true.

        See Also
        --------
        newFile : Creates a new file
        saving : Creates the log system

        """

        # While the folders have already been created, check for the next one
        folderName = './Outputs/' + self.label + "_" + self.date
        if exists(folderName):
            i = 1
            folderName += '_no_'
            while exists(folderName + str(i)):
                i += 1
            folderName += str(i)

        folderName += "/"
        makedirs(folderName)

        if self.simRun or self.saveFittingProgress:
            makedirs(folderName + 'data/')

        if self.pickleData:
            makedirs(folderName + 'Pickle/')

        self.outputFolder = folderName

    ### File management
    def newFile(self, handle, extension):
        """
        Creates a new file withe the name <handle> and the extension <extension>

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

        if not self.save:
            return ''

        if extension == '':
            end = ''
        else:
            end = "." + extension

        fileName = self.outputFolder + handle
        fileNameForm = fileName + end

        lastCount = self.outputFileCounts[fileNameForm]
        self.outputFileCounts[fileNameForm] += 1
        if lastCount > 0:
            fileName += "_" + str(lastCount)
        # if exists(fileName + end):
        #     i = 1
        #     while exists(fileName + "_" + str(i) + end):
        #         i += 1
        #     fileName += "_" + str(i)
        fileName += end

        return fileName

    ### Logging
    def getLogger(self, name):
        """
        Returns a named logger stream

        Parameters
        ----------
        name : string
            Name of the logger

        Returns
        -------
        logger : logging.logger instance
            The named logger
        """

        logger = logging.getLogger(name)

        return logger

    def saving(self):
        """
        Creates the folder structure for the saved data and created the log file
        as log.txt

        See Also
        --------
        folderSetup : creates the folders
        """

        if self.save:
            self.folderSetup()
            self.logFile = self.newFile('log', 'txt')

            if self.saveScript:
                cwd = getcwd().replace("\\", "/")
                for s in stack():
                    p = s[1].replace("\\", "/")
                    if ("outputting(" in s[4][0]) or (cwd in p and "outputting.py" not in p):
                        shu.copy(p, self.outputFolder)
                        break

        else:
            self.outputFolder = ''
            self.logFile = ''

    def fancyLogger(self, logFile="./log.txt", logLevel=logging.INFO, npErrResp='log'):
        """
        Sets up the style of logging for all the simulations

        Parameters
        ----------
        logFile : string, optional
            Provides the path the log will be written to. Default "./log.txt"
        logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
            Defines the level of the log. Default logging.INFO
        npErrResp : {'log', 'raise'}
            Defines the response to numpy errors. Default ``log``. See numpy.seterr

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
            self.consoleHandler = console
            # add the handler to the root logger
            logging.getLogger('').addHandler(console)
        else:
            logging.basicConfig(datefmt='%m-%d %H:%M',
                                format='%(name)-12s %(levelname)-8s %(message)s',
                                level=logLevel)

        # Set the standard error output
        sys.stderr = streamLoggerSim(self.getLogger('STDERR'), logging.ERROR)
        # Set the numpy error output and save the old one
        self.oldnperrcall = seterrcall(streamLoggerSim(self.getLogger('NPSTDERR'), logging.ERROR))
        seterr(all=npErrResp)

        logger = self.getLogger("Setup")
        logger.info(self.date)
        logger.info("Log initialised")
        if logFile:
            logger.info("The log you are reading was written to " + str(logFile))

    def logSimParams(self, expParams, modelParams, simID):
        """
        Writes to the log the description and the label of the experiment and model

        Parameters
        ----------
        expParams : dict
            The experiment parameters
        modelParams : dict
            The model parameters
        simID : string
            The identifier for each simulation.

        See Also
        --------
        recordSimParams : Records these parameters for later use
        """

        expDesc = expParams.pop('Name') + ": "
        expDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in expParams.iteritems()]
        expDesc += ", ".join(expDescriptors)

        modelDesc = modelParams.pop('Name') + ": "
        modelDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in modelParams.iteritems()]
        modelDesc += ", ".join(modelDescriptors)

        message = "Simulation " + simID + " contains the experiment '" + expDesc + "'."
        message += "The model used is '" + modelDesc + "'."
        self.loggerSim.info(message)

    def logSimFittingParams(self, modelName, modelFitVars, modelOtherArgs, expParams={}):
        """
        Logs the model and experiment parameters that used as initial fitting
        conditions

        Parameters
        ----------
        modelName : string
            The name of the model
        modelFitVars : dict
            The model parameters that will be fitted over and varied.
        modelOtherArgs : dict
            The other parameters used in the model whose attributes have been
            modified by the user
        expParams : dict, optional
            The experiment parameters. Default ``dict()``
        """
        message = "The fit will use the model '" + modelName + "'"

        modelFitParams = [k + ' around ' + str(v).strip('[]()') for k, v in modelFitVars.iteritems()]
        message += " fitted with the parameters " + ", ".join(modelFitParams)

        modelParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelOtherArgs.iteritems() if not isinstance(v, Callable)]
        modelFuncs = [k + ' = ' + callableDetailsString(v) for k, v in modelOtherArgs.iteritems() if isinstance(v, Callable)]
        message += " and using the other user specified parameters " + ", ".join(modelParams)
        message += " and the functions " + ", ".join(modelFuncs)

        if len(expParams) > 0:
            message += ". This is based on the experiment '" + expParams['Name'] + "' "

            expDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in expParams.iteritems() if k != 'Name']
            message += "with the parameters " + ", ".join(expDescriptors) + "."

        self.loggerSim.info(message)

    def logModFittedParams(self, modelFitVars, modelParams, fitQuality, partName):
        """
        Logs the model and experiment parameters that used as initial fitting
        conditions

        Parameters
        ----------
        modelFitVars : dict
            The model parameters that have been fitted over and varied.
        modelParams : dict
            The model parameters for the fitted model
        fitQuality : float
            The value of goodness of fit
        partName : int or string
            The identifier for each participant
        """
        params = modelFitVars.keys()

        modelFitParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelParams.iteritems() if k in params]
        message = "The fitted values for participant " + str(partName) + " are " + ", ".join(modelFitParams)

        message += " with a fit quality of " + str(fitQuality) + "."

        self.loggerSim.info(message)

    def logFittingParams(self, fitInfo):
        """
        Records and outputs to the log the parameters associated with the fitting algorithms

        Parameters
        ----------
        fitInfo : dict
            The details of the fitting
        """

        self.fitInfo = fitInfo

        log = logging.getLogger('Framework')

        message = "Fitting information:"
        log.info(message)

        for f in fitInfo:
            message = "For " + f['Name'] + ":"
            log.info(message)

            for k, v in f.iteritems():
                if k == "Name":
                    continue

                message = k + ": " + repr(v)
                log.info(message)

    ### Data collection
    def recordSim(self, expData, modelData, simID):
        """
        Records the data from an experiment-model run. Creates a pickled version

        Parameters
        ----------
        expData : dict
            The data from the experiment
        modelData : dict
            The data from the model
        simID : basestring
            The label identifying the simulation

        See Also
        --------
        pickleLog : records the picked data
        """

        message = "Beginning simulation output processing"
        self.logger.info(message)

        label = "_sim-" + simID

        if self.outputFolder:

            message = "Store data for simulation " + simID
            self.logger.info(message)

            if self.simRun:
                self._simModelLog(modelData, simID)

            if self.pickleData:
                self.pickleLog(expData, "_expData" + label)
                self.pickleLog(modelData, "_modelData" + label)

    def recordParticipantFit(self, participant, partName, modelData, modelName, fitQuality, fittingData, partModelVars, expData=None):
        """
        Record the data relevant to the participant fitting

        Parameters
        ----------
        participant : dict
            The participant data
        partName : int or string
            The identifier for each participant
        modelData : dict
            The data from the model
        modelName : basestring
            The label given to the model
        fitQuality : float
            The quality of the fit as provided by the fitting function
        fittingData : dict
            Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
            tested, in the order they were tested, and a list of the fit qualities of these parameters
        partModelVars : dict of string
            A dictionary of model settings whose values should vary from participant to participant based on the
            values found in the imported participant data files. The key is the label given in the participant data file,
            as a string, and the value is the associated label in the model, also as a string.
        expData : dict, optional
            The data from the experiment. Default ``None``

        See Also
        --------
        pickleLog : records the picked data
        """

        partNameStr = str(partName)

        message = "Recording participant " + partNameStr + " model fit"
        self.logger.info(message)

        label = "_Model-" + modelName + "_Part-" + partNameStr

        participantName = "Participant " + partNameStr

        participant.setdefault("Name", participantName)
        participant.setdefault("assignedName", participantName)
        fittingData.setdefault("Name", participantName)

        if self.outputFolder:

            message = "Store data for " + participantName
            self.logger.info(message)

            self.recordFitting(fittingData, label, participant, partModelVars)

            if self.pickleData:
                if expData is not None:
                    self.pickleLog(expData, "_expData" + label)
                self.pickleLog(modelData, "_modelData" + label)
                self.pickleLog(participant, "_partData" + label)
                self.pickleLog(fittingData, "_fitData" + label)

    ### Recording
    def recordFitting(self, fittingData, label, participant, partModelVars):
        """
        Records formatted versions of the fitting data

        Parameters
        ----------
        fittingData : dict, optional
            Dictionary of details of the different fits, including an ordered dictionary containing the parameter values
            tested, in the order they were tested, and a list of the fit qualities of these parameters.
        label : basestring
            The label used to identify the fit in the file names
        participant : dict
            The participant data
        partModelVars : dict of string
            A dictionary of model settings whose values should vary from participant to participant based on the
            values found in the imported participant data files. The key is the label given in the participant data file,
            as a string, and the value is the associated label in the model, also as a string.

        Returns
        -------

        """
        extendedLabel = "ParameterFits" + label

        self.participantFit["Name"].append(participant["Name"])
        self.participantFit["assignedName"].append(participant["assignedName"])
        for k in filter(lambda x: 'fitQuality' in x, fittingData.keys()):
            self.participantFit[k].append(fittingData[k])
        for k, v in fittingData["finalParameters"].iteritems():
            self.participantFit[k].append(v)
        for k, v in partModelVars.iteritems():
            self.participantFit[v] = participant[k]

        if self.saveFittingProgress:
            self._makeFittingDataSet(fittingData.copy(), extendedLabel, participant)

    ### Pickle
    def pickleRec(self, data, handle):
        """
        Writes the data to a pickle file

        Parameters
        ----------
        data : object
            Data to be written to the file
        handle : string
            The name of the file
        """

        outputFile = self.newFile(handle, 'pkl')

        with open(outputFile, 'w') as w:
            pickle.dump(data, w)

    def pickleLog(self, results, label=""):
        """
        Stores the data in the appropriate pickle file in a Pickle subfolder
        of the outputting folder

        Parameters
        ----------
        results : dict
            The data to be stored
        label : string, optional
            A label for the results file
        """

        if not self.save:
            return

        # TODO: remove the pulling out of ``Name`` from inside this method and make it more explicit higher up
        name = results["Name"]
        if isinstance(name, basestring):
            handle = 'Pickle/{}'.format(name)
        else:
            raise TypeError("The ``Name`` in the participant data is of type {} and not str".format(type(name)))

        if label:
            handle += label

        self.pickleRec(results, handle)

    ### Excel
    def _simModelLog(self, modelData, simID):

        data = dictData2Lists(modelData)
        record = pd.DataFrame(data)
        name = "data/modelSim_" + simID
        outputFile = self.newFile(name, 'csv')
        record.to_csv(outputFile)
        #outputFile = self.newFile(name, 'xlsx')
        #xlsxT = pd.ExcelWriter(outputFile)
        #record.to_excel(xlsxT, sheet_name='modelLog')
        #xlsxT.save()

    def _makeFittingDataSet(self, fittingData, extendedLabel, participant):

        data = OrderedDict()
        data['folder'] = self.outputFolder
        partFittingKeys, partFittingMaxListLen = listDictKeySet(participant)
        partData = newListDict(partFittingKeys, partFittingMaxListLen, participant, 'part')
        data.update(partData)

        paramFittingDict = copy(fittingData["testedParameters"])
        paramFittingDict['partFitName'] = fittingData.pop("Name")
        #paramFittingDict['fitQuality'] = fittingData.pop("fitQuality")
        #paramFittingDict["fitQualities"] = fittingData.pop("fitQualities")
        for k, v in fittingData.pop("finalParameters").iteritems():
            paramFittingDict[k + "final"] = v
        paramFittingDict.update(fittingData)
        data.update(paramFittingDict)
        recordFittingKeys, recordFittingMaxListLen = listDictKeySet(data)
        recordData = newListDict(recordFittingKeys, recordFittingMaxListLen, data, "")

        record = pd.DataFrame(recordData)

        name = "data/" + extendedLabel
        outputFile = self.newFile(name, 'xlsx')
        xlsxT = pd.ExcelWriter(outputFile)
        # TODO: Remove the engine specification when moving to Python 3
        record.to_excel(xlsxT, sheet_name='ParameterFits', engine='XlsxWriter')
        xlsxT.save()


### Utils
def reframeListDicts(store, storeLabel=''):
    """
    Take a list of dictionaries and turn it into a dictionary of lists

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
        Any dictionary keys containing lists in the input have been split
        into multiple numbered keys

    See Also
    --------
    flatDictKeySet, newFlatDict
    """

    keySet = flatDictKeySet(store)

    # For every key now found
    newStore = newFlatDict(keySet, store, storeLabel)

    return newStore


def reframeSelectListDicts(store, keySet, storeLabel=''):
    """Take a list of dictionaries and turn it into a dictionary of lists
    containing only the useful keys

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys
    keySet : list of strings
        The keys whose data will be included in the return dictionary
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
    flatDictSelectKeySet, newFlatDict

    """

    keySet = flatDictSelectKeySet(store, keySet)

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


def flatDictKeySet(store):
    """
    Generates a dictionary of keys and identifiers for the new dictionary,
    splitting any keys with lists into a set of keys, one for each element
    in the original key.

    These are named <key><location>

    Parameters
    ----------
    store : list of dicts
        The dictionaries would be expected to have many of the same keys.
        Any dictionary keys containing lists in the input have been split
        into multiple numbered keys

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted

    See Also
    --------
    reframeListDicts, newFlatDict
    """

    # Find all the keys
    keySet = OrderedDict()

    for s in store:
        for k in s.iterkeys():
            if k in keySet:
                continue
            v = s[k]
            if isinstance(v, (list, ndarray)):
                listSet, maxListLen = listKeyGen(v, maxListLen=None, returnList=False, abridge=False)
                if listSet is not NoneType:
                    keySet[k] = listSet
            elif isinstance(v, dict):
                dictKeySet, maxListLen = dictKeyGen(v, maxListLen=None, returnList=False, abridge=False)
                keySet[k] = dictKeySet
            else:
                 keySet[k] = None

    return keySet


def flatDictSelectKeySet(store, keys):
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
    keys : list of strings
        The keys whose data will be included in the return dictionary

    Returns
    -------
    keySet : OrderedDict with values of OrderedDict, list or None
        The dictionary of keys to be extracted

    See Also
    --------
    reframeSelectListDicts, newFlatDict
    """

    # Find all the keys
    keySet = OrderedDict()

    for s in store:
        sKeys = (k for k in s.iterkeys() if k in keys)
        for k in sKeys:
            if k in keySet:
                continue
            v = s[k]
            if isinstance(v, (list, ndarray)):
                listSet, maxListLen = listKeyGen(v, maxListLen=None, returnList=False, abridge=True)
                if listSet is not NoneType:
                    keySet[k] = listSet
            elif isinstance(v, dict):
                dictKeySet, maxListLen = dictKeyGen(v, maxListLen=None, returnList=False, abridge=True)
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
    newStore = OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.iteritems():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            subStore = [s[key] for s in store]
            keyStoreSet = newFlatDict(loc, subStore, newKey)
            newStore.update(keyStoreSet)

        elif isinstance(loc, (list, ndarray)):
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

    newStore = OrderedDict()

    if labelPrefix:
        labelPrefix += "_"

    for key, loc in keySet.iteritems():

        newKey = labelPrefix + str(key)

        if isinstance(loc, dict):
            keyStoreSet = newListDict(loc, maxListLen, store[key], newKey)
            newStore.update(keyStoreSet)

        elif isinstance(loc, (list, ndarray)):
            for locCo in loc:
                vals = list(listSelection(store[key], locCo))
                vals = pad(vals, maxListLen)
                newStore[newKey + "_" + str(locCo)] = vals

        else:
            v = store[key]
            if isinstance(v, (list, ndarray)):
                vals = pad(list(v), maxListLen)
            else:
                # We assume the object is a single value or string
                vals = pad([v], maxListLen)
            newStore[newKey] = vals

    return newStore


def pad(vals, maxListLen):
    vLen = size(vals)
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
    keySet = OrderedDict()

    for k in store.keys():
        v = store[k]
        if isinstance(v, (list, ndarray)):
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
        dataShape = list(shape(data))
        dataShapeFirst = dataShape.pop(-1)
        if type(maxListLen) is NoneType:
            maxListLen = dataShapeFirst
        elif dataShapeFirst > maxListLen:
            maxListLen = dataShapeFirst

    else:
        dataShape = shape(data)

    # If we are creating an abridged dataset and the length is too long, skip it. It will just clutter up the document
    if abridge and prod(dataShape) > 10:
        return None, maxListLen

    # We need to calculate every combination of co-ordinates in the array
    arrSets = [range(0, i) for i in dataShape]
    # Now record each one
    locList = [tuple(loc) for loc in listMerGen(*arrSets)]
    listItemLen = len(locList[0])
    if listItemLen == 1:
        returnList = array(locList)#.flatten()
    elif listItemLen == 0:
        return None, maxListLen
    else:
        returnList = locList

    return returnList, maxListLen


def date():
    """
    Calculate today's date as a string in the form <year>-<month>-<day>
    and returns it

    """
    d = dt.datetime(1987, 1, 14)
    d = d.today()
    todayDate = str(d.year) + "-" + str(d.month) + "-" + str(d.day)

    return todayDate
