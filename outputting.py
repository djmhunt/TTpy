# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

import matplotlib
#matplotlib.interactive(True)
import logging
import sys

import matplotlib.pyplot as plt
import cPickle as pickle
import pandas as pd
import datetime as dt

import shutil as shu
from os import getcwd, makedirs
from os.path import isfile, exists
from inspect import stack
from numpy import seterr, seterrcall, array, ndarray, shape, prod, log10, around, size
from itertools import izip
from collections import OrderedDict, Callable
from types import NoneType
from copy import copy, deepcopy

from utils import listMerGen, callableDetailsString
from participants.fitResults import plotFitting
from plotting import paramDynamics, addPoint


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
        Default ``False``
    silent : bool, optional
        States if a log is not written to stdout. Defaults to ``False``
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
    date : Identifies todays date
    saving : Sets up the log file and folder to save results
    fancyLogger : Log creator
    numpy.seterr : The function npErrResp is passed to for defining the response to numpy errors
    """

    def __init__(self, **kwargs):

        self.silent = kwargs.get('silent', False)
        self.save = kwargs.get('save', True)
        self.saveScript = kwargs.get('saveScript', False)
        self.pickleData = kwargs.get('pickleData', False)
        self.simRun = kwargs.get('simRun', False)
        self.label = kwargs.pop("simLabel", "Untitled")
        self.logLevel = kwargs.pop("logLevel", logging.INFO)# logging.DEBUG
        self.maxLabelLength = kwargs.pop("maxLabelLength", 18)
        self.npErrResp = kwargs.pop("npErrResp", 'log')
        self.saveFittingProgress = kwargs.pop("saveFittingProgress", False)

        self.date = date()

        self.saving()

        self.fancyLogger(logFile=self.logFile, logLevel=self.logLevel, npErrResp=self.npErrResp)

        self.logger = logging.getLogger('Framework')
        self.loggerSim = logging.getLogger('Simulation')

        message = "Beginning experiment labelled: " + self.label
        self.logger.info(message)

        # Initialise the stores of information

        self.expStore = []
        self.expParamStore = []
        self.expLabelStore = []
        self.expGroupNum = []
        self.modelStore = []
        self.modelParamStore = []
        self.modelLabelStore = []
        self.modelGroupNum = []
        self.partStore = []
        self.fitInfo = None
        self.fitQualStore = []

        self.modelSetSize = 0
        self.modelsSize = 0
        self.expSetSize = 0
        self.modelSetNum = 0
        self.modelSetsNum = 0
        self.expSetNum = 0

        self.lastExpLabelID = 0
        self.lastModelLabelID = 0

    def end(self):
        """
        To run once everything has been completed. Displays the figures if not
        silent.
        """

        if not self.silent:
            plt.show()

        message = "Experiment completed. Shutting down"
        self.logger.info(message)

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
        if exists(fileName + end):
            i = 1
            while exists(fileName + "_" + str(i) + end):
                i += 1
            fileName += "_" + str(i)

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
                    if cwd in p and "outputting.py" not in p:
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
            # add the handler to the root logger
            logging.getLogger('').addHandler(console)
        else:
            logging.basicConfig(datefmt='%m-%d %H:%M',
                                format='%(name)-12s %(levelname)-8s %(message)s',
                                level=logLevel)

        # Set the standard error output
        sys.stderr = streamLoggerSim(logging.getLogger('STDERR'), logging.ERROR)
        # Set the numpy error output
        seterrcall(streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR))
        seterr(all=npErrResp)

        logging.info(self.date)
        logging.info("Log initialised")
        if logFile:
            logging.info("The log you are reading was written to " + str(logFile))

    def logSimParams(self, expDesc, expPltLabel, modelDesc, modelPltLabel):
        """
        Writes to the log the description and the label of the experiment and model

        Parameters
        ----------
        expDesc : string
            The description to be logged of the experiment
        expPltLabel : string
            The label used for this experiment
        modelDesc : string
            The description to be logged of the model
        modelPltLabel : string
            The label used for this model

        See Also
        --------
        recordSimParams : Records these parameters for later use
        """

        message = "Simulation contains the experiment '" + expDesc + "'"
        if expDesc == expPltLabel:
            message += ". "
        else:
            message += " output with the label '" + expPltLabel + "'. "

        message += "The model used is '" + modelDesc + "'"
        if modelDesc == modelPltLabel:
            message += "."
        else:
            message += " output with the label '" + modelPltLabel + "'."
        self.loggerSim.info(message)

    def logSimFittingParams(self, expParams, modelName, modelFitVars, modelOtherArgs):
        """
        Logs the model and experiment parameters that used as initial fitting
        conditions

        Parameters
        ----------
        expParams : dict
            The experiment parameters
        modelName : string
            The name of the model
        modelFitVars : dict
            The model parameters that will be fitted over and varied.
        modelOtherArgs : dict
            The other parameters used in the model whose attributes have been
            modified by the user
        """
        message = "The fit will use the model '" + modelName + "'"

        modelFitParams = [k + ' around ' + str(v).strip('[]()') for k, v in modelFitVars.iteritems()]
        message += " fitted with the parameters " + ", ".join(modelFitParams)

        modelParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelOtherArgs.iteritems() if not isinstance(v, Callable)]
        modelFuncs = [k + ' = ' + callableDetailsString(v) for k, v in modelOtherArgs.iteritems() if isinstance(v, Callable)]
        message += " and using the other user specified parameters " + ", ".join(modelParams)
        message += " and the functions " + ", ".join(modelFuncs)

        message += ". This is based on the experiment '" + expParams['Name'] + "' "

        expDescriptors = [k + ' = ' + str(v).strip('[]()') for k, v in expParams.iteritems() if k != 'Name']
        message += "with the parameters " + ", ".join(expDescriptors) + "."

        self.loggerSim.info(message)

    def logModFittedParams(self, modelFitVars, modelParams, fitQuality):
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
        """
        params = modelFitVars.keys()

        modelFitParams = [k + ' = ' + str(v).strip('[]()') for k, v in modelParams.iteritems() if k in params]
        message = "The fitted values are " + ", ".join(modelFitParams)

        message += " with a fit quality of " + str(fitQuality) + "."

        self.loggerSim.info(message)

    ### Data collection

    def recordSimParams(self, expParams, modelParams):
        """
        Record the model and experiment parameters

        Parameters
        ----------
        expParams : dict
            The experiment parameters
        modelParams : dict
            The model parameters

        Returns
        -------
        expDesc : string
            The description to be logged of the experiment
        expPltLabel : string
            The label used for this experiment
        modelDesc : string
            The description to be logged of the model
        modelPltLabel : string
            The label used for this model

        See Also
        --------
        paramIdentifier, logSimParams
        """

        expDesc, expPltLabel, lastExpLabelID = self.paramIdentifier(expParams, self.lastExpLabelID)
        modelDesc, modelPltLabel, lastModelLabelID = self.paramIdentifier(modelParams, self.lastModelLabelID)

        self.lastExpLabelID = lastExpLabelID
        self.lastModelLabelID = lastModelLabelID

        self.expLabelStore.append(expPltLabel)
        self.expParamStore.append(expParams)
        self.modelLabelStore.append(modelPltLabel)
        self.modelParamStore.append(modelParams)

        return expDesc, expPltLabel, modelDesc, modelPltLabel

    def paramIdentifier(self, params, lastLabelID):
        """
        Processes the parameters of an experiment or model

        If the description is too long to make a plot label then an ID is generated

        Parameters
        ----------
        params : dict
            The parameters of the experiment or model. One is expected to be ``Name``
        lastLabelID: int
            The last ID number used

        Returns
        -------
        descriptor : string
            The generated description of the model or experiment
        plotLabel : string
            The label to be used for this in plots
        lastLabelID : int
            The last label ID used
        """

        name = params['Name'] + ": "

        descriptors = [k + ' = ' + str(v).strip('[]()') for k, v in params.iteritems() if k != 'Name']
        descriptor = name + ", ".join(descriptors)

        if len(descriptor) > self.maxLabelLength:
            plotLabel = name + "Run " + str(lastLabelID)
            lastLabelID += 1
        else:
            plotLabel = descriptor

        return descriptor, plotLabel, lastLabelID

    def recordSim(self, expData, modelData):
        """
        Records the data from an experiment-model run. Creates a pickled version

        Parameters
        ----------
        expData : dict
            The data from the experiment
        modelData : dict
            The data from the model

        See Also
        --------
        pickleLog : records the picked data
        """

        message = "Beginning simulation output processing"
        self.logger.info(message)

        label = "_Exp-" + str(self.expSetNum) + "_Model-" + str(self.modelSetNum) + "'" + str(self.modelSetSize)

        if self.outputFolder:

            message = "Store data for simulation " + str(self.modelSetSize)
            self.logger.info(message)

            self._simModelLog(modelData)

            if self.pickleData:
                self.pickleLog(expData, label)
                self.pickleLog(modelData, label)

        self.expStore.append(expData)
        self.modelStore.append(modelData)

        self.expGroupNum.append(self.expSetNum)
        self.modelGroupNum.append(self.modelSetNum)

        self.expSetSize += 1
        self.modelsSize += 1
        self.modelSetSize += 1

    def recordParticipantFit(self, participant, expData, modelData, fitQuality=None, fittingData=None):
        """
        Record the data relevant to the participant fitting

        Parameters
        ----------
        participant : dict
            The participant data
        expData : dict
            The data from the experiment
        modelData : dict
            The data from the model
        fitQuality : float, optional
            The quality of the fit as provided by the fitting function
            Default is None
        fittingData : tuple of OrderedDict and list
            The tuple contains an ordered dictionary containing the parameter values tested, in the order they were
            tested, and the fit qualities of these parameters.
        """

        message = "Recording participant model fit"
        self.logger.info(message)

        label = "_Model-" + str(self.modelSetNum) + "_Part-" + str(self.modelSetSize)

        participantName = "Participant " + str(self.modelSetSize)

        participant.setdefault("Name", participantName)

        if self.outputFolder:

            message = "Store data for participant " + str(self.modelSetSize)
            self.logger.info(message)

            ep = plotFitting(participant, modelData, fitQuality)
            self.savePlots(ep)

            if self.saveFittingProgress:
                self.recordFittingSequence(fittingData, fitQuality, label, participant)

            if self.pickleData:
                self.pickleLog(expData, label)
                self.pickleLog(modelData, label)
                self.pickleLog(participant, label)

        self.expStore.append(expData)
        self.modelStore.append(modelData)
        self.partStore.append(participant)
        self.fitQualStore.append(fitQuality)

        self.expGroupNum.append(self.expSetNum)
        self.modelGroupNum.append(self.modelSetNum)

        self.expSetSize += 1
        self.modelsSize += 1
        self.modelSetSize += 1

    def recordFittingParams(self, fitInfo):
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

    ### Ploting
    def recordFittingSequence(self, fittingData, fitQuality, label, participant):

        extendedLabel = "ParameterFits" + label

        paramSet = fittingData[0]
        paramLabels = paramSet.keys()
        fitQualities = fittingData[1]

        self._makeFittingDataSet(paramSet, fitQualities, fitQuality, extendedLabel, participant)

        axisLabels = {"title": "Tested parameters with final fit quality of " + str(around(fitQuality, 1))}

        if len(paramSet) == 1:
            axisLabels["yLabel"] = r'log_{10}(Fit quality)'
            fig = paramDynamics(paramSet,
                                log10(fitQualities),
                                axisLabels)
            addPoint([paramSet[paramLabels[0]][-1]], [fitQuality], fig.axes[0])
        elif len(paramSet) == 2:
            axisLabels["cbLabel"] = r'log_{10}(Fit quality)'
            fig = paramDynamics(paramSet,
                                log10(fitQualities),
                                axisLabels,
                                contour=True,
                                heatmap=False,
                                scatter=True,
                                cmap="viridis")
            addPoint([paramSet[paramLabels[0]][-1]], [paramSet[paramLabels[1]][-1]], fig.axes[0])
        else:
            fig = paramDynamics(paramSet,
                                log10(fitQualities),
                                axisLabels)

        self.savePlots([(extendedLabel, fig)])

    def plotModel(self, modelPlot):
        """
        Feeds the model data into the relevant plotting functions for the class

        Parameters
        ----------
        modelPlot : model.modelPlot
            The model's modelPlot class

        See Also
        --------
        model.modelPlot : The template for modelPlot class for each model
        savePlots : Saves the plots created by modelPlot
        """

        mp = modelPlot(self.modelStore[-1], self.modelParamStore[-1], self.modelLabelStore[-1])

        message = "Produce plots for the model " + self.modelLabelStore[-1]
        self.logger.info(message)

        self.savePlots(mp)

    def plotModelSet(self, modelSetPlot):
        """
        Feeds the model set data into the relevant plotting functions for the class

        Parameters
        ----------
        modelSetPlot : model.modelSetPlot
            The model's modelSetPlot class

        See Also
        --------
        model.modelSetPlot : The template for modelSetPlot class for each model
        savePlots : Saves the plots created by modelSetPlot
        """

        modelSet = self.modelStore[-self.modelSetSize:]
        modelParams = self.modelParamStore[-self.modelSetSize:]
        modelLabels = self.modelLabelStore[-self.modelSetSize:]

        mp = modelSetPlot(modelSet, modelParams, modelLabels)

        message = "Produce plots for model set " + str(self.modelSetNum)
        self.logger.info(message)

        self.savePlots(mp)

        self.modelSetSize = 0
        self.modelSetNum += 1

    def plotExperiment(self, expInput):
        """
        Feeds the experiment data into the relevant plotting functions for the class

        Parameters
        ----------
        expInput : (experiment.experimentPlot, dict)
            The experiment's experimentPlot class and a dictionary of plot
            attributes

        See Also
        --------
        experiment.experimentPlot : The template for experimentPlot class for each experiment
        savePlots : Saves the plots created by experimentPlot
        """

        expPlot, plotArgs = expInput

        expSet = self.expStore[-self.modelsSize:]
        expParams = self.expParamStore[-self.modelsSize:]
        expLabels = self.expLabelStore[-self.modelsSize:]
        modelSet = self.modelStore[-self.modelsSize:]
        modelParams = self.modelParamStore[-self.modelsSize:]
        modelLabels = self.modelLabelStore[-self.modelsSize:]

        # Initialise the class
        ep = expPlot(expSet, expParams, expLabels, modelSet, modelParams, modelLabels, plotArgs)

        message = "Produce plots for experiment set " + str(self.modelSetsNum)
        self.logger.info(message)

        self.savePlots(ep)

        self.modelsSize = 0
        self.modelSetsNum += 1

    def plotExperimentSet(self, expInput):
        """
        Feeds the experiment set data into the relevant plotting functions for the class

        Parameters
        ----------
        expInput : (experiment.experimentSetPlot, dict)
            The experiment's experimentSetPlot class and a dictionary of plot
            attributes

        See Also
        --------
        experiment.experimentSetPlot : The template for experimentSetPlot class for each experiment
        savePlots : Saves the plots created by experimentPlot
        """

        expPlot, plotArgs = expInput

        expSet = self.expStore[-self.expSetSize:]
        expParams = self.expParamStore[-self.expSetSize:]
        expLabels = self.expLabelStore[-self.expSetSize:]
        modelSet = self.modelStore[-self.expSetSize:]
        modelParams = self.modelParamStore[-self.expSetSize:]
        modelLabels = self.modelLabelStore[-self.expSetSize:]

        # Initialise the class
        ep = expPlot(expSet, expParams, expLabels, modelSet, modelParams, modelLabels, plotArgs)

        message = "Produce plots for experiment set " + str(self.expSetNum)
        self.logger.info(message)

        self.savePlots(ep)

        self.expSetSize = 0
        self.expSetNum += 1

    def savePlots(self, plots):
        """
        Saves a list of plots in the appropriate way

        Parameters
        ----------
        plots : list of (string, savable object)
            The currently accepted objects are matplotlib.pyplot.figure,
            pandas.DataFrame and xml.etree.ElementTree.ElementTree
            The string is the handle of the figure

        See Also
        --------
        vtkWriter, plotting, matplotlib.pyplot.figure, pandas.DataFrame,
        pandas.DataFrame.to_excel, xml.etree.ElementTree.ElementTree,
        xml.etree.ElementTree.ElementTree.outputTrees

        """

        for handle, plot in plots:
            if hasattr(plot, "savefig") and callable(getattr(plot, "savefig")):

                fileName = self.newFile(handle, 'png')

                self.outputFig(plot, fileName)

            elif hasattr(plot, "outputTrees") and callable(getattr(plot, "outputTrees")):

                if self.save:
                    fileName = self.newFile(handle, '')

                    plot.outputTrees(fileName)

            elif hasattr(plot, "to_excel") and callable(getattr(plot, "to_excel")):
                outputFile = self.newFile(handle, 'xlsx')

                if self.save:
                    plot.to_excel(outputFile, sheet_name=handle)

    def outputFig(self, fig, fileName):
        """Saves the figure to a .png file and/or displays it on the screen.

        Parameters
        ----------
        fig : MatPlotLib figure object
            The figure to be output
        fileName : string
            The file to be saved to

        """

#        plt.figure(fig.number)

        if self.save:
            ndpi = fig.get_dpi()
            fig.savefig(fileName, dpi=ndpi)

        if not self.silent:
            plt.figure(fig.number)
            plt.draw()
        else:
            plt.close(fig)

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

        handle = 'Pickle/' + results["Name"]

        if label:
            handle += label

        self.pickleRec(results, handle)

    ### Excel
    def simLog(self):
        """
        Outputs relevant data to an excel file with all the data and a csv file
        with the estimated pertinent data
        """

        if not self.save:
            return

        self._abridgedLog()
        self._totalLog()

    def _totalLog(self):

        message = "Produce log of all experiments"
        self.logger.info(message)

        data = self._makeDataSet()
        record = pd.DataFrame(data)
        outputFile = self.newFile('simRecord', 'csv')
        record.to_csv(outputFile)
        outputFile = self.newFile('simRecord', 'xlsx')
        xlsxT = pd.ExcelWriter(outputFile)
        record.to_excel(xlsxT, sheet_name='simRecord')
        xlsxT.save()

    def _abridgedLog(self):

        message = "Produce an abridged log of all experiments"
        self.logger.info(message)

        pertinantData = self._makePertinantDataSet()
        pertRecord = pd.DataFrame(pertinantData)
        outputFile = self.newFile('abridgedRecord', 'csv')
        pertRecord.to_csv(outputFile)
        outputFile = self.newFile('simAbridgedRecord', 'xlsx')
        xlsxA = pd.ExcelWriter(outputFile)
        pertRecord.to_excel(xlsxA, sheet_name='abridgedRecord')
        xlsxA.save()

    def _simModelLog(self, modelData):

        data = dictData2Lists(modelData)
        record = pd.DataFrame(data)
        name = "data/modelSim_" + str(self.modelSetSize)
        outputFile = self.newFile(name, 'csv')
        record.to_csv(outputFile)
        outputFile = self.newFile(name, 'xlsx')
        xlsxT = pd.ExcelWriter(outputFile)
        record.to_excel(xlsxT, sheet_name='modelLog')
        xlsxT.save()

    def _makeDataSet(self):

        data = OrderedDict()
        data['exp_Label'] = self.expLabelStore
        data['model_Label'] = self.modelLabelStore
        data['exp_Group_Num'] = self.expGroupNum
        data['model_Group_Num'] = self.modelGroupNum
        data['folder'] = self.outputFolder

        expData = reframeListDicts(self.expStore, 'exp')
        modelData = reframeListDicts(self.modelStore, 'model')
        if type(self.fitInfo) is not NoneType:
            partData = reframeListDicts(self.partStore, 'part')
            partData['fit_quality'] = self.fitQualStore
            data.update(partData)

        data.update(modelData)
        data.update(expData)

        return data

    def _makePertinantDataSet(self):

        data = OrderedDict()
        data['exp_Label'] = self.expLabelStore
        data['model_Label'] = self.modelLabelStore
        data['exp_Group_Num'] = self.expGroupNum
        data['model_Group_Num'] = self.modelGroupNum

        # Get parameters and fitting data
        modelParams = reframeListDicts(self.modelParamStore)
        modelUsefulParams = OrderedDict((('model_' + k, v) for k, v in modelParams.iteritems() if v.count(v[0]) != len(v)))
        data.update(modelUsefulParams)

        ### Must do this for experiment parameters as well
#        data.update(expData)
        fitInfo = self.fitInfo
        if type(fitInfo) is not NoneType:

            usefulKeys = []
            for fitSet in fitInfo:
                for k, v in fitSet.iteritems():
                    if "Param" in k and "model" or "participant" in k:
                        usefulKeys.append(v)

            modelData = reframeSelectListDicts(self.modelStore, usefulKeys, 'model')
            partData = reframeSelectListDicts(self.partStore, usefulKeys, 'part')
            partData['fit_quality'] = self.fitQualStore
            data.update(modelData)
            data.update(partData)

        return data

    def _makeFittingDataSet(self, paramSet, fitQualities, fitQuality, extendedLabel, participant):

        data = OrderedDict()
        data['exp_Label'] = self.expLabelStore
        data['model_Label'] = self.modelLabelStore
        data['exp_Group_Num'] = self.expGroupNum
        data['model_Group_Num'] = self.modelGroupNum
        data['folder'] = self.outputFolder
        partFittingKeys, partFittingMaxListLen = listDictKeySet(participant)
        partData = newListDict(partFittingKeys, partFittingMaxListLen, participant, 'part')
        data.update(partData)

        paramFittingDict = copy(paramSet)
        paramFittingDict['fitQuality'] = fitQuality
        paramFittingDict["fitQualities"] = fitQualities
        data.update(paramFittingDict)
        recordFittingKeys, recordFittingMaxListLen = listDictKeySet(data)
        recordData = newListDict(recordFittingKeys, recordFittingMaxListLen, data, "")

        record = pd.DataFrame(recordData)
        name = "data/" + extendedLabel
        outputFile = self.newFile(name, 'xlsx')
        xlsxT = pd.ExcelWriter(outputFile)
        record.to_excel(xlsxT, sheet_name='ParameterFits')
        xlsxT.save()

    # def _makeSingleModelDataSet(self):
    #
    #     data = OrderedDict()
    #     modelData = dictArray2Lists(self.modelStore, '')
    #
    #     data.update(modelData)
    #
    #     return data


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




