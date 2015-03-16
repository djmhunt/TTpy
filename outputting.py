# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division

import matplotlib
#matplotlib.interactive(True)
import logging
import sys

import matplotlib.pyplot as plt
import cPickle as pickle
import pandas as pd
import datetime as dt

from os import getcwd, makedirs
from os.path import isfile, exists
from numpy import seterr, seterrcall, array, ndarray, shape, prod
from itertools import izip
from collections import OrderedDict

from utils import listMerGen

class outputting(object):

    """An class which manages the outputting to the screen and to files of all 
    data in any form for the simulation
    
    Parameters
    ----------
    save : bool, optional
        If true the data will be saved to files. Default True
    silent : bool, optional
        States if a log is not written to stdout. Defaults to False
    simLabel : string, optional
        The label for the simulation
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        Defines the level of the log. Default logging.INFO
    maxLabelLength : int, optional
        The maximum length of a label to be used as a reference for an 
        individual model-experiment combination. Default 18
        
    See Also
    --------
    date : Identifies todays date
    saving : Sets up the log file and folder to save results
    fancyLogger : Log creator    
    """

    def __init__(self,**kwargs):


        self.silent = kwargs.get('silent',False)
        self.save = kwargs.get('save', True)
        self.label = kwargs.pop("simLabel","Untitled")
        self.logLevel = kwargs.pop("logLevel",logging.INFO)#logging.DEBUG
        self.maxLabelLength = kwargs.pop("maxLabelLength",18)

        self.date()

        self.saving()

        self.fancyLogger()

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
        self.expSetSize = 0
        self.modelSetNum = 0
        self.expSetNum = 0

        self.lastExpLabelID = 1
        self.lastModelLabelID = 1

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
        
        Folder will be created as "./Outputs/<date>_<simLabel>/". If that had 
        previously been created then it is created as
        "./Outputs/<date>_<simLabel>_no_<#>/", where "<#>" is the first
        avalable integer.
        
        A subfolder is also created with the name "Pickle"
        
        See Also
        --------
        newFile : Creates a new file
        saving : Ceates the log system

        """

        # While the folders have already been created, check for the next one
        folderName = './Outputs/' + self.date + "_" + self.label
        if exists(folderName):
            i = 1
            folderName += '_no_'
            while exists(folderName + str(i)):
                i += 1
            folderName += str(i)

        folderName += "/"
        makedirs(folderName  + 'Pickle/')

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
        else:
            self.outputFolder = ''
            self.logFile =  ''

    def fancyLogger(self):
        """
        Sets up the style of logging for all the simulations        
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


        if self.logFile:
            logging.basicConfig(filename = self.logFile,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                level = self.logLevel,
                                filemode= 'w')

            consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
            console = logging.StreamHandler()
            console.setLevel(self.logLevel)
            console.setFormatter(consoleFormat)
            # add the handler to the root logger
            logging.getLogger('').addHandler(console)
        else:
            logging.basicConfig(datefmt='%m-%d %H:%M',
                                format='%(name)-12s %(levelname)-8s %(message)s',
                                level = self.logLevel)

        # Set the standard error output
        sys.stderr = streamLoggerSim(logging.getLogger('STDERR'), logging.ERROR)
        # Set the numpy error output
        seterrcall( streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR) )
        seterr(all='log')#'raise')#'log')

        logging.info(self.date)
        logging.info("Log initialised")
        if self.logFile:
            logging.info("The log you are reading was written to " + str(self.logFile))

    ### Data collection

    def recordSimParams(self,expParams,modelParams):
        """
        Record any parameters that are user specified
        
        Parameters
        ----------
        expParams : dict
            The experiment parameters
        modelParams : dict
            The model parameters
        """

        expDesc, expPltLabel, lastExpLabelID =  self._params(expParams, self.lastExpLabelID)
        modelDesc, modelPltLabel, lastModelLabelID =  self._params(modelParams, self.lastModelLabelID)

        self.lastExpLabelID = lastExpLabelID
        self.lastModelLabelID = lastModelLabelID

        self.expLabelStore.append(expPltLabel)
        self.expParamStore.append(expParams)
        self.modelLabelStore.append(modelPltLabel)
        self.modelParamStore.append(modelParams)

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

    def _params(self, params, lastLabelID):
        """ Processes the parameters of an experiment or model"""

        name = params['Name'] + ": "

        descriptors = [k + ' = ' + str(v).strip('[]()') for k,v in params.iteritems() if k != 'Name']
        descriptor = name + ", ".join(descriptors)

        if len(descriptor)>self.maxLabelLength:
            plotLabel = name + "Run " + str(lastLabelID)
            lastLabelID += 1
        else:
            plotLabel = descriptor

        return descriptor, plotLabel, lastLabelID

    def recordSim(self,expData,modelData):
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

        message = "Beginning output processing"
        self.logger.info(message)

        label = "_Exp-" + str(self.expSetNum) + "_Model-" + str(self.modelSetNum) + "'" + str(self.modelSetSize)

        if self.outputFolder:
            self.pickleLog(expData,self.outputFolder,label)
            self.pickleLog(modelData,self.outputFolder,label)

        self.expStore.append(expData)
        self.modelStore.append(modelData)

        self.expGroupNum.append(self.expSetNum)
        self.modelGroupNum.append(self.modelSetNum)

        self.expSetSize += 1
        self.modelSetSize += 1

    def recordParticipantFit(self, participant, expData, modelData, fitQuality = None):
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
        """

        message = "Recording participant model fit"
        self.logger.info(message)

        label = "_Model-" + str(self.modelSetNum) + "_Part-" + str(self.modelSetSize)

        participant.setdefault("Name","Participant" + str(self.modelSetSize))

        if self.outputFolder:
            self.pickleLog(expData,self.outputFolder,label)
            self.pickleLog(modelData,self.outputFolder,label)
            self.pickleLog(participant,self.outputFolder,label)

        self.expStore.append(expData)
        self.modelStore.append(modelData)
        self.partStore.append(participant)
        self.fitQualStore.append(fitQuality)

        self.expGroupNum.append(self.expSetNum)
        self.modelGroupNum.append(self.modelSetNum)

        self.expSetSize += 1
        self.modelSetSize += 1

    def recordFittingParams(self,fitInfo):
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

            for k,v in f.iteritems():
                if k == "Name":
                    continue

                message = k + ": " + repr(v)
                log.info(message)


    ### Ploting
    def plotModel(self,modelPlot):
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

    def plotModelSet(self,modelSetPlot):
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
        plots : list of savable objects
            The currently accepted objects are matplotlib.pyplot.figure, 
            pandas.DataFrame and xml.etree.ElementTree.ElementTree
            
        See Also
        --------
        vtkWriter, plotting, matplotlib.pyplot.figure, pandas.DataFrame, 
        pandas.DataFrame.to_excel, xml.etree.ElementTree.ElementTree, 
        xml.etree.ElementTree.ElementTree.outputTrees
        
        """

        for handle, plot in plots:
            if hasattr(plot,"savefig") and callable(getattr(plot,"savefig")):

                fileName = self.newFile(handle, '')

                self.outputFig(plot,fileName)

            elif hasattr(plot,"outputTrees") and callable(getattr(plot,"outputTrees")):

                if self.save:
                    fileName = self.newFile(handle, '')

                    plot.outputTrees(fileName)

            elif hasattr(plot,"to_excel") and callable(getattr(plot,"to_excel")):
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
            fig.savefig(fileName,dpi=ndpi)

        if not self.silent:
            plt.figure(fig.number)
            plt.draw()
        else:
            plt.close(fig)

    ### Pickle
    def pickleRec(self,data, handle):
        """
        Writes the data to a picke file 
        
        Parameters
        ----------
        data : object
            Data to be written to the file
        handle : string
            The name of the file
        """

        outputFile = self.newFile(handle, 'pkl')

        with open(outputFile,'w') as w :
            pickle.dump(data, w)

    def pickleLog(self, results,folderName, label=""):
        """
        Stores the data in the appropriate pickle file in a Pickle subfolder 
        of the outputting folder
        
        Parameters
        ----------
        results : dict
            The data to be stored
        folderName : string
            The path to the outputting folder
        label : string, optional
            A label for the results file
        """

        if not self.save:
            return

        handle = 'Pickle/' + results["Name"]

        if label:
            handle += label

        self.pickleRec(results,handle)

    ### Excel
    def simLog(self):
        """
        Outputs relavent data to an excel file with all the data and a csv file
        with the estimated pertinant data
        """

        if not self.save:
            return

        message = "Produce log of all experiments"
        self.logger.info(message)

        data = self._makeDataSet()
        pertinantData = self._makePertinantDataSet()

        record = pd.DataFrame(data)
        pertRecord = pd.DataFrame(pertinantData)

#        record = record.set_index('sim')

        outputFile = self.newFile('simRecord', 'csv')
        record.to_csv(outputFile)

        outputFile = self.newFile('abridgedRecord', 'csv')
        pertRecord.to_csv(outputFile)

        outputFile = self.newFile('simRecord', 'xlsx')
        xlsx = pd.ExcelWriter(outputFile)
        record.to_excel(xlsx, sheet_name='simRecord')
        pertRecord.to_excel(xlsx,sheet_name = 'abridgedRecord')
        xlsx.save()

    def _makeDataSet(self):

        data = OrderedDict()
        data['exp_Label'] = self.expLabelStore
        data['model_Label'] = self.modelLabelStore
        data['exp_Group_Num'] = self.expGroupNum
        data['model_Group_Num'] = self.modelGroupNum
        data['folder'] = self.outputFolder

        expData = self.reframeStore(self.expStore, 'exp_')
        modelData = self.reframeStore(self.modelStore, 'model_')
        if self.fitInfo != None:
            partData = self.reframeStore(self.partStore, 'part_')
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
        modelParams = self.reframeStore(self.modelParamStore)
        modelUsefulParams = OrderedDict((('model_' + k,v) for k, v in modelParams.iteritems() if v.count(v[0]) != len(v)))
        data.update(modelUsefulParams)

        ### Must do this for experiment parameters as well
#        data.update(expData)
        fitInfo = self.fitInfo
        if fitInfo != None:

            usefulKeys = []
            for fitSet in fitInfo:
                for k,v in fitSet.iteritems():
                    if "Param" in k and "model" or "participant" in k:
                        usefulKeys.append(v)

            modelData = self.reframeSelectStore(self.modelStore, usefulKeys, 'model_')
            partData = self.reframeSelectStore(self.partStore, usefulKeys, 'part_')
            partData['fit_quality'] = self.fitQualStore
            data.update(modelData)
            data.update(partData)


        return data


    ### Utils
    def reframeStore(self, store, storeLabel = ''):
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
        dictKeySet, newDict
        """

        keySet = self.dictKeySet(store)

        # For every key now found
        newStore = self.newDict(keySet,store,storeLabel)

        return newStore

    def reframeSelectStore(self, store, keySet, storeLabel = ''):
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
        dictSelectKeySet, newDict
        
        """

        keySet = self.dictSelectKeySet(store,keySet)

        # For every key now found
        newStore = self.newDict(keySet,store,storeLabel)

        return newStore

    def dictKeySet(self,store):
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
        keySet : dict
            The keys are the keys for the new dictionary. The values contain a 
            two element tuple. The first element is the original name of the 
            key and the second is the location of the value to be stored in the 
            original dictionary value array.
            
        See Also
        --------
        reframeStore, newDict
        """

        # Find all the keys
        keySet = OrderedDict()

        for s in store:
            for k in s.keys():
                v = s[k]
                if isinstance(v, (list,ndarray)):
                    #We need to calculate every combination of co-ordinates in the array
                    arrSets = [range(0,i) for i in shape(v)]
                    # Now record each one
                    for genLoc in listMerGen(*arrSets):
                        if len(genLoc) == 1:
                            loc = genLoc[0]
                        else:
                            loc = tuple(genLoc)
                        keySet.setdefault(k+str(loc), (k, loc))
                else:
                    keySet.setdefault(k, (None, None))

        return keySet

    def dictSelectKeySet(self,store, keys):
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
        keySet : dict
            The keys are the keys for the new dictionary. The values contain a 
            two element tuple. The first element is the original name of the 
            key and the second is the location of the value to be stored in the 
            original dictionary value array.
            
        See Also
        --------
        reframeSelectStore, newDict
        """

        # Find all the keys
        keySet = OrderedDict()

        for s in store:
            sKeys = (k for k in s.iterkeys() if k in keys)
            for k in sKeys:
                v = s[k]
                if isinstance(v, (list,ndarray)):
                    vShape = shape(v)
                    # If the length is too long, skip it. It will just clutter up the document
                    if prod(vShape) > 10:
                        continue
                    #We need to calculate every combination of co-ordinates in the array
                    arrSets = [range(0,i) for i in vShape]
                    # Now record each one
                    for genLoc in listMerGen(*arrSets):
                        if len(genLoc) == 1:
                            loc = genLoc[0]
                        else:
                            loc = tuple(genLoc)
                        keySet.setdefault(k+str(loc), (k, loc))
                else:
                    keySet.setdefault(k, (None, None))

        return keySet

    def newDict(self,keySet,store,storeLabel):
        """
        Takes a list of dictionaries and returns a dictionary of 1D lists. 
        
        If a dictionary did not have that key or list element, then 'None' is 
        put in its place
        
        Parameters
        ----------
        keySet : dict
            The keys are the keys for the new dictionary. The values contain a 
            two element tuple. The first element is the original name of the 
            key and the second is the location of the value to be stored in the 
            original dictionary value array.
        store : list of dicts
            The dictionaries would be expected to have many of the same keys. 
            Any dictionary keys containing lists in the input have been split 
            into multiple numbered keys
        storeLabel : string
            An identifier to be added to the beginning of each key string.
            
        
        Returns
        -------
        newStore : dict
            The new dictionary with the keys from the keySet and the values as 
            1D lists with 'None' if the keys, value pair was not found in the 
            store.
        
        """

        partStore = OrderedDict()

        for key, (initKey, loc) in keySet.iteritems():

            partStore.setdefault(key,[])

            if initKey == None:
                vals = [repr(s.get(key,None)) for s in store]
                partStore[key].extend(vals)

            else:
                for s in store:
                    rawVal = s.get(initKey,None)
                    if rawVal == None:
                        v = None
                    else:
                        v = rawVal[loc]
                    partStore[key].append(v)

        newStore = OrderedDict(((storeLabel + k, v) for k,v in partStore.iteritems()))

        return newStore

    def date(self):
        """
        Calculate todays date as a string in the form <year>-<month>-<day>
        and stores it in self.date
        
        """
        d = dt.datetime(1987, 1, 14)
        d = d.today()
        self.date = str(d.year) + "-" + str(d.month) + "-" + str(d.day)







