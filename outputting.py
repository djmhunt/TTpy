# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

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
from numpy import seterr, seterrcall, array
from itertools import izip

class outputting(object):

    """The documentation for the class"""

    def __init__(self,*args,**kwargs):
        """

        fileName:   The name and path of the file the figure should be saved to. If ommited
                    the file will be saved to a default name.
        saveFig:    If true the figure will be saved.
        silent:     If false the figure will be plotted to the screen. If true the figure
                    will be closed"""

        self.silent = kwargs.get('silent',False)
        self.save = kwargs.get('save', True)
        self.label = kwargs.pop("simLabel","Untitled")
        self.logLevel = kwargs.pop("logLevel",logging.DEBUG)
        self.maxLabelLength = kwargs.pop("maxLabelLength",18)

        self._date()

        self._saving()

        self._fancyLogger()

        self.logger = logging.getLogger('Framework')

        message = "Begining experiment labeled: " + self.label
        self.logger.info(message)

        # Initialise the stores of information

        self.expStore = []
        self.expParamStore = []
        self.expLabelStore = []
        self.modelStore = []
        self.modelParamStore = []
        self.modelLabelStore = []

        self.modelSetSize = 0
        self.expSetSize = 0


    def _saving(self):

        if self.save:
            self.folderSetup()
            self.logFile = self._newFile(self, 'log', '.txt')
        else:
            self.outputFolder = ''
            self.logFile =  ''

    def _fancyLogger(self):
        """
        Sets up the style of logging for all the simulations
        fancyLogger(logLevel, logFile="", silent = False)

        logLevel = [logging.DEBUG|logging.INFO|logging.WARNING|logging.ERROR|logging.CRITICAL]"""

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
        else:
            logging.basicConfig(datefmt='%m-%d %H:%M',
                                    level = self.logLevel)

        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
        console = logging.StreamHandler()
        console.setLevel(self.logLevel)
        console.setFormatter(consoleFormat)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    #    if not silent:
    #        if not self.logFile:
    #            logging.basicConfig(datefmt='%m-%d %H:%M',
    #                                level = logLevel,)
    #        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
    #        console = logging.StreamHandler()
    #        console.setLevel(logLevel)
    #        console.setFormatter(consoleFormat)
    #        # add the handler to the root logger
    #        logging.getLogger('').addHandler(console)

        # Set the standard error output
        sys.stderr = streamLoggerSim(logging.getLogger('STDERR'), logging.ERROR)
        # Set the numpy error output
        seterrcall( streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR) )
        seterr(all='log')

        logging.info(self.date)
        logging.info("Log initialised")
        if self.logFile:
            logging.info("The log you are reading was written to " + str(self.logFile))


    def _folderSetup(self):
        """Identifies and creates the folder the data will be stored in

        folderSetup()"""

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


    def end(self):
        """ """
        message = "Experiment completed. Shutting down"
        self.logger.info(message)

    def recordSimParams(self,expParams,modelParams):
        """Record any parameters that are user specified"""

        expDesc, expPltLabel =  self._params(expParams)
        modelDesc, modelPltLabel =  self._params(modelParams)

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
        self.logger.info(message)

    def _params(self, params):
        """ Processes the parameters of an experiment or model"""

        name = params['Name'] + ": "

        descriptors = [k + ' = ' + str(v).strip('[]()') for k,v in params.iterItems() if k != 'Name']
        descriptor = name + ", ".join(descriptors)

        if len(descriptor)>self.maxLabelLength:
            plotLabel = name + "Group " + str(self.lastLabelID)
            self.lastLabelID += 1
        else:
            plotLabel = descriptor

        return descriptor, plotLabel


    def recordSim(self,expData,modelData):

        message = "Begining output processing"
        self.logger.info(message)

        if self.outputFolder:
            self.pickleLog(expData,self.outputFolder)
            self.pickleLog(modelData,self.outputFolder)

        self.expStore.append(expData)
        self.modelStore.append(modelData)

        self.expSetSize += 1
        self.modelSetSize += 1

    ### Pickled outputs
    def pickleRec(self,data, handle):

        outputFile = _newFile(self, handle, '.pkl')

        with open(outputFile,'w') as w :
            pickle.dump(data, w)

    def pickleLog(self, results,folderName, label=""):

        handle = 'Pickle\\' + results["Name"]

        if label:
            handle += label

        self.pickleRec(results,handle)

    def getLogger(self, name):

        logger = logging.getLogger(name)

        return logger

    def plotModel(self,modelPlot):
        """ Feeds the model data into the relevant plotting functions for the class """

        mp = modelPlot(self.modelStore[-1:])

        self.savePlots(mp)

    def plotModelSet(self,modelSetPlot):

        modelSet = self.modelStore[-self.modelSetSize:]
        modelParams = self.modelParamStore[-self.modelSetSize:]
        modelLabels = self.modelLabelStore[-self.modelSetSize:]

        mp = modelSetPlot(modelSet, modelParams, modelLabels)

        self.savePlots(mp)

        self.modelSetSize = 0

    def plotExperiment(self, expPlot):
        """ Feeds the experiment data into the relevant plotting functions for the experiment class """

        expSet = self.expStore[-self.expSetSize:]
        expParams = self.expParamStore[-self.expSetSize:]
        expLabels = self.expLabelStore[-self.expSetSize:]
        modelSet = self.modelStore[-self.expSetSize:]
        modelParams = self.modelParamStore[-self.expSetSize:]
        modelLabels = self.modelLabelStore[-self.expSetSize:]

        # Initialise the class
        ep = expPlot(expSet, expParams, expLabels, modelSet, modelParams, modelLabels)

        self.savePlots(ep)

        self.expSetSize = 0

    def savePlots(self, plots):

        message = "Produce plots"
        self.logger.info(message)

        for handle, plot in plots:
            if hasattr(plot,"savefig") and callable(getattr(plot,"savefig")):

                fileName = self._newFile(self, handle, '')

                self._outputFig(plot,fileName)

            elif hasattr(plot,"outputTrees") and callable(getattr(plot,"outputTrees")):

                if self.save:
                    fileName = self._newFile(handle, '')

                    plot.outputTrees(fileName)

            elif hasattr(plot,"to_excel") and callable(getattr(plot,"to_excel")):
                outputFile = self._newFile(handle, '.xlsx')

                if self.save:
                    plot.to_excel(outputFile, sheet_name=handle)

    def _outputFig(self, fig, fileName):
        """Saves the figure to a .png file and/or displays it on the screen.
        fig:        MatPlotLib figure object

        self._outputFig(fig)
        """

#        plt.figure(fig.number)

        if self.save:
            ndpi = fig.get_dpi()
            fig.savefig(fileName,dpi=ndpi)

        if not self.silent:
            if not matplotlib.is_interactive():
                fig.show()
            else:
                fig.draw()
        else:
            plt.close(fig)

    def simLog(self):

        data = {'exp_Label': self.expLabelStore,
                'model_Label': self.modelLabelStore,
                'folder': self.outputFolder}

        expData = self._reframeStore(self.expStore, 'exp_')
        modelData = self._reframeStore(self.modelStore, 'model_')

        data.update(expData,modelData)

        record = pd.DataFrame(data)

#        record = record.set_index('sim')

        outputFile = self._newFile(self, 'simRecord', '.xlsx')

        record.to_excel(outputFile, sheet_name='simRecord')

    def _reframeStore(self, store, storeType):
        """Take a list of dictionaries and turn it into a dictionary of lists"""

        # Find all the keys
        keySet = set()
        for s in store:
            keySet = keySet.union(s.keys())

        # For every key
        partStore = {k:[] for k in keySet}
        for key in keySet:
            for s in store:
                v = repr(s.get(key,None))
                partStore[key].append(v)

        newStore = {storeType + k : v for k,v in partStore}

        return newStore

    def _newFile(self, handle, extension):

        fileName = self.outputFolder + handle
        if exists(fileName + extension):
            i = 1
            while exists(fileName + "_" + str(i) + extension):
                i += 1
            fileName += "_" + str(i) + extension

        return fileName

    def _date(self):
        d = dt.datetime(1987, 1, 14)
        d = d.today()
        self.date = str(d.year) + "-" + str(d.month) + "-" + str(d.day)

    if not silent or save:
        pl = varDynamics(params, paramVals, array(decisionTimes), **plotArgs)
        majFigureSets = (("firstDecision",pl),)
        plots(experiment, folderName, silent, save, label, modelResults, *majFigureSets, **plotArgs)







