# -*- coding: utf-8 -*-
"""

@author: Dominic
"""



class outputting(object):

    def __doc__(self):
        """The documentation for the class"""

    def __init__(self,*args,**kwargs):
        """ """

        self.silent = kwargs.get('silent',False)
        self.save = kwargs.get('save', True)
        self.label = kwargs.pop("simLabel","Untitled")
        self.logLevel = kwargs.pop("logLevel",logging.DEBUG)
        self.maxLabelLength = kwargs.pop("maxLabelLength",18)

        self.saving()

        self.fancyLogger()

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


    def _saving(self):

        if self.save:
            self.folderSetup()
            self.logFile = self.outputFolder + "log.txt"
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
                                level = logLevel,
                                filemode= 'w')
        else:
            logging.basicConfig(datefmt='%m-%d %H:%M',
                                    level = logLevel,)

        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logLevel)
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

        logging.info(date)
        logging.info("Log initialised")
        if self.logFile:
            logging.info("The log you are reading was written to " + str(self.logFile))


    def _folderSetup(self):
        """Identifies and creates the folder the data will be stored in

        folderSetup()"""

        # While the folders have already been created, check for the next one
        folderName = './Outputs/' + date + "_" + self.label
        if exists(folderName):
            i = 1
            folderName += '_no_'
            while exists(folderName + str(i)):
                i += 1
            folderName += str(i)

        folderName += "/"
        makedirs(folderName  + 'Pickle/')

        self.outputFolder = folderName


    def __del__(self):
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
            pickleLog(expData,self.outputFolder)
            pickleLog(modelData,self.outputFolder)

        self.expStore.append(expData)
        self.modelStore.append(modelData)

    def getLogger(self, name):

        logger = logging.getLogger(name)

        return logger

    def plotModel(self,modelPlot):
        """ Feeds the model data into the relevant plotting functions for the class """

        mp = modelPlot(self.modelStore[-1:])

        savePlots(mp)

    def plotModelSet(self,modelSetPlot):

        modelSet = self.modelStore[-self.modelSetSize:]
        modelParam = self.modelParamStore[-self.modelSetSize:]
        modelLabel = self.modelLabelStore[-self.modelSetSize:]

        mp = modelSetPlot(modelSet)

        savePlots(mp)

        self.modelSetSize = 0

    def plotExperiment(self, expPlot):
        """ Feeds the experiment data into the relevant plotting functions for the experiment class """

        # Initialise the class
        ep = expPlot(self.expStore[-1:])

        savePlots(ep)

    def savePlots(plots):

        for p in plots:
            # save or display the figures






    decisionTimes = []

    readableLog_paramText = []
    readableLog_paramVals = []
    readableLog_firstDecision = []


        decisionTimes.append(modelResult["firstDecision"])



        if folderName:
            pickleLog(expData,folderName)
            pickleLog(modelResult,folderName,label= paramText)

            # Store the added data for the record set
            readableLog_paramText.append(paramText)
            readableLog_paramVals.append(p)
            readableLog_firstDecision.append(modelResult["firstDecision"])


    message = "Begining output processing"
    logger1.info(message)

    if not silent or save:
        pl = varDynamics(params, paramVals, array(decisionTimes), **plotArgs)
        majFigureSets = (("firstDecision",pl),)
        plots(experiment, folderName, silent, save, label, modelResults, *majFigureSets, **plotArgs)
#        varDynamics(params, paramVals, decisionTimes)

    if save:

        varCategoryDynamics(params, paramVals, array(decisionTimes), folderName)

        simSetLog(readableLog_paramText,
                  params,
                  array(readableLog_paramVals),
                  array(readableLog_firstDecision),
                  expName,
                  modelName,
                  kwargs,
                  label,
                  folderName)





