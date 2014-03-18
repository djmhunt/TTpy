# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import datetime as dt

import logging
import sys

from numpy import seterr, seterrcall
from os import getcwd, makedirs
from os.path import exists

# For analysing the state of the computer
# import psutil

d = dt.datetime(1987, 1, 14)
d = d.today()
date = str(d.year) + "-" + str(d.month) + "-" + str(d.day)

### +++++ Internal utilities

def fancyLogger(logLevel, fileName="", silent = False):
    """
    Sets up the style of logging for all the simulations
    loggingSetup(logLevel, fileName="", silent = False)

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

    if fileName:
        logging.basicConfig(filename = fileName,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            level = logLevel,
                            filemode= 'w')
    if not silent:
        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logLevel)
        console.setFormatter(consoleFormat)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)


    # Set the standard error output
    sys.stderr = streamLoggerSim(logging.getLogger('STDERR'), logging.ERROR)
    # Set the numpy error output
    seterrcall( streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR) )
    seterr(all='log')

    logging.info(date)
    logging.info("Log initialised")
    if fileName:
        logging.info("The log you are reading was written to " + str(fileName))


def folderSetup(simType):
    """Identifies and creates the folder the data will be stored in

    folderSetup(simType)"""

    # While the folders have already been created, check for the next one
    folderName = './Outputs/' + date + "_" + simType
    if exists(folderName):
        i = 1
        folderName += '_no_'
        while exists(folderName + str(i)):
            i += 1
        folderName += str(i)

    folderName += "/"
    makedirs(folderName  + 'Pickle/')

    return folderName

def saving(save, label):

    if save:
        folderName = folderSetup(label)
        fileName = folderName + "log.txt"
    else:
        folderName = ''
        fileName =  ''

    return folderName, fileName

def argProcess(**kwargs):

    modelArgs = dict()
    expArgs = dict()
    otherArgs = dict()
    for k in kwargs.iterkeys():
        if "m_" in k:
            modelArgs[k.strip("m_")] = kwargs.pop(k)
        elif "e_" in k:
            expArgs[k.strip("e_")] = kwargs.pop(k)
        else:
            otherArgs[k] = kwargs.pop(k)

    return expArgs, modelArgs, otherArgs

def listMerge(*args):

    r=[[]]
    for x in args:
        r = [i+[y] for y in x for i in r]
#        Equivalent to:
#        t = []
#        for y in x:
#            for i in r:
#                t.append(i+[y])
#        r = t

    return r