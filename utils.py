# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import datetime as dt

import logging
import sys

from numpy import seterr, seterrcall, meshgrid, array, amax
from itertools import izip
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


    if fileName:
        logging.basicConfig(filename = fileName,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            level = logLevel,
                            filemode= 'w')

        consoleFormat = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logLevel)
        console.setFormatter(consoleFormat)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(datefmt='%m-%d %H:%M',
                            format='%(name)-12s %(levelname)-8s %(message)s',
                            level = logLevel)

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
    plotArgs = dict()
    otherArgs = dict()
    for k in kwargs.iterkeys():
        if k.startswith("m_"):
            modelArgs[k[2:]] = kwargs.get(k)
        elif k.startswith("e_"):
            expArgs[k[2:]] = kwargs.get(k)
        elif k.startswith("p_"):
            plotArgs[k[2:]] = kwargs.get(k)
        else:
            otherArgs[k] = kwargs.get(k)

    return expArgs, modelArgs, plotArgs, otherArgs

def listMerge(*args):

    """Obselite? Should be replaced by listMergeNP"""

    r=[[]]
    for x in args:
        r = [i+[y] for y in x for i in r]
#        Equivalent to:
#        t = []
#        for y in x:
#            for i in r:
#                t.append(i+[y])
#        r = t

    return array(r)

def listMergeNP(*args):

    if len(args) == 1:
        a = array(args[0])
        r = a.reshape((amax(a.shape),1))

        return r

    else:
        A = meshgrid(*args)

        r = array([i.flatten()for i in A])

        return r.T

def listMerGen(*args):

    if len(args) == 1:
        a = array(args[0])
        r = a.reshape((amax(a.shape),1))

    else:
        A = meshgrid(*args)

        r = array([i.flatten()for i in A]).T

    for i in r:
        yield i

def varyingParams(intObjects,params):
    """Takes a list of models or experiments and returns a dictionary with only the parameters
    which vary and their values"""

    initDataSet = {param:[i[param] for i in intObjects] for param in params}
    dataSet = {param:val for param,val in initDataSet.iteritems() if val.count(val[0])!=len(val)}

    return dataSet

if __name__ == '__main__':
    from timeit import timeit
    from numpy import fromfunction

#    print listMerge([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])
#    print listMergeNP([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])

#    print timeit('listMerge([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])', setup="from __main__ import listMerge",number=500000)
#    print timeit('listMergeNP([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])', setup="from __main__ import listMergeNP",number=500000)

#    for p in listMergeNP(array([1,2,3,4,5,6,7,8,9]).T): print p
#
#    for p in listMergeNP(fromfunction(lambda i, j: i/40+2.6, (20, 1))): print p