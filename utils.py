# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

import datetime as dt

import logging
import sys
import collections

from numpy import seterr, seterrcall, meshgrid, array, amax
from itertools import izip, chain
from os import getcwd, makedirs
from os.path import exists
from collections import defaultdict, Callable
from types import NoneType
from sys import exc_info
from traceback import extract_tb


# For analysing the state of the computer
# import psutil

def fancyLogger(logLevel, fileName="", silent = False):
    """
    Sets up the style of logging for all the simulations

    Parameters
    ----------
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        The lowest level to which logging is recorded
    fileName : string, optional
        The filename that the log will be written to. If empty no log will be
        written to a file. Default is empty
    silent : bool, optional
        States if a log is not written to stdout. Defaults to False
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

    logging.info(date())
    logging.info("Log initialised")
    if fileName:
        logging.info("The log you are reading was written to " + str(fileName))


def folderSetup(simType):
    """Identifies and creates the folder the data will be stored in

    Parameters
    ----------
    simType : string
        A description of the experiment

    Returns
    -------
    folderName : string
        The path to the folder
    """

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
    """
    Creates a folder and produces a log filename

    Parameters
    ----------
    save : bool
        If a file is to be saved
    label : string
        A special label to add to the folder name

    Returns
    -------
    folderName : string
        Path for the folder. If save is False, ""
    fileName : string
        Full file name of log.txt with path. If save is False, ""
    """

    if save:
        folderName = folderSetup(label)
        fileName = folderName + "log.txt"
    else:
        folderName = ''
        fileName =  ''

    return folderName, fileName
    
def newFile(handle, extension, outputFolder):
    """
    Creates a new file withe the name <handle> and the extension <extension>

    Parameters
    ----------
    handle : string
        The file name
    extension : string
        The extension of the file
    outputFolder : string
        The full path of where the file will be stored

    Returns
    -------
    fileName : string
        The filename allowed for the file
        
    Examples
    --------
    >>> newFile("handle", "txt", "./")
    './handle.txt'
    
    >>> newFile("handle", "txt", "./Outputting/")
    './Outputting/handle.txt'
    """

    if extension == '':
        end = ''
    else:
        end = "." + extension

    fileName = outputFolder + handle
    if exists(fileName + end):
        i = 1
        while exists(fileName + "_" + str(i) + end):
            i += 1
        fileName += "_" + str(i)

    fileName += end

    return fileName

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
    """For merging lists with objects that are not solely numbers

    Parameters
    ----------
    args : list of lists
        A list of 1D lists of objects

    Returns
    -------
    combinations : array
        An array with len(args) columns and a row for each combination

    Examples
    --------
    >>> utils.listMerge([1,2,3],[5,6,7]).T
    array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
           [5, 5, 5, 6, 6, 6, 7, 7, 7]])


    """

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
    """Fast merging of lists of numbers

    Parameters
    ----------
    args : list of lists of numbers
        A list of 1D lists of numbers

    Returns
    -------
    combinations : array
        An array with len(args) columns and a row for each combination

    Examples
    --------
    >>> utils.listMergeNP([1,2,3],[5,6,7]).T
    array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
           [5, 5, 5, 6, 6, 6, 7, 7, 7]])

    """

    if len(args) == 0:
        return array([[]])

    elif len(args) == 1:
        a = array(args[0])
        r = a.reshape((amax(a.shape),1))

        return r

    else:
        A = meshgrid(*args)

        r = array([i.flatten()for i in A])

        return r.T

def listMerGen(*args):
    """Fast merging of lists of numbers

    Parameters
    ----------
    args : list of lists of numbers
        A list of 1D lists of numbers

    Yields
    ------
    combination : numpy.array of 1 x len(args)
        Array of all combinations

    Examples
    --------
    >>> from utils import listMerGen
    >>> for i in listMerGen(0.7): print(repr(i))
    array([ 0.7])
    >>> for i in listMerGen([0.7,0.1]): print(repr(i))
    array([ 0.7])
    array([ 0.1])
    >>> for i in listMerGen([0.7,0.1],[0.6]): print(repr(i))
    array([ 0.7,  0.6])
    array([ 0.1,  0.6])
    >>> for i in listMerGen([0.7,0.1],[]): print(repr(i))

    >>> for i in listMerGen([0.7,0.1],0.6): print(repr(i))
    array([ 0.7,  0.6])
    array([ 0.1,  0.6])
    """
    if len(args) == 0:
        r = array([[]])
    elif len(args) == 1:
        a = array(args[0])
        if a.shape:
            r = a.reshape((amax(a.shape),1))
        else:
            r = array([[a]])

    else:
        A = meshgrid(*args)

        r = array([i.flatten()for i in A]).T

    for i in r:
        yield i

def varyingParams(intObjects,params):
    """
    Takes a list of models or experiments and returns a dictionary with only the parameters
    which vary and their values
    """

    initDataSet = {param:[i[param] for i in intObjects] for param in params}
    dataSet = {param:val for param,val in initDataSet.iteritems() if val.count(val[0])!=len(val)}

    return dataSet

def mergeDatasetRepr(data, dataLabel=''):
    """
    Take a list of dictionaries and turn it into a dictionary of lists of strings

    Parameters
    ----------
    data : list of dicts containing strings, lists or numbers
    dataLabel : string, optional
        This string will be appended to the front of each key in the new dataset
        Default blank

    Returns
    -------
    newStore : dictionary of lists of strings
        For each key a list will be formed of the string representations of
        each of the former key values.

    """

    # Find all the keys
    keySet = set()
    for s in data:
        keySet = keySet.union(s.keys())

    # For every key
    partStore = defaultdict(list)
    for key in keySet:
        for s in data:
            v = repr(s.get(key,None))
            partStore[key].append(v)

    newStore = {dataLabel + k : v for k,v in partStore.iteritems()}

    return newStore

def mergeDatasets(data, extend = False):
    """
    Take a list of dictionaries and turn it into a dictionary of lists of objects

    Parameters
    ----------
    data : list of dicts containing strings, lists or numbers
    extend : bool, optional
        If lists should be extended rather than appended. Default False

    Returns
    -------
    newStore : dictionary of lists of objects
        For each key a list will be formed of the former key values. If a
        data set did not contain a key a value of None will be entered for it.

    Examples
    --------
    >>> data = [{'a':[1,2,3],'b':[7,8,9]},
                {'b':[4,5,6],'c':'string','d':5}]
    >>> mergeDatasets(data)
    {'a': [[1, 2, 3], None],
     'b': [[7, 8, 9], [4, 5, 6]],
     'c': [None, 'string'],
     'd': [None, 5]}
    >>> mergeDatasets(data, extend = True)
    {'a': [1, 2, 3, None],
     'b': [7, 8, 9, 4, 5, 6],
     'c': [None, 'string'],
     'd': [None, 5]}

     >>> from numpy import array
     >>> data = [{'b':array([[7,8,9],[1,2,3]])}, {'b':array([[4,5,6],[2,3,4]])}]
     >>> mergeDatasets(data, extend = True)
     {'b': [array([7, 8, 9]), array([1, 2, 3]), array([4, 5, 6]), array([2, 3, 4])]}
     >>> mergeDatasets(data)
     {'b': [array([[7, 8, 9],
             [1, 2, 3]]), array([[4, 5, 6],
             [2, 3, 4]])]}


    """

    # Find all the keys
    keySet = set(k for d in data for k in d.keys() )

    # For every key
    newStore = defaultdict(list)
    for key in keySet:
        for d in data:
            dv = d.get(key,None)
            if extend and isinstance(dv, collections.Iterable) and not isinstance(dv, basestring):
                newStore[key].extend(dv)
            else:
                newStore[key].append(dv)

    return dict(newStore)

def date():
    """
    Provides a string of todays date

    Returns
    -------
    date : string
        The string is of the form [year]-[month]-[day]
    """
    d = dt.datetime(1987, 1, 14)
    d = d.today()
    return str(d.year) + "-" + str(d.month) + "-" + str(d.day)

def flatten(l):
    """
    Yields the elements in order from any N dimentional itterable

    Parameters
    ----------
    l : iterable

    Yields
    ------
    ID : (string,list)
        A pair containing the value at each location and the co-ordinates used
        to access them.

    Examples
    --------
    >>> a = [[1,2,3],[4,5,6]]
    >>> for i, loc in flatten(a): print(i,loc)
    1 [0, 0]
    2 [0, 1]
    3 [0, 2]
    4 [1, 0]
    5 [1, 1]
    6 [1, 2]

    """
    for i, v in enumerate(l):
        if isinstance(v, collections.Iterable) and not isinstance(v, basestring):
            for sub, loc in flatten(v):
                yield sub,[i] + loc
        else:
            yield repr(v),[i]

def mergeTwoDicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy

    Assumes different keys in both dictionaries

    Parameters
    ----------
    x : dictionary
    y : dictionary

    Returns
    -------
    mergedDict : dictionary


    """
    mergedDict = x.copy()
    mergedDict.update(y)
    return mergedDict

def mergeDicts(*args):
    """Merges any number of dictionaries with different keys into a new dict

    Precedence goes to key value pairs in latter dicts

    Parameters
    ----------
    args : list of dictionaries

    Returns
    -------
    mergedDict : dictionary

    """
    mergedDict = {}

    for dictionary in args:
        mergedDict.update(dictionary)

    return mergedDict

def callableDetails(item):
    """
    Takes a callable item and extracts the details.

    Currently only extracts things stored in ``item.Name`` and ``item.Params``

    Parameters
    ----------
    item : callable item

    Returns
    -------
    details : tuple pair with string and dictionary of strings
        Contains the properties of the

    Examples
    --------
    >>> from utils import callableDetails
    >>> def foo():
    >>>     print("foo")
    >>>
    >>> foo.Name = "boo"
    >>> callableDetails(foo)
    ('boo', None)

    >>> foo.Params = {1: 2, 2: 3}
    >>> callableDetails(foo)
    ('boo', {'1': '2', '2': '3'})

    """

    if isinstance(item, Callable):
        try:
            details = {str(k): str(v).strip('[]()') for k,v in item.Params.iteritems()}
        except:
            details = None

        return (item.Name,details)

    else:
        return (None, None)

def callableDetailsString(item):
    """
    Takes a callable item and returns a string detailing the function.

    Currently only extracts things stored in ``item.Name`` and ``item.Params``

    Parameters
    ----------
    item : callable item

    Returns
    -------
    description : string
        Contains the properties and name of the callable

    Examples
    --------
    >>> from utils import callableDetailsString
    >>> def foo():
    >>>     print("foo")
    >>>
    >>> foo.Name = "boo"
    >>> callableDetailsString(foo)
    'boo'

    >>> foo.Params = {1: 2, 2: 3}
    >>> callableDetailsString(foo)
    'boo with 1 : 2, 2 : 3'

    """

    Name, details = callableDetails(item)

    if details:
        properties = [k + ' : ' + str(v).strip('[]()') for k,v in details.iteritems()]

        desc = Name + " with " + ", ".join(properties)
    else:
        desc = Name

    return desc

def errorResp():
    """
    Takes an error that has been caught and returns the details as a string

    Returns
    -------
    description : string
        Contains the description of the error

    Examples
    --------
    >>> try:
    >>>     a = 1/0.0
    >>> except:
    >>>     print(errorResp())
    A <type 'exceptions.ZeroDivisionError'> : "float division by zero" in <input> line 1 function <module>: a = 1/0.0
    
    >>> try:
    >>>     a = b()
    >>> except:
    >>>     print(errorResp())
    A <type 'exceptions.NameError'> : "name 'b' is not defined" in <input> line 2 function <module>: a = b()
    
    """
    errorType, value, traceback = exc_info()
    errorLoc = extract_tb(traceback)[-1]
    description = "A " + str(errorType) + ' : "%s"' % (value)
    description += " in " + errorLoc[0] + " line " + str(errorLoc[1])
    description += " function " + errorLoc[2] + ": " + errorLoc[3]

    return description

#if __name__ == '__main__':
#    from timeit import timeit
#    from numpy import fromfunction

#    print(listMerge([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4]))
#    print(listMergeNP([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4]))

#    print(timeit('listMerge([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])', setup="from __main__ import listMerge",number=500000))
#    print(timeit('listMergeNP([1,2,3,4,5,6,7,8,9],[5,6,7,8,9,1,2,3,4])', setup="from __main__ import listMergeNP",number=500000))

#    for p in listMergeNP(array([1,2,3,4,5,6,7,8,9]).T): print(p)
#
#    for p in listMergeNP(fromfunction(lambda i, j: i/40+2.6, (20, 1))): print(p)