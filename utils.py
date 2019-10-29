# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import datetime as dt
import numpy as np
import scipy.stats as stats

import logging
import sys
import collections
import os
import traceback

# For analysing the state of the computer
# import psutil


def fancyLogger(logLevel, fileName=""):
    """
    Sets up the style of logging for all the simulations

    Parameters
    ----------
    logLevel : {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        The lowest level to which logging is recorded
    fileName : string, optional
        The filename that the log will be written to. If empty no log will be
        written to a file. Default is empty
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
        logging.basicConfig(filename=fileName,
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
    np.seterrcall(streamLoggerSim(logging.getLogger('NPSTDERR'), logging.ERROR))
    np.seterr(all='log')

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
    if os.path.exists(folderName):
        i = 1
        folderName += '_no_'
        while os.path.exists(folderName + str(i)):
            i += 1
        folderName += str(i)

    folderName += "/"
    os.makedirs(folderName + 'Pickle/')

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
        fileName = ''

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
    if os.path.exists(fileName + end):
        i = 1
        while os.path.exists(fileName + "_" + str(i) + end):
            i += 1
        fileName += "_" + str(i)

    fileName += end

    return fileName


def argProcess(**kwargs):

    modelArgs = dict()
    expArgs = dict()
    otherArgs = dict()
    for k in kwargs.iterkeys():
        if k.startswith("m_"):
            modelArgs[k[2:]] = kwargs.get(k)
        elif k.startswith("e_"):
            expArgs[k[2:]] = kwargs.get(k)
        else:
            otherArgs[k] = kwargs.get(k)

    return expArgs, modelArgs, otherArgs


def listMerge(*args):
    """For merging lists with objects that are not solely numbers

    Parameters
    ----------
    args : list of lists
        A list of 1D lists of objects

    Returns
    -------
    combinations : np.array
        An np.array with len(args) columns and a row for each combination

    Examples
    --------
    >>> listMerge([1, 2, 3], [5, 6, 7]).T
    array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
           [5, 5, 5, 6, 6, 6, 7, 7, 7]])


    """

    r = [[]]
    for x in args:
        r = [i+[y] for y in x for i in r]
#        Equivalent to:
#        t = []
#        for y in x:
#            for i in r:
#                t.append(i+[y])
#        r = t

    return np.array(r)


def listMergeNP(*args):
    """Fast merging of lists of numbers

    Parameters
    ----------
    args : list of lists of numbers
        A list of 1D lists of numbers

    Returns
    -------
    combinations : np.array
        An np.array with len(args) columns and a row for each combination

    Examples
    --------
    >>> utils.listMergeNP([1, 2, 3], [5, 6, 7]).T
    array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
           [5, 5, 5, 6, 6, 6, 7, 7, 7]])

    """

    if len(args) == 0:
        return np.array([[]])

    elif len(args) == 1:
        a = np.array(args[0])
        r = a.reshape((np.amax(a.shape), 1))

        return r

    else:
        A = np.meshgrid(*args)

        r = np.array([i.flatten()for i in A])

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
    >>> for i in listMerGen(0.7): print(repr(i))
    array([ 0.7])
    >>> for i in listMerGen([0.7, 0.1]): print(repr(i))
    array([ 0.7])
    array([ 0.1])
    >>> for i in listMerGen([0.7, 0.1], [0.6]): print(repr(i))
    array([ 0.7,  0.6])
    array([ 0.1,  0.6])
    >>> for i in listMerGen([0.7, 0.1], []): print(repr(i))

    >>> for i in listMerGen([0.7, 0.1], 0.6): print(repr(i))
    array([ 0.7,  0.6])
    array([ 0.1,  0.6])
    """
    if len(args) == 0:
        r = np.array([[]])
    elif len(args) == 1:
        a = np.array(args[0])
        if a.shape:
            r = a.reshape((np.amax(a.shape), 1))
        else:
            r = np.array([[a]])

    else:
        A = np.meshgrid(*args)

        r = np.array([i.flatten() for i in A]).T

    for i in r:
        yield i


def varyingParams(intObjects, params):
    """
    Takes a list of models or experiments and returns a dictionary with only the parameters
    which vary and their values
    """

    initDataSet = {param: [i[param] for i in intObjects] for param in params}
    dataSet = {param: val for param, val in initDataSet.iteritems() if val.count(val[0]) != len(val)}

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
    partStore = collections.defaultdict(list)
    for key in keySet:
        for s in data:
            v = repr(s.get(key, None))
            partStore[key].append(v)

    newStore = {dataLabel + k: v for k, v in partStore.iteritems()}

    return newStore


def mergeDatasets(data, extend=False):
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
    >>> data = [{'a':[1, 2, 3],'b':[7, 8, 9]},
                {'b':[4, 5, 6],'c':'string','d':5}]
    >>> mergeDatasets(data)
    {'a': [[1, 2, 3], None],
     'b': [[7, 8, 9], [4, 5, 6]],
     'c': [None, 'string'],
     'd': [None, 5]}
    >>> mergeDatasets(data, extend=True)
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
    keySet = set(k for d in data for k in d.keys())

    # For every key
    newStore = collections.defaultdict(list)
    for key in keySet:
        for d in data:
            dv = d.get(key, None)
            if extend and isinstance(dv, collections.Iterable) and not isinstance(dv, basestring):
                newStore[key].extend(dv)
            else:
                newStore[key].append(dv)

    return dict(newStore)


def date():
    """
    Provides a string of today's date

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
    Yields the elements in order from any N dimensional iterable

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
    >>> a = [[1, 2, 3],[4, 5, 6]]
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
                yield sub, [i] + loc
        else:
            yield repr(v), [i]


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

    if isinstance(item, collections.Callable):
        try:
            details = {str(k): str(v).strip('[]()') for k, v in item.Params.iteritems()}
        except:
            details = None

        return (item.Name, details)

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
        properties = [k + ' : ' + str(v).strip('[]()') for k, v in details.iteritems()]

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
    errorType, value, tracebackval = sys.exc_info()
    errorLoc = traceback.extract_tb(tracebackval)[-1]
    description = "A " + str(errorType) + ' : "%s"' % (value)
    description += " in " + errorLoc[0] + " line " + str(errorLoc[1])
    description += " function " + errorLoc[2] + ": " + errorLoc[3]

    return description


def unique(seq, idfun=None):
    """
    Finds the unique items in a list and returns them in order found.
    
    Inspired by discussion on ``http://www.peterbe.com/plog/uniqifiers-benchmark``
    Notably f10 Andrew Dalke and f8 by Dave Kirby
    
    Parameters
    ----------
    seq : an iterable object
        The sequence from which the unique list will be compiled
    idfun: function, optional
        A hashing function for transforming the items into the form that is to
        be compared. Default is the ``None``

    Returns
    -------
    result : list
        The list of unique items

    Examples
    --------
    >>> a=list('ABeeE')
    >>> unique(a)
    ['A','B','e','E']
    
    >>> unique(a, lambda x: x.lower())
    ['A','B','e'] 
    
    Note
    ----
    Unless order is needed it is best to use list(set(seq))

    """
    seen = set()
    if idfun is None:
        return [x for x in seq if x not in seen and not seen.add(x)]
    else:
        return [x for x in seq if idfun(x) not in seen and not seen.add(idfun(x))]


def movingaverage(data, windowSize, edgeCorrection=False):
    # type: (List[float], int, Optional[bool]) -> ndarray
    """
    Average over an array

    Parameters
    ----------
    data : list of floats
        The data to average
    windowSize : int
        The size of the window
    edgeCorrection : bool
        If ``True`` the edges are repaired so that there is no unusual dropoff

    Returns
    -------
    convolution : array

    Examples
    --------
    >>> movingaverage([1, 1, 1, 1, 1], 3)
    array([ 0.66666667, 1, 1, 1, 0.66666667])

    >>> movingaverage([1, 1, 1, 1, 1, 1, 1, 1], 4)
    array([ 0.5 ,  0.75,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  0.75])

    >>> movingaverage([1, 1, 1, 1, 1], 3, edgeCorrection=True)
    array([ 1,  1,  1,  1,  1])

    >>> movingaverage([1, 2, 3, 4, 5], 3, edgeCorrection=True)
    array([ 1.5,  2,  3,  4,  4.5])

    >>> movingaverage([1, 1, 1, 1, 1, 1, 1, 1], 4, edgeCorrection=True)
    array([1 ,  1,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1])

    >>> movingaverage([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 7, edgeCorrection=True)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """
    window = np.ones(int(windowSize)) / float(windowSize)
    convolution = np.convolve(data, window, 'same')

    if edgeCorrection and windowSize > 1:
        leftEdge = windowSize // 2
        leftSet = np.arange(leftEdge)
        convolution[:leftEdge] /= ((leftEdge + (windowSize % 2) + leftSet) / windowSize)
        rightEdge = (windowSize - 1) // 2
        rightSet = np.arange(rightEdge, 0, -1)
        convolution[-rightEdge:] /= ((leftEdge + rightSet) / windowSize)

    return convolution


def runningMean(oldMean, newValue, numValues):
    # type: (float, float, int) -> float
    """
    A running mean

    Parameters
    ----------
    oldMean : float
        The old running average mean
    newValue : float
        The new value to be added to the mean
    numValues : int
        The number of values in the new running mean once this value is included

    Returns
    -------
    newMean : float
        The new running average mean

    Notes
    -----
    Based on Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition and taken from
    https://www.johndcook.com/blog/standard_deviation/

    Examples
    --------
    >>> runningMean(1, 2, 2)
    1.5
    >>> runningMean(1.5, 3, 3)
    2.0
    """

    newMean = oldMean + (newValue - oldMean) / numValues

    return newMean


def runningAverage(data):
    # type: (list) -> ndarray
    """
    An accumulating mean

    Parameters
    ----------
    data : list or 1-D array of floats
        The set of values to be averaged

    Returns
    -------
    results : ndArray of length data
        The values from the moving average

    Examples
    --------
    >>> runningAverage([1,2,3,4])
    array([ 1. ,  1.5,  2. ,  2.5])
    """

    count = 2
    results = np.ones(len(data))
    i = data[0]
    results[0] = i
    for n in data[1:]:
        i = runningMean(i, n, count)
        results[count-1] = i
        count += 1

    return results


def discountAverage(data, discount):
    # type: (list, float) -> ndarray
    """
    An accumulating mean

    Parameters
    ----------
    data : list or 1-D array of floats
        The set of values to be averaged
    discount : float
        The value by which each previous value is discounted

    Returns
    -------
    results : ndArray of length data
        The values from the moving average

    Examples
    --------
    >>> discountAverage([1, 2, 3, 4], 1)
    array([ 1,  1.5,  2,  2.5])

    >>> discountAverage([1, 2, 3, 4], 0.25)
    array([ 1,  1.8,  2.71428571,  3.68235294])

    """
    counter = np.arange(0, len(data), 1)
    weights = discount ** counter
    results = np.ones(len(data))
    for c in counter:
        chosenWeights = weights[c::-1]
        weighted = data[:c+1] * chosenWeights
        results[c] = np.sum(weighted) / np.sum(chosenWeights)

    return results


def runningSTD(oldSTD, oldMean, newMean, newValue):
    # type: (float, float, float, float) -> float
    """

    Parameters
    ----------
    oldSTD : float
        The old running average standard deviation
    oldMean : float
        The old running average mean
    newMean : float
        The new running average mean
    newValue : float
        The new value to be added to the mean

    Returns
    -------
    newSTD : float
        The new running average standard deviation

    Notes
    -----
    Based on Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition (which is based on
    B. P. Welford (2012) Note on a Method for Calculating Corrected Sums of Squares and Products, Technometrics,
    4:3, 419-420, DOI: 10.1080/00401706.1962.10490022
    This version is taken from https://www.johndcook.com/blog/standard_deviation/

    Examples
    --------
    >>> runningSTD(0, 1, 1.5, 2)
    0.5

    >>> runningSTD(0.5, 1.5, 2.0, 3)
    2.0
    """

    newSTD = oldSTD + (newValue - oldMean)*(newValue - newMean)

    return newSTD


def kendalw(data, ranked=False):
    # type: (Union[list, ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'.

    Parameters
    ----------
    data : list or ndarray
        The data in the form of an n*m array with n items and m 'judges'
    ranked : bool, optional
        If the data has already been ranked or not. Default ``False``

    Returns
    -------
    w : float
        The Kendall's W

    Notes
    -----
    Based on Legendre, P. (2010). Coefficient of Concordance. In Encyclopedia of Research Design (pp. 164–169). 2455 Teller Road, Thousand Oaks California 91320 United States: SAGE Publications, Inc. http://doi.org/10.4135/9781412961288.n55

    Examples
    --------
	>>> data = array([[2., 0., 5., 1.],
                      [3., 3., 3., 4.],
                      [1., 5., 3., 5.],
                      [1., 1., 4., 2.],
                      [2., 4., 5., 1.],
                      [1., 0., 0., 2.]])
    >>> kendalw(data)
    0.22857

	>>> data = array([[1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3],
                      [4, 4, 4, 4],
                      [5, 5, 5, 5],
                      [6, 6, 6, 6]])
    >>> kendalw(data)
    1.0

    """
    ranks = data
    if not ranked:
        rankVals = []
        for r in data.T:
            rankVals.append(stats.rankdata(r))
        ranks = np.array(rankVals).T

    sranks = np.sum(abs(np.array(ranks)), 1)
    mrank = np.mean(sranks)
    ssdrank = np.sum((sranks-mrank)**2)
    (n,m) = ranks.shape
    w = (12*ssdrank)/((n**3-n)*m**2)
    return w


def kendalwt(data, ranked = False):
    # type: (Union[list, ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'. Corrects for ties.
    
    Parameters
    ----------
    data : list or ndarray
        The data in the form of an n*m array with n items and m 'judges'
    ranked : bool, optional
        If the data has already been ranked or not. Default ``False``

    Returns
    -------
    w : float
        The Kendall's W

    Notes
    -----
    Based on Legendre, P. (2010). Coefficient of Concordance. In Encyclopedia of Research Design (pp. 164–169). 2455 Teller Road, Thousand Oaks California 91320 United States: SAGE Publications, Inc. http://doi.org/10.4135/9781412961288.n55

    Examples
    --------
    >>> data = array([[2., 0., 5., 1.],
                      [3., 3., 3., 4.],
                      [1., 5., 3., 5.],
                      [1., 1., 4., 2.],
                      [2., 4., 5., 1.],
                      [1., 0., 0., 2.]])
    >>> kendalwt(data)
    0.24615

    >>> data = array([[1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3],
                      [4, 4, 4, 4],
                      [5, 5, 5, 5],
                      [6, 6, 6, 6]])
    >>> kendalwt(data)
    1.0
    """
    ranks = data
    if not ranked:
        rankVals = []
        for r in data.T:
            rankVals.append(stats.rankdata(r))
        ranks = np.array(rankVals).T
    
    sranks = np.sum(abs(np.array(ranks)), 1)
    mrank = np.mean(sranks)
    ssdrank = np.sum((sranks-mrank)**2)
    (n, m) = ranks.shape

    T = np.zeros(m)
    for (i, counts) in ((i, collections.Counter(x).most_common()) for i, x in enumerate(ranks.T)):
        for (num, count) in counts:
            if count > 1:
                T[i] += count**3 - count
    
    w1 = 12*ssdrank
    w3 = ((n**3)-n)*(m**2)
    w4 = m*np.sum(T)
    w = w1 / (w3-w4)
    return w


def kendalwts(data, ranked = False):
    # type: (Union[list, ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'. Corrects for ties.
    
    Parameters
    ----------
    data : list or ndarray
        The data in the form of an n*m array with n items and m 'judges'
    ranked : bool, optional
        If the data has already been ranked or not. Default ``False``

    Returns
    -------
    w : float
        The Kendall's W

    Notes
    -----
    Based on Legendre, P. (2010). Coefficient of Concordance. In Encyclopedia of Research Design (pp. 164–169). 2455 Teller Road, Thousand Oaks California 91320 United States: SAGE Publications, Inc. http://doi.org/10.4135/9781412961288.n55

    Examples
    --------
    >>> data = array([[2., 0., 5., 1.],
                      [3., 3., 3., 4.],
                      [1., 5., 3., 5.],
                      [1., 1., 4., 2.],
                      [2., 4., 5., 1.],
                      [1., 0., 0., 2.]])
    >>> kendalws(data)
    0.24615

    >>> data = array([[1, 1, 1, 1],
                      [2, 2, 2, 2],
                      [3, 3, 3, 3],
                      [4, 4, 4, 4],
                      [5, 5, 5, 5],
                      [6, 6, 6, 6]])
    >>> kendalws(data)
    1.0
    """
    ranks = data
    if not ranked:
        rankVals = []
        for r in data.T:
            rankVals.append(stats.rankdata(r))
        ranks = np.array(rankVals).T
    
    sranks = np.sum(abs(np.array(ranks)), 1)
    mrank = np.mean(sranks)
    ssdrank = np.sum((sranks-mrank)**2)
    (n,m) = ranks.shape

    T = np.zeros(m)
    for (i, counts) in ((i, collections.Counter(x).most_common()) for i, x in enumerate(ranks.T)):
        for (num, count) in counts:
            if count > 1:
                T[i] += count**3 - count
    
    w1 = 12*np.sum(sranks**2)
    w2 = 3*n*(m**2)*((n+1)**2)
    w3 = ((n**3)-n)*(m**2)
    w4 = m*np.sum(T)
    w = ((w1-w2)/(w3-w4))

    return w


def kldivergence(m0, m1, c0, c1):
    """
    Calculates the Kullback–Leibler divergence between two distributions using the means and covariances

    Parameters
    ----------
    m0 : array of N floats
        The means of distribution 0
    m1 : array of N floats
        The means of distribution 1
    c0 : NxN array of floats
        The covariance matrix for distribution 0
    c1 : NxN array of floats
        The covariance matrix for distribution 1

    Returns
    -------
    kl : float
        The Kullback–Leibler divergence

    """

    ic0 = np.linalg.inv(c0)
    ic1 = np.linalg.inv(c1)

    cm = np.dot(c1, ic0)
    ldcms, ldcm = np.linalg.slogdet(cm)
    kl = 0.5 * (ldcm + np.trace(np.dot(ic1, np.dot(np.array([m0 - m1]).T, np.array([m0 - m1])) + c0 - c1)))

    return kl


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