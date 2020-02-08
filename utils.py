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
import inspect
import imp
import traceback

# TODO: replace imp with importlib when moving to python 3

# For analysing the state of the computer
# import psutil


class ClassNameError(Exception):
    pass


class FunctionNameError(Exception):
    pass


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


def folderSetup(simDescription, path='./'):
    """Identifies and creates the folder the data will be stored in

    Parameters
    ----------
    simDescription : string
        A description of the task

    Returns
    -------
    folder_name : string
        The path to the folder
    """

    # While the folders have already been created, check for the next one
    if path[-1] not in ['\\', '/']:
        outputs = '/Outputs/'
    else:
        outputs = 'Outputs/'
    folder_name = path + outputs + date() + "_" + simDescription
    if os.path.exists(folder_name):
        i = 1
        folder_name += '_no_'
        while os.path.exists(folder_name + str(i)):
            i += 1
        folder_name += str(i)

    folder_name += "/"

    # TODO : remove the automatic construction of Pickle and make it dependent on the saving of Pickle data
    os.makedirs(folder_name + 'Pickle/')

    return folder_name


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


def find_class(class_name, class_folder, inherited_class, excluded_files=None):
    """
    Finds and imports a class from a given folder. Does not look in subfolders

    Parameters
    ----------
    class_name : string
        The name of the class to be used
    class_folder : basestring
        The path where the class is likely to be found
    inherited_class : class
        The class that the searched for class inherits from
    excluded_files : list, optional
        A list of modules to be excluded from the search. Can be described using portions of file names.

    Returns
    -------
    sought_class : inherited_class
        The uninstansiated class sought
    """
    folder_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/{}'.format(class_folder)
    potential_files = [f[:-3] for f in os.listdir(folder_path) if f[-2:] == 'py' and f[0] is not '_']
    if excluded_files:
        potential_files_filtered = [f for f in potential_files if f not in excluded_files]
    else:
        potential_files_filtered = potential_files

    sought_class = None
    for potential_file in potential_files_filtered:
        if sought_class:
            break
        # This is necessary to deal with imp.load_module reloading modules and changing class signatures
        # see https://thingspython.wordpress.com/2010/09/27/another-super-wrinkle-raising-typeerror/
        if potential_file in sys.modules:
            potential_modules = [v for k, v in sys.modules.items() if potential_file in k]
        else:
            file_path = '{}/{}.py'.format(folder_path, potential_file)
            module_info = inspect.getmoduleinfo(file_path)
            with open(file_path) as open_file:
                potential_modules = [imp.load_module(potential_file, open_file, file_path, module_info[1:])]

        for potential_module in potential_modules:
            module_classes = inspect.getmembers(potential_module,
                                                lambda x: inspect.isclass(x)
                                                          and issubclass(x, inherited_class)
                                                          and x.__name__ == class_name
                                                )

            if module_classes and len(module_classes) == 1:
                sought_class = module_classes[0][1]
                break
            elif len(module_classes) > 1:
                raise Exception('This should not have happened.')

    if sought_class:
        return sought_class
    else:
        raise ClassNameError('Unknown {} of class {}'.format(inherited_class, class_name))


def find_function(function_name, function_folder, excluded_files=None):
    """
    Finds and imports a function from a given folder. Does not look in subfolders

    Parameters
    ----------
    function_name : string
        The name of the function to be used
    function_folder : basestring
        The path where the function is likely to be found
    excluded_files : list, optional
        A list of modules to be excluded from the search. Can be described using portions of file names.

    Returns
    -------
    sought_class : inherited_class
        The uninstansiated class sought
    """
    folder_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/{}'.format(function_folder)
    potential_files = [f[:-3] for f in os.listdir(folder_path) if f[-2:] == 'py' and f[0] is not '_']
    if excluded_files:
        potential_files_filtered = [f for f in potential_files if f not in excluded_files]
    else:
        potential_files_filtered = potential_files

    sought_function = None
    for potential_file in potential_files_filtered:
        if sought_function:
            break
        # This is necessary to deal with imp.load_module reloading modules and changing class signatures
        # see https://thingspython.wordpress.com/2010/09/27/another-super-wrinkle-raising-typeerror/
        if potential_file in sys.modules:
            potential_modules = [v for k, v in sys.modules.items() if potential_file in k]
        else:
            file_path = '{}/{}.py'.format(folder_path, potential_file)
            module_info = inspect.getmoduleinfo(file_path)
            with open(file_path) as open_file:
                potential_modules = [imp.load_module(potential_file, open_file, file_path, module_info[1:])]

        for potential_module in potential_modules:
            module_functions = inspect.getmembers(potential_module,
                                                  lambda x: inspect.isfunction(x)
                                                             and x.__name__ == function_name
                                                 )
            if module_functions and len(module_functions) == 1:
                sought_function = module_functions[0][1]
                break
            elif len(module_functions) > 1:
                raise Exception('This should not have happened.')
    if sought_function:
        return sought_function
    else:
        raise FunctionNameError('Unknown function {}'.format(function_name))


def getClassArgs(inspected_class, arg_ignore=['self']):
    """
    Finds the arguments that could be passed into the specified class
    """
    # TODO: when moving to python 3 replace inspect.getargspec with inspect.getfullargspec or inspect.signature
    arg_spec = inspect.getargspec(inspected_class.__init__)
    args = arg_spec.args
    if arg_spec.keywords is not None:
        base_class_arg_spec = inspect.getargspec(inspected_class.__bases__[0].__init__)
        base_args = base_class_arg_spec.args
        new_base_args = [arg for arg in base_args if arg not in args]
        args.extend(new_base_args)

    filtered_args = [arg for arg in args if arg not in arg_ignore]

    return filtered_args


def getClassAttributes(inspected_class, ignore=['self']):
    """
    Finds the public attributes of the specified class
    """

    attributes = [k for k in inspected_class.__dict__.keys() if not k[0]=='_']
    filtered_attributes = [attribute for attribute in attributes if attribute not in ignore]

    return filtered_attributes


def getFuncArgs(inspected_function):
    """
    Finds the arguments that could be passed into the specified function

    :param inspected_function:
    :return:
    """
    arg_spec = inspect.getargspec(inspected_function)
    args = arg_spec.args

    return args


def list_all_equal(data):
    """
    Checks if all of the elements of a list are the same.

    Parameters
    ----------
    data : list of 1D
        The list of elements to compare

    Returns
    -------
    equivalence: bool
        True if the elements are all the same

    Notes
    -----
    Based on https://stackoverflow.com/questions/3844801
    """

    equivalence = data.count(data[0]) == len(data)

    return equivalence


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
    >>> listMergeNP([1, 2, 3], [5, 6, 7]).T
    array([[1, 2, 3, 1, 2, 3, 1, 2, 3], [5, 5, 5, 6, 6, 6, 7, 7, 7]])

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


def listMergeGen(*args):
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
    >>> for i in listMergeGen(0.7): print(repr(i))
    array([0.7])
    >>> for i in listMergeGen([0.7, 0.1]): print(repr(i))
    array([0.7])
    array([0.1])
    >>> for i in listMergeGen([0.7, 0.1], [0.6]): print(repr(i))
    array([0.7,  0.6])
    array([0.1,  0.6])
    >>> for i in listMergeGen([0.7, 0.1], []): print(repr(i))

    >>> for i in listMergeGen([0.7, 0.1], 0.6): print(repr(i))
    array([0.7,  0.6])
    array([0.1,  0.6])
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
    Takes a list of models or tasks and returns a dictionary with only the parameters
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
    >>> data = [{'a':[1, 2, 3], 'b':[7, 8, 9]}, {'b':[4, 5, 6], 'c':'string', 'd':5}]
    >>> mergeDatasets(data)
    {'a': [[1, 2, 3], None], 'c': [None, 'string'], 'b': [[7, 8, 9], [4, 5, 6]], 'd': [None, 5]}
    >>> mergeDatasets(data, extend=True)
    {'a': [1, 2, 3, None], 'c': [None, 'string'], 'b': [7, 8, 9, 4, 5, 6], 'd': [None, 5]}
     >>> data = [{'b': np.array([[7, 8, 9], [1, 2, 3]])}, {'b': np.array([[4, 5, 6], [2, 3, 4]])}]
     >>> mergeDatasets(data, extend = True)
     {'b': [array([7, 8, 9]), array([1, 2, 3]), array([4, 5, 6]), array([2, 3, 4])]}
     >>> mergeDatasets(data)
     {'b': [array([[7, 8, 9], [1, 2, 3]]), array([[4, 5, 6], [2, 3, 4]])]}
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


def flatten(data):
    """
    Yields the elements in order from any N dimensional iterable

    Parameters
    ----------
    data : iterable

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
    for i, v in enumerate(data):
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
    >>> def foo(): print("foo")
    >>> foo.Name = "boo"
    >>> callableDetails(foo)
    ('boo', None)
    >>> foo.Params = {1: 2, 2: 3}
    >>> callableDetails(foo)
    ('boo', {'1': '2', '2': '3'})

    """
    # TODO : clean up this and the functions calling it. This should be unnecessary now
    if isinstance(item, collections.Callable):
        if hasattr(item, 'Name'):
            name = item.Name
        elif hasattr(item, 'get_name'):
            name = item.get_name()
        else:
            raise AttributeError('{} does not have the attribute ``Name`` or ``get_name``'.format(item))

        try:
            details = {str(k): str(v).strip('[]()') for k, v in item.Params.iteritems()}
        except:
            details = None

        return name, details

    else:
        return None, None


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
    >>> def foo(): print("foo")
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
    ['A', 'B', 'e', 'E']
    
    >>> unique(a, lambda x: x.lower())
    ['A', 'B', 'e']
    
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
    # type: (list[float], int, Optional[bool]) -> np.ndarray
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
    array([0.66666667, 1.        , 1.        , 1.        , 0.66666667])
    >>> movingaverage([1, 1, 1, 1, 1, 1, 1, 1], 4)
    array([0.5 , 0.75, 1.  , 1.  , 1.  , 1.  , 1.  , 0.75])
    >>> movingaverage([1, 1, 1, 1, 1], 3, edgeCorrection=True)
    array([1., 1., 1., 1., 1.])
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
    # type: (list) -> np.ndarray
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
    array([1. ,  1.5,  2. ,  2.5])
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
    # type: (list, float) -> np.ndarray
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
    array([1. , 1.5, 2. , 2.5])
    >>> discountAverage([1, 2, 3, 4], 0.25)
    array([1.        , 1.8       , 2.71428571, 3.68235294])

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
    # type: (Union[list, np.ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'.

    Parameters
    ----------
    data : list or np.ndarray
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
	>>> data = np.array([[2., 0., 5., 1.], [3., 3., 3., 4.], [1., 5., 3., 5.], [1., 1., 4., 2.], [2., 4., 5., 1.], [1., 0., 0., 2.]])
    >>> kendalw(data)
    0.22857142857142856

	>>> data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
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


def kendalwt(data, ranked=False):
    # type: (Union[list, np.ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'. Corrects for ties.
    
    Parameters
    ----------
    data : list or np.ndarray
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
    >>> data = np.array([[2., 0., 5., 1.], [3., 3., 3., 4.], [1., 5., 3., 5.], [1., 1., 4., 2.], [2., 4., 5., 1.], [1., 0., 0., 2.]])
    >>> kendalwt(data)
    0.24615384615384617

    >>> data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
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


def kendalwts(data, ranked=False):
    # type: (Union[list, np.ndarray], Optional[bool]) -> float
    """
    Calculates Kendall's W for a n*m array with n items and m 'judges'. Corrects for ties.
    
    Parameters
    ----------
    data : list or np.ndarray
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
    >>> data = np.array([[2., 0., 5., 1.], [3., 3., 3., 4.], [1., 5., 3., 5.], [1., 1., 4., 2.], [2., 4., 5., 1.], [1., 0., 0., 2.]])
    >>> kendalwts(data)
    0.24615384615384617

    >>> data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    >>> kendalwts(data)
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
