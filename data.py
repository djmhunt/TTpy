# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import cPickle as pickle

from os import listdir
from scipy.io import loadmat
from scipy.io.matlab.mio5_params import mat_struct
from numpy import array, shape
from itertools import izip, chain
from pandas import read_excel, read_csv
from types import NoneType
from collections import Iterable, defaultdict, deque
from re import search, finditer

from utils import listMerge


def datasets(folders, fileTypes):
    """
    A function for reading in multiple datasets in one go

    Parameters
    ----------
    folders : list of strings
        The folder strings should end in a "/"
    fileTypes : list of strings
        The file extensions found after the ".". Currently only mat and xlsx files are
        supported.

    Returns
    -------
    dataSet : list of dictionaries

    See Also
    --------
    data : The function called by this one
    """
    dataSetList = []
    for folder, fileType in izip(folders, fileTypes):

        d = data(folder, fileType)

        dataSetList.append(d)

    dataSet = list(chain(*dataSetList))

    return dataSet


def data(folder, fileType, **kwargs):
    """A function for reading in and returning a dataset

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    fileType : string
        The file extension found after the ".". Currently only mat, csv, pkl and xlsx files are
        supported.
    **kwargs : dict
        The keyword arguments used by the

    Returns
    -------
    dataSet : list of dictionaries

    Examples
    --------
    >>> folder = "./Data/"
    >>> fileType = "mat"
    >>> data(folder,fileType)
    [{'cumpts': array([ 7, 17, 19, 21], dtype=uint8)},
     {'cumpts': array([12, 22, 24, 26], dtype=uint8)},
     {'cumpts': array([ 5, 15, 17, 19], dtype=uint8)}]

    """

    groupby = kwargs.pop('groupby', None)

    fileType = fileType
    folder = folder

    files, fileIDs = getFiles(folder, fileType, **kwargs)

    if fileType == "mat":

        dataSet = getmatData(folder, files, fileIDs)

    elif fileType == "csv":

        dataSet = getcsvData(folder, files, fileIDs, **kwargs)

    elif fileType == "xlsx":

        dataSet = getxlsxData(folder, files, fileIDs, **kwargs)

    elif fileType == "pkl":

        dataSet = getpickleData(folder, files, fileIDs, **kwargs)

    else:
        dataSet = []

    if callable(groupby):
        dataSet = groupby(dataSet)

    return dataSet


def getFiles(folder, fileType, **kwargs):
    """
    Produces the list of valid input files

    Parameters
    ----------
    folder : string
        The folder string should end in a ``/``
    fileType : string
        The file extension found after the ``.``.
    validFiles : list of strings or None, optional
        A list of the names of numbered pickled files to be imported. Default None
    terminalID : bool, optional
        Is there an ID number at the end of the filename? If not then a more general search will be performed. Default ``True``

    Returns
    -------
    dataFiles : list
        A sorted list of the the files
    fileIDs : list of strings
        A list of unique parts of the filenames, in the order of dataFiles

    See Also
    --------
    sortStrings : sorts the files found

    Examples
    --------
    >>> folder = "./Data/"
    >>> fileType = "mat"
    >>> getFiles(folder, fileType)
    ['subj1.mat', 'subj2.mat', 'subj11.mat']

    """
    validFiles = kwargs.pop('validFiles', None)
    terminalID = kwargs.pop('terminalID', True)

    files = listdir(folder)

    dataFiles = [f for f in files if f.endswith(fileType)]

    if type(validFiles) is NoneType:
        validFileList = dataFiles
    elif callable(validFiles):
        validFileList = validFiles(dataFiles)
    else:
        # TODO This should be broken out of this and turned into something that can be passed in as a function
        validFileList = []
        for f in dataFiles:
            for v in validFiles:
                if f.startswith(v):
                    validFileList.append(f)

    # TODO This should be broken out of this and turned into something that can be passed in as a function
    sortedFiles, fileIDs = sortStrings(validFileList, "." + fileType, terminalID=terminalID)

    return sortedFiles, fileIDs


def sortStrings(unorderedList, suffix, terminalID=True):
    """
    Takes an unordered list of strings and sorts them if possible and necessary

    Parameters
    ----------
    unorderedList : list of strings
        A list of valid strings
    suffix : string
        A known suffix for the string
    terminalID : bool, optional
        Is there an ID number at the end of the filename? If not then a more general search will be performed. Default ``True``

    Returns
    -------
    sortedList : list of strings
        A sorted list of the the strings
    fileIDs : list of strings
        A list of unique parts of the filenames, in the order of dataFiles

    See Also
    --------
    intCore : sorts the strings with the prefix and suffix removed if they are a number
    getUniquePrefix : identifies prefixes all strings have

    Examples
    --------
    >>> files = ['subj1.mat', 'subj11.mat', 'subj2.mat']
    >>> fileType = ".mat"
    >>> sortStrings(files, fileType)
    ['subj1.mat', 'subj2.mat', 'subj11.mat']
    """
    if len(unorderedList) <= 1:
        return unorderedList, ["all"]

    suffixLen = len(suffix)
    if not terminalID:
        suffix = getUniqueSuffix(unorderedList, suffixLen)
        suffixLen = len(suffix)

    prefix = getUniquePrefix(unorderedList, suffixLen)

    sortedList, fileIDs = intCore(unorderedList, prefix, suffix)
    if not sortedList:
        sortedList, fileIDs = strCore(unorderedList, len(prefix), suffixLen)

    return sortedList, fileIDs


# TODO work out how you want to integrate this into getFiles
def sortbylastnum(dataFiles):

    # sort by the last number on the filename
    footSplit = [search(r"\.(?:[a-zA-Z]+)$", f).start() for f in dataFiles]
    numsplit = [search(r"\d+(\.\d+|$)?$", f[:n]).start() for n, f in izip(footSplit, dataFiles)]

    # check if number part is a float or an int (assuming the same for all) and use the appropriate conversion
    if "." in dataFiles[0][numsplit[0]:footSplit[0]]:
        numRepr = float
    else:
        numRepr = int

    fileNameSections = [(f[:n], numRepr(f[n:d]), f[d:]) for n, d, f in izip(numsplit, footSplit, dataFiles)]

    # Sort the keys for groupFiles
    sortedFileNames = sorted(fileNameSections, key=lambda fileGroup: fileGroup[1])

    dataSortedFiles = [head + str(num) + foot for head, num, foot in sortedFileNames]

    return dataSortedFiles


def getUniqueSuffix(unorderedList, knownSuffixLen):
    """

    Parameters
    ----------
    unorderedList : list of strings
        A list of strings to be ordered
    knownSuffixLen : int
        The length of the suffix identified so far

    Returns
    -------
    suffixLen : int
        The length of the discovered suffix

    """

    for i in xrange(knownSuffixLen, len(unorderedList[0])):  # Starting with the known string-suffix
        sec = unorderedList[0][-i:]
        if all((sec == d[-i:] for d in unorderedList)):
            continue
        else:
            break

    return unorderedList[0][-i + 1:]


def getUniquePrefix(unorderedList, suffixLen):
    """
    Identifies any initial part of strings that are identical
    for all string

    Parameters
    ----------
    unorderedList : list of strings
        A list of strings to be ordered
    suffixLen : int
        The length of the identified suffix

    Returns
    -------
    prefix : string
        The initial part of the strings that is identical for all strings in
        the list

    Examples
    --------
    >>> dataFiles = ['subj1.mat', 'subj11.mat', 'subj2.mat']
    >>> suffixLen = 4
    >>> getUniquePrefix(dataFiles, suffixLen)
    'subj'

    """

    for i in xrange(1, len(unorderedList[0])-suffixLen+2):  # Assuming the prefix might be the string-suffix
        sec = unorderedList[0][:i]
        if all((sec == d[:i] for d in unorderedList)):
            continue
        else:
            break
    return unorderedList[0][:i-1]


def strCore(unorderedList, prefixLen, suffixLen):
    """
    Takes the *core* part of a string and, assuming it is a string,
    sorts them. Returns the list sorted

    Parameters
    ----------
    unorderedList : list of strings
        The list of strings to be sorted
    prefixLen : int
        The length of the unchanging start of each filename
    suffixLen : int
        The length of the unchanging end of each filename

    Returns
    -------
    orderedList : list of strings
        The strings now sorted

    """

    sortingList = ((f, f[prefixLen:-suffixLen]) for f in unorderedList)
    sortedList = sorted(sortingList, key=lambda s: s[1])
    orderedList = [s[0] for s in sortedList]
    fileIDs = [s[1] for s in sortedList]

    return orderedList, fileIDs


def intCore(unorderedList, prefix, suffix):
    """Takes the *core* part of a string and, assuming it is an integer,
    sorts them. Returns the list sorted

    Parameters
    ----------
    unorderedList : list of strings
        The list of strings to be sorted
    prefix : string
        The unchanging part of the start each string
    suffix : string
        The unchanging known end of each string

    Returns
    -------
    sortedStrings : list of strings
        The strings now sorted

    Examples
    --------
    >>> dataFiles = ['me001.mat', 'me051.mat', 'me002.mat', 'me052.mat']
    >>> prefix = 'me0'
    >>> suffix = '.mat'
    >>> intCore(dataFiles,prefix,suffix)
    ['me001.mat', 'me002.mat', 'me051.mat', 'me052.mat']

    >>> dataFiles = ['subj1.mat', 'subj11.mat', 'subj12.mat', 'subj2.mat']
    >>> prefix = 'subj'
    >>> suffix = 'mat'
    >>> intCore(dataFiles,prefix,suffix)
    ['subj1.mat', 'subj2.mat', 'subj11.mat', 'subj12.mat']


    """

    try:
        if suffix:
            testItem = int(unorderedList[0][len(prefix):-len(suffix)])
        else:
            testItem = int(unorderedList[0][len(prefix):])
    except ValueError:
        return [], []

    if suffix:
        core = [(d[len(prefix):-(len(suffix))], i) for i, d in enumerate(unorderedList)]
    else:
        core = [(d[len(prefix):], i) for i, d in enumerate(unorderedList)]
    coreInt = [(int(c), i) for c, i in core]

    coreSorted = sorted(coreInt)
    coreStr = [(str(c), i) for c, i in coreSorted]

    sortedStrings = [''.join([prefix, '0'*(len(core[i][0])-len(s)), s, suffix]) for s, i in coreStr]

    return sortedStrings, [c for c, i in core]


def getmatData(folder, files, fileIDs):
    """
    Loads the data from MATLAB files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
    fileIDs : list of strings
        A list of the changing parts of filenames

    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------
    >>> folder = './Data/'
    >>> dataFiles = ['subj1.mat', 'subj2.mat', 'subj11.mat']
    >>> getmatData(folder, dataFiles)
    [{'cumpts': array([ 7, 17, 19, 21], dtype=uint8)},
     {'cumpts': array([12, 22, 24, 26], dtype=uint8)},
     {'cumpts': array([ 5, 15, 17, 19], dtype=uint8)}]


    """

    dataSets = []

    for f, i in izip(files, fileIDs):

        mat = loadmat(folder + f, struct_as_record=False, squeeze_me=True)

        dataD = {"fileName": f,
                 "fileID": i,
                 "folder": folder}

        for m, v in mat.iteritems():
            if m[0:2] == "__":
                continue
            elif type(v) == mat_struct:
                matstructData = {sk: getattr(v, sk) for sk in v._fieldnames}
                dataD.update(matstructData)
            else:
                dataD[m] = v

        dataSets.append(dataD)

    return dataSets


def getxlsxData(folder, files, fileIDs, **kwargs):
    """
    Loads the data from xlsx files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
    fileIDs : list of strings
        A list of the changing parts of filenames
    splitBy : string or list, optional
        If multiple participants datasets are in one file sheet, this specifies
        the column or columns that can distinguish and identify the rows for
        each participant. Default ``[]``
    **kwargs : dict, optional
        The keyword arguments for pandas.read_excel


    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------

    See Also
    --------
    pandas.read_excel

    """

    splitBy = kwargs.pop('splitBy', [])
    if isinstance(splitBy, basestring):
        splitBy = [splitBy]

    dataSets = []

    for f, i in izip(files, fileIDs):

        # In case the file is open, this will in fact be a temporary file and not a valid file.
        if f.startswith('~$'):
            continue

        dat = read_excel(folder + f, **kwargs)

        if len(splitBy) > 0:
            # The data must be split
            classifierList = (sortStrings(list(set(dat[s])), '')[0] for s in splitBy)
            participants = listMerge(*classifierList)

            for p in participants:

                subDat = dat[(dat[splitBy] == p).all(axis=1)]
                subDatDict = subDat.to_dict(orient='list')
                subDatDict["fileName"] = f
                subDatDict["fileID"] = i
                subDatDict["folder"] = folder
                subDatDict["Name"] = "-".join(p)
                dataSets.append(subDatDict)
        else:
            datDict = dat.to_dict(orient='list')
            datDict["fileName"] = f
            datDict["fileID"] = i
            datDict["folder"] = folder
            dataSets.append(datDict)

    return dataSets


def getcsvData(folder, files, fileIDs, **kwargs):
    """
    Loads the data from csv files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
    fileIDs : list of strings
        A list of the changing parts of filenames
    splitBy : string or list, optional
        If multiple participants datasets are in one file sheet, this specifies
        the column or columns that can distinguish and identify the rows for
        each participant. Default ``[]``
    **kwargs : dict, optional
        The keyword arguments for pandas.read_excel


    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------

    See Also
    --------
    pandas.read_csv

    """

    splitBy = kwargs.pop('splitBy', [])
    if isinstance(splitBy, basestring):
        splitBy = [splitBy]

    dataSets = []

    for f, i in izip(files, fileIDs):

        dat = read_csv(folder + f, **kwargs)

        if len(splitBy) > 0:
            # The data must be split
            classifierList = (sortStrings(list(set(dat[s])), '') for s in splitBy)
            participants = listMerge(*classifierList)

            for p in participants:

                subDat = dat[(dat[splitBy] == p).all(axis=1)]
                subDatDict = subDat.to_dict(orient='list')
                subDatDict["fileName"] = f
                subDatDict["fileID"] = i
                subDatDict["folder"] = folder
                subDatDict["Name"] = "-".join(p)
                dataSets.append(subDatDict)
        else:
            datDict = dat.to_dict(orient='list')
            datDict["fileName"] = f
            datDict["fileID"] = i
            datDict["folder"] = folder
            dataSets.append(datDict)

    return dataSets


def getpickleData(folder, files, fileIDs, **kwargs):
    """
    Loads the data from python pickle files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
    fileIDs : list of strings
        A list of the changing parts of filenames
    groupbyNumber : bool, optional
        Defines if the different valid files should be put together by their number. Default ``False``

    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------

    """

    groupbyNumber = kwargs.pop('groupbyNumber', False)

    if groupbyNumber:
        dataSets = getpickledGroupedFiles(folder, files, fileIDs)
    else:
        dataSets = getpickleFiles(folder, files, fileIDs)

    return dataSets


def getpickleFiles(folder, files, fileIDs):
    """
    Loads the data from valid python pickle files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of the names of numbered pickled files to be imported
    fileIDs : list of strings
        A list of the changing parts of filenames

    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------

    """
    dataSets = []
    for fileName, ID in izip(files, fileIDs):
        fileData = getpickledFileData(folder, fileName)
        fileData["fileID"] = ID
        dataSets.append(fileData)

    return dataSets


def getpickledGroupedFiles(folder, files, fileIDs):
    """
    Loads the data from valid python pickle files and returns them grouped by number

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of the names of numbered pickled files to be imported

    Returns
    -------
    dataSet : list of dictionaries
        Each dictionary should represent the data of one participant

    Examples
    --------

    """
    # TODO: See how this can be shifted to have more flexible sorting
    # group by the last number on the filename
    footSplit = [search(r"\.(?:[a-zA-Z]+)$", f).start() for f in files]
    numsplit = [search(r"\d+(\.\d+|$)?$", f[:n]).start() for n, f in izip(footSplit, files)]

    if "." in files[0][numsplit[0]:footSplit[0]]:
        numRepr = float
    else:
        numRepr = int

    groupedHeaders = defaultdict(list)
    groupedFooters = defaultdict(list)
    for n, d, f in izip(numsplit, footSplit, files):
        groupedHeaders[numRepr(f[n:d])].append(f[:n])
        groupedFooters[numRepr(f[n:d])].append(f[d:])

    # Sort the keys for groupFiles
    sortedGroupNums = sorted(groupedHeaders.iterkeys())

    groupedData = []
    for gn in sortedGroupNums:
        groupData = {}
        headers = groupedHeaders[gn]
        footers = groupedFooters[gn]
        fileNames = [head + str(gn) + foot for head, foot in izip(headers, footers)]

        #find the unique part of the file headers
        headerLen = len(headers[0])
        for i in xrange(headerLen):
            if len(set(h[i] for h in headers)) != 1:
                labelStart = i-1
                break
            if i == headerLen-1:
                labelStart = headerLen
                break
        labels = [h[labelStart:] for h in headers]

        for header, fileName in izip(labels, fileNames):
            groupData.update(getpickledFileData(folder, fileName, header=header))
        groupedData.append(groupData)

    return groupedData


def getpickledFileData(folder, fileName, header=''):
    """
    Loads the data from a python pickle file and returns it as a dictionary if it is not already one

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    validFileList : list of strings
        A list of the names of numbered pickled files to be imported
    header : string, optional
        A prefix to put in front of each dictionary key

    Returns
    -------
    dataSet : dictionary
        A dictionary of the data contained in the pickled file.

    Examples
    --------

    """
    with open(folder + fileName) as o:
        dat = pickle.load(o)
        dat["fileName"] = fileName
        dat["folder"] = folder

        if isinstance(dat, dict):
            finalData = {header + k: v for k, v in dat.iteritems()}
        elif isinstance(dat, Iterable):
            transformedData = dict()
            for n, i in enumerate(dat):
                transformedData[header + str(n)] = i
            finalData = transformedData
        else:
            finalData = {header + "data": dat}

    return finalData
