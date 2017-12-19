# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import cPickle as pickle

from os import listdir
from scipy.io import loadmat
from numpy import array, shape
from itertools import izip, chain
from pandas import read_excel
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
        The file extension found after the ".". Currently only mat and xlsx files are
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

    fileType = fileType
    folder = folder

    files = getFiles(folder, fileType, **kwargs)

    if fileType == "mat":

        dataSet = getmatData(folder, files)

    elif fileType == "xlsx":

        dataSet = getxlsxData(folder, files, **kwargs)

    elif fileType == "pkl":

        dataSet = getpickleData(folder, files, **kwargs)

    else:
        dataSet = []

    return dataSet


def getFiles(folder, fileType, **kwargs):
    """
    Produces the list of valid input files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    fileType : string
        The file extension found after the ".".
    validFiles : list of strings or None, optional
        A list of the names of numbered pickled files to be imported. Default None

    Returns
    -------
    dataFiles : list
        A sorted list of the the files

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

    files = listdir(folder)

    dataFiles = [f for f in files if f.endswith(fileType)]

    if type(validFiles) is NoneType:
        validFileList = dataFiles
    else:
        validFileList = []
        for f in dataFiles:
            for v in validFiles:
                if f.startswith(v):
                    validFileList.append(f)

    sortedFiles = sortStrings(validFileList, "." + fileType)

    return sortedFiles


def sortStrings(unorderedList, suffix):
    """
    Takes an unordered list of strings and sorts them if possible and necessary

    Parameters
    ----------
    unorderedList : list of strings
        A list of valid strings
    suffix : string
        A known suffix for the string

    Returns
    -------
    sorted list : list of strings
        A sorted list of the the strings

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
        return unorderedList

    suffixLen = len(suffix)

    prefix = getUniquePrefix(unorderedList, suffixLen)

    sortedList = intCore(unorderedList, prefix, suffix)
    if sortedList:
        return sortedList
    else:
        return unorderedList

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

    for i in xrange(1, len(unorderedList[0])-suffixLen+2): # Assuming the prefix might be the string-suffix
        sec = unorderedList[0][:i]
        if all((sec == d[:i] for d in unorderedList)):
            continue
        else:
            break
    return unorderedList[0][:i-1]


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
            core = [(d[len(prefix):-(len(suffix))], i) for i, d in enumerate(unorderedList)]
        else:
            core = [(d[len(prefix):], i) for i, d in enumerate(unorderedList)]
        coreInt = [(int(c), i) for c, i in core]
    except:
        return []

    coreSorted = sorted(coreInt)
    coreStr = [(str(c), i) for c, i in coreSorted]

    sortedStrings = [''.join([prefix, '0'*(len(core[i][0])-len(s)), s, suffix]) for s, i in coreStr]

    return sortedStrings


def getmatData(folder, files):
    """
    Loads the data from MATLAB files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames

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

    for f in files:

        mat = loadmat(folder + f)

        dataD = {"fileName": f,
                 "folder": folder}

        for m, v in mat.iteritems():
            if m[0:2] != "__":
                d = array(v)
                if len(shape(d)) != 1:
                    d = d.T[0]
                if len(d) == 1:
                    dataD[m] = d[0]
                else:
                    dataD[m] = d

        dataSets.append(dataD)

    return dataSets


def getxlsxData(folder, files, **kwargs):
    """
    Loads the data from xlsx files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
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

    for f in files:

        # In case the file is open, this will in fact be a temporary file and not a valid file.
        if f.startswith('~$'):
            continue

        dat = read_excel(folder + f, **kwargs)

        if len(splitBy) > 0:
            # The data must be split
            classifierList = (sortStrings(list(set(dat[s])), '') for s in splitBy)
            participants = listMerge(*classifierList)

            for p in participants:

                subDat = dat[(dat[splitBy] == p).all(axis=1)]
                subDatDict = subDat.to_dict(orient='list')
                subDatDict["fileName"] = f
                subDatDict["folder"] = folder
                subDatDict["Name"] = "-".join(p)
                dataSets.append(subDatDict)
        else:
            datDict = dat.to_dict(orient='list')
            datDict["fileName"] = f
            datDict["folder"] = folder
            dataSets.append(datDict)

    return dataSets


def getpickleData(folder, files, **kwargs):
    """
    Loads the data from python pickle files

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    files : list of strings
        A list of filenames
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
        dataSets = getpickledGroupedFiles(folder, files)
    else:
        dataSets = getpickleFiles(folder, files)

    return dataSets


def getpickleFiles(folder, files):
    """
    Loads the data from valid python pickle files

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
    dataSets = []
    for fileName in files:
        dataSets.append(getpickledFileData(folder, fileName))

    return dataSets


def getpickledGroupedFiles(folder, files):
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
    groupedData = []

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
