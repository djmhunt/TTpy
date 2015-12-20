# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
from __future__ import division, print_function

from os import listdir
from scipy.io import loadmat
from numpy import array, shape
from itertools import izip, chain
from pandas import read_excel
from types import NoneType

from utils import listMerge

def datasets(folders, fileTypes):
    """
    A function for reading in multiple datasets in one go
    
    Parameters
    ----------
    folders : list of strings
        The folder strings should end in a "/"
    fileType : list of strings
        The file extensions found after the ".". Currently only mat files are 
        supported.
    
    Returns
    -------
    dataSet : list of dictionaries
    
    See Also
    --------
    data : The function called by this one
    """
    dataSetList = []
    for folder, fileType in izip(folders,fileTypes):
        
        d = data(folder,fileType)
        
        dataSetList.append(d)
        
    dataSet = list(chain(*dataSetList))
        
    return dataSet

def data(folder,fileType, **kwargs):
    """A function for reading in and returning a dataset

    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    fileType : string
        The file extension found after the ".". Currently only mat files are 
        supported.
    
    Returns
    -------
    dataSet : list of dictionaries
    
    See Also
    --------
    data: The
    
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

    files = getFiles(folder, fileType)

    if fileType == "mat":

        dataSet = getmatData(folder, files)
        
    elif fileType == "xlsx":
        
        dataSet = getxlsxData(folder, files, **kwargs)

    return dataSet

def getFiles(folder, fileType):
    """
    Produces the list of valid input files
    
    Parameters
    ----------
    folder : string
        The folder string should end in a "/"
    fileType : string
        The file extension found after the ".".
    
    Returns
    -------
    dataFiles : list
        A sorted list of the the files
        
    See Also
    --------
    sortFiles : sorts the files found
    
    Examples
    --------
    >>> folder = "./Data/"
    >>> fileType = "mat"
    >>> getFiles(folder, fileType)
    ['subj1.mat', 'subj2.mat', 'subj11.mat']

    """

    files = listdir(folder)
    
    dataFiles = [f for f in files if f.endswith(fileType) ]

    sortedFiles = sortFiles(dataFiles, fileType)

    return sortedFiles
    
def sortFiles(files, fileType):
    """
    Takes valid files and sorts them if possible and necessary
    
    Parameters
    ----------
    files : list of strings
        A list of valid filenames
    fileType : string
        The file extension found after the ".".
    
    Returns
    -------
    dataFiles : list
        A sorted list of the the files
        
    See Also
    --------
    intCore : sorts the files found if they are a number
    getPrefix : identifies file name prefixes
    
    Examples
    --------
    >>> files = ['subj1.mat', 'subj11.mat', 'subj2.mat']
    >>> fileType = "mat"
    >>> sortFiles(files, fileType)
    ['subj1.mat', 'subj2.mat', 'subj11.mat']
    """

    suffixLen = len(fileType)+1 # the +1 for the dot
    
    prefix = getPrefix(files, suffixLen)
    
    sortedFiles = intCore(files,prefix,fileType)
    if sortedFiles:
        return sortedFiles
    else:
        return files   
    
def getPrefix(unorderedList, suffixLen):
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
    >>> getPrefix(dataFiles, suffixLen)
    'subj'

    """
    
    for i in xrange(1,len(unorderedList[0])-suffixLen+2): # Assuming the prefix might be the string-suffix
        sec = unorderedList[0][:i]
        if all((sec == d[:i] for d in unorderedList)):
            continue
        else:
            break
    return unorderedList[0][:i-1]
    
def intCore(unorderedList,prefix,suffix):
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
        core = [(d[len(prefix):-(len(suffix))],i) for i,d in enumerate(unorderedList)]
    except:
        return []
        
    coreInt = [(int(c),i) for c,i in core]  
    coreSorted = sorted(coreInt)
    coreStr = [(str(c),i) for c,i in coreSorted]
    
    sortedStrings = [''.join([prefix,'0'*(len(core[i][0])-len(s)),s,suffix]) for s,i in coreStr]
    
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
    >>> getmatData(folder, files)
    [{'cumpts': array([ 7, 17, 19, 21], dtype=uint8)},
     {'cumpts': array([12, 22, 24, 26], dtype=uint8)},
     {'cumpts': array([ 5, 15, 17, 19], dtype=uint8)}]

    
    """

    dataSets = []
    folder = folder

    for f in files:

        mat = loadmat(folder + f)

        dataD = {}
        
        dataD["fileName"] = f

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
    >>> folder = './Data/'
    >>> dataFiles = ['subj1.mat', 'subj2.mat', 'subj11.mat']
    >>> getmatData(folder, files)
    [{'cumpts': array([ 7, 17, 19, 21], dtype=uint8)},
     {'cumpts': array([12, 22, 24, 26], dtype=uint8)},
     {'cumpts': array([ 5, 15, 17, 19], dtype=uint8)}]
     
    See Also
    --------
    pandas.read_excel

    
    """
    
    splitBy = kwargs.pop('splitBy',[])
    if isinstance(splitBy, str):
        splitBy = [splitBy]
    
    dataSets = []
    folder = folder

    for f in files:

        dat = read_excel(folder + f, **kwargs)
        
        if len(splitBy) > 0:
            # The data must be split
            participants = listMerge((list(set(dat[s])).sort() for s in splitBy))
                
            for p in participants:
                
                subDat = dat[(dat[splitBy] == p).all(axis=1)]
                subDatDict = subDat.to_dict(orient='list')
                subDatDict["fileName"] = f
                dataSets.append(subDatDict)
        else:
            datDict = dat.to_dict(orient='list')
            datDict["fileName"] = f
            dataSets.append(datDict)
            
    return dataSets