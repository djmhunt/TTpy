# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
from __future__ import division

import pandas as pd

from os import listdir
from scipy.io import loadmat
from numpy import array, shape, concatenate
from itertools import izip

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
        
    dataSet = concatenate(dataSetList)
        
    return dataSet

def data(folder,fileType):
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

    dataFiles = sortFiles(files, fileType)

    return dataFiles
    
def sortFiles(files, fileType):
    """
    Identifies valid files and sorts them if possible
    
    Parameters
    ----------
    files : list of strings
        A list of filenames
    fileType : string
        The file extension found after the ".".
    
    Returns
    -------
    dataFiles : list
        A sorted list of the the files
        
    See Also
    --------
    intCore : sorts the files found if they are a number
    getFilePrefix : identifies file name prefixes
    
    Examples
    --------
    >>> files = ['Experiment.pptx', 'info.txt', 'subj1.mat', 'subj11.mat', 'subj2.mat']
    >>> fileType = "mat"
    >>> sortFiles(files, fileType)
    ['subj1.mat', 'subj2.mat', 'subj11.mat']
    """
    
    dataFiles = [f for f in files if f.endswith(fileType) ]
    
    suffixLen = len(fileType)
    
    prefix = getFilePrefix(dataFiles, suffixLen)
    
    sortedFiles = intCore(dataFiles,prefix,fileType)
    if sortedFiles:
        return sortedFiles
    else:
        return dataFiles   
    
def getFilePrefix(dataFiles, suffixLen):
    """
    Identifies any initial part of the filenames that is identical 
    for all files
    
    Parameters
    ----------
    dataFiles : list of strings
        A list of filenames
    suffixLen : int
        The length of the file extension found after the ".".
    
    Returns
    -------
    prefix : string
        The initial part of the filenames that is identical for all files
        
    Examples
    --------
    >>> dataFiles = ['subj1.mat', 'subj11.mat', 'subj2.mat']
    >>> suffixLen = 3
    >>> getFilePrefix(dataFiles, suffixLen)
    'subj'

    """
    
    for i in xrange(1,len(dataFiles[0])-suffixLen):
        sec = dataFiles[0][:i]
        if all((sec == d[:i] for d in dataFiles)):
            continue
        else:
            break
    return dataFiles[0][:i-1]
    
def intCore(dataFiles,prefix,suffix):
    """Takes the *core* part of a filename and, assuming it is an integer, 
    sorts them. Returns the file list sorted
    
    Parameters
    ----------
    dataFiles : list of strings
        The complete filenames (without path)
    prefix : string
        The unchanging part of the start each file name
    suffix : string
        The file type
        
    Returns
    -------
    sortedFiles : list of strings
        The filenames now sorted
        
    Examples
    --------
    >>> dataFiles = ['me001.mat', 'me051.mat', 'me002.mat', 'me052.mat']
    >>> prefix = 'me0'
    >>> suffix = 'mat'
    >>> intCore(dataFiles,prefix,suffix)
    ['me001.mat', 'me002.mat', 'me051.mat', 'me052.mat']
    
    >>> dataFiles = ['subj1.mat', 'subj11.mat', 'subj12.mat', 'subj2.mat']
    >>> prefix = 'subj'
    >>> suffix = 'mat'
    >>> intCore(dataFiles,prefix,suffix)
    ['subj1.mat', 'subj2.mat', 'subj11.mat', 'subj12.mat']


    """
    
    try:
        core = [(d[len(prefix):-(len(suffix)+1)],i) for i,d in enumerate(dataFiles)]
    except:
        return []
        
    coreInt = [(int(c),i) for c,i in core]  
    coreSorted = sorted(coreInt)
    coreStr = [(str(c),i) for c,i in coreSorted]
    
    sortedFiles = [''.join([prefix,'0'*(len(core[i][0])-len(s)),s,'.',suffix]) for s,i in coreStr]
    
    return sortedFiles
    

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