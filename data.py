# -*- coding: utf-8 -*-
"""
This module allows for the importing of participant data for use in fitting

:Author: Dominic Hunt
"""
from __future__ import division

import pandas as pd

from itertools import izip
from os import listdir
from scipy.io import loadmat
from numpy import array, shape

from utils import listMerGen, mergeDatasets

def data(folder,fileType):
    """A function for reading in and returning a dataset

    Parameters
    ----------
    folder : string
    fileType : string
    
    Returns
    -------
    dataSet : dictionary
    """

    fileType = fileType
    folder = folder

    files = _getFiles(folder, fileType)

    if fileType == "mat":

        dataSet = _getmatData(folder, files)

    return dataSet

def _getFiles(folder, fileType):
    """Produces the list of valid input files"""

    files = listdir(folder)

    dataFiles = _sortFiles(files, fileType)

    return dataFiles
    
def _sortFiles(files, fileType):
    """Identifies valid files and sorts them if possible"""
    
    dataFiles = [f for f in files if f.endswith(fileType) ]
    
    suffixLen = len(fileType)
    
    prefix = _getFilePrefix(dataFiles, suffixLen)
    
    sortedFiles = _floatCore(dataFiles,prefix,fileType)
    if sortedFiles:
        return sortedFiles
    else:
        return dataFiles   
    
def _getFilePrefix(dataFiles, suffixLen):
    """Identifies any initial part of the filenames which is identical 
    for all files"""
    
    for i in xrange(1,len(dataFiles[0])-suffixLen):
        sec = dataFiles[0][:i]
        if all((sec == d[:i] for d in dataFiles)):
            continue
        else:
            break
    return dataFiles[0][:i-1]
    
def _floatCore(dataFiles,prefix,suffix):
    """Takes the *core* part of a filename and, assuming it is a number, 
    sorts them. Returns the filelist sorted"""
    
    try:
        core = [int(d[len(prefix):-(len(suffix)+1)]) for d in dataFiles]
    except:
        return []
        
    core.sort()
    
    sortedFiles = [''.join([prefix,str(c),'.',suffix]) for c in core]
    
    return sortedFiles
    

def _getmatData(folder, files):
    """Loads the data from MATLAB files"""

    dataSets = []
    folder = folder

    for f in files:

        mat = loadmat(folder + f)

        data = {}

        for m, v in mat.iteritems():
            if m[0:2] != "__":
                d = array(v)
                if len(shape(d)) != 1:
                    d = d.T[0]
                if len(d) == 1:
                    data[m] = d[0]
                else:
                    data[m] = d

        dataSets.append(data)

    return dataSets