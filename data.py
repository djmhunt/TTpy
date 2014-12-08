# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import pandas as pd

from itertools import izip
from os import listdir
from scipy.io import loadmat
from numpy import array, shape

from utils import listMerGen, mergeDatasets

def data(folder,fileType):

    """A function for reading in and returning a dataset

    dataSet = data(folder,fileType)

    folder : string
    fileType: string
    dataSet : pandas dataframe
    """

    fileType = fileType
    folder = folder

    files = _getFiles(folder, fileType)

    if fileType == "mat":

        dataSet = _getmatData(folder, files)

    return dataSet

def _getFiles(folder, fileType):

    files = listdir(folder)

    dataFiles = _sortFiles(files, fileType)

    return dataFiles
    
def _sortFiles(files, fileType):
    
    dataFiles = [f for f in files if f.endswith(fileType) ]
    
    suffixLen = len(fileType)
    
    prefix = _getFilePrefix(dataFiles, suffixLen)
    
    sortedFiles = _floatCore(dataFiles,prefix,fileType)
    if sortedFiles:
        return sortedFiles
    else:
        return dataFiles   
    
def _getFilePrefix(dataFiles, suffixLen):
    
    for i in xrange(1,len(dataFiles[0])-suffixLen):
        sec = dataFiles[0][:i]
        if all((sec == d[:i] for d in dataFiles)):
            continue
        else:
            break
    return dataFiles[0][:i-1]
    
def _floatCore(dataFiles,prefix,suffix):
    
    try:
        core = [int(d[len(prefix):-(len(suffix)+1)]) for d in dataFiles]
    except:
        return []
        
    core.sort()
    
    sortedFiles = [''.join([prefix,str(c),'.',suffix]) for c in core]
    
    return sortedFiles
    

def _getmatData(folder, files):

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

#    dataSet = mergeDatasets(dataSets)

    # Create one DataFrame for the timeseries data

#    record = pd.DataFrame(dataSet)

#        record = record.set_index('dfile')

#    return record
    return dataSets

#
#    def _params(self,model, parameters, otherArgs):
#
#        """ For the given model returns the appropreate list for constructing the model instances
#
#        Each line has:
#        (model, {dict of model arguments})
#        """
#
#        params = parameters.keys()
#        paramVals = parameters.values()
#
#        paramCombs = listMerGen(*paramVals)
#
#        modelSet = []
#        for p in paramCombs:
#
#            args = {k:v for k,v in izip(params,p)}
#            for k,v in otherArgs:
#                args[k] = v
#
#            modelSet.append([model, args])
#
#        self.models.append(modelSet)