# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import collections
import types

import pandas as pd
import numpy as np

import utils


def exportClassData(data, parameters, outputFolder="./"):
    """
    Takes the data returned by a model or task class instance and saves
    it to an Excel file
    
    Parameters
    ----------
    data : dict
        A dictionary containing two dimensional arrays and some 1D
        parameters and identifiers
    parameters : dict
        A dictionary containing some 1D arrays, floats and strings. These are
        seen as not changing over the simulation, so will be split if there are
        multiple items
    outputFolder : string, optional
        The full path of where the file will be stored. Default is "./"

    Returns
    -------
    Nothing : None
        It saves an xlsx
    """
    name = data['Name']
    outName = name + '_record'

    timeData = reframeEventDicts(data, parameters)
    record = pd.DataFrame(timeData)
    outputFile = utils.newFile(outName, 'xlsx', outputFolder=outputFolder)
    xlsxT = pd.ExcelWriter(outputFile)
    record.to_excel(xlsxT, sheet_name=outName)
    xlsxT.save()


def reframeEventDicts(data, parameters, storeLabel=''):
    """
    Takes the data returned by a model or task class instance and
    returns a dictionary of 1D lists or single items.

    Parameters
    ----------
    data : dict
        A dictionary containing two dimensional arrays and some 1D
        parameters and identifiers
    parameters : dict
        A dictionary containing some 1D arrays, floats and strings. These are
        seen as not changing over the simulation, so will be split if there are
        multiple items
    storeLabel : string, optional
        An identifier to be added to the beginning of each key string.
        Default is ''.

    Returns
    -------
    newStore : dict of 1D lists or single items
        Any dictionary keys containing 2D lists in the input have been split
        into its constituent columns

    See Also
    --------
    exportClassData, eventDictKeySet, newEventDict
    """

    keySet, T = eventDictKeySet(data, parameters)

    # For every key now found
    newStore = newEventDict(keySet, data, T, storeLabel)

    return newStore


def eventDictKeySet(data, parameters):
    """
    Generates a dictionary of keys and identifiers for the new dictionary,
    splitting any keys with 2D lists into a set of keys, one for each column
    in the original key.

    These are named <key><column number>

    Parameters
    ----------
    data : dict
        A dictionary containing two dimensional arrays and some 1D
        parameters and identifiers
    parameters : dict
        A dictionary containing some 1D arrays, floats and strings. These are
        seen as not changing over the simulation, so will be split if there are
        multiple items

    Returns
    -------
    keySet : dict
        The keys are the keys for the new dictionary. The values contain a
        two element tuple. The first element is the original name of the
        key and the second is the location of the value to be stored in the
        original dictionary value array.
    T : int
        The length of the longest array. This represents the number of 
        trialsteps that have elapsed since the start of the simulation.

    See Also
    --------
    exportClassData, newEventDict, reframeEventDicts
    """

    # Find all the keys
    keySet = collections.OrderedDict()
    paramSet = collections.OrderedDict()
    dataSet = collections.OrderedDict()
    T = 1
    
    for p in parameters.iterkeys():
        v = parameters[p]
        if isinstance(v, (list, np.ndarray)):
            shp = np.shape(v)
            if np.size(shp) == 1:
                if np.size(v) == 1:
                    paramSet.setdefault(p, (None, None))
                else:
                    paramSet.update(_genVarKeys(p, v))
            else:
                paramSet.update(_genVarKeys(p, v))
        else:
            paramSet.setdefault(p, (None, None))

    for k in data.iterkeys():
        if k in parameters:
            continue
        
        v = data[k]
        if isinstance(v, (list, np.ndarray)):
            shp = np.shape(v)
            sze = shp[0]
            if sze > T:
                    T = sze
                    
            if np.size(shp) == 1:
                dataSet.setdefault(k, (None, None))
            else:
                for col in xrange(0, shp[1]):
                    dataSet.setdefault(k+str(col), (k, col))
        else:
            dataSet.setdefault(k, (None, None))
            
    keySet.update(paramSet)
    keySet.update(dataSet)

    return keySet, T


def newEventDict(keySet, data, T, dataLabel=''):
    """
    Takes the data returned by a model or task class instance and
    returns a dictionary of 1D lists.


    Parameters
    ----------
    keySet : dict
        The keys are the keys for the new dictionary. The values contain a
        two element tuple. The first element is the original name of the
        key and the second is the location of the value to be stored in the
        original dictionary value array.
    data : dict
        A dictionary containing two dimensional arrays and some 1D
        parameters and identifiers
    T : int
        The length of the longest array. This represents the number of 
        trialsteps that have elapsed since the start of the simulation.
    dataLabel : string
        An identifier to be added to the beginning of each key string.
        Default ''

    Returns
    -------
    newStore : dict
        The new dictionary with the keys from the keySet and the values as
        1D lists with 'None' if the keys, value pair was not found in the
        store.
        
    See Also
    --------
    exportClassData, eventDictKeySet, reframeEventDicts

    """

    partStore = collections.OrderedDict()

    for key, (initKey, col) in keySet.iteritems():

        partStore.setdefault(key, [])

        if type(initKey) is types.NoneType:
            v = data.get(key, None)
            
        else:
            rawVal = data.get(initKey, None)
            if type(rawVal) is types.NoneType:
                v = None
            elif np.size(np.shape(rawVal)) == 1:
                v = np.array(rawVal)[col]
            else:
                v = np.array(rawVal)[:, col]
            
        if not isinstance(v, (list, np.ndarray)):
            v = [v]
        
        sz = np.size(v)
        partStore[key].extend(v)
        if sz != T:
            diff = T- sz
            partStore[key].extend([None for i in xrange(diff)])

    newStore = collections.OrderedDict(((dataLabel + k, v) for k, v in partStore.iteritems()))

    return newStore


def _genVarKeys(p, v):
    
    pSet = collections.OrderedDict()
    
    arrSets = [range(0, i) for i in np.shape(v)]
    # Now record each one
    for genLoc in utils.listMergeGen(*arrSets):
        if len(genLoc) == 1:
            loc = genLoc[0]
        else:
            loc = tuple(genLoc)
        pSet.setdefault(p+str(loc), (p, loc))
        
    return pSet
