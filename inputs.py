# -*- coding: utf-8 -*-
"""

@author: Dominic
"""
from __future__ import division

import pandas as pd
import cPickle as pickle

import logging

from os import listdir
from os.path import isfile, exists

def unpickleModels(folderName):

    pickleFolderName = folderName + 'Pickle\\'
    for f in listdir(pickleFolderName):
        if f.startswith("model_"):
            fileName = pickleFolderName + f
            with open(fileName) as o:
                label = f[6:][:-4]
                data = pickle.load(o)
                yield label, data

def unpickleSimDescription(folderName):

    fileName = folderName + 'Pickle\\simStore.pkl'
    if isfile(fileName):
        with open(fileName) as o:
            data = pickle.load(o)
            return data
    else:
        logger1 = logging.getLogger('Inputs')
        logger1.warning("The simulation records were not kept. Check how things were performed")


def readData(fileName):

    # "../beads task stuff/Beads Data Exp 2 by CTA.xlsx"
    xlf = pd.ExcelFile(fileName)
    par = xlf.parse('Sheet2',header=0,index_col=0)

    return par