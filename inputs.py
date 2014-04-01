# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

import pandas as pd
import cPickle as pickle

from os import listdir

def unpickleModels(folderName):

    for f in listdir(folderName):
        if f.startswith("model_"):
            fileName = folderName + f
            with open(fileName) as o:
                yield pickle.load(o)


def readData(fileName):

    # "../beads task stuff/Beads Data Exp 2 by CTA.xlsx"
    xlf = pd.ExcelFile(fileName)
    par = xlf.parse('Sheet2',header=0,index_col=0)

    return par