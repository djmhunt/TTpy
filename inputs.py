# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

import cPickle as pickle

def unpickleLog(fileName):

    with open(fileName) as o:
        results = pickle.load(o)

    return results