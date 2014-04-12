# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt
import cPickle as pickle
import pandas as pd

from os.path import isfile, exists
from numpy import array


def plots(experiment, folderName, silent, saveFig, ivText, models, *majFigureSets, **kwargs):
    """ Plots the results for each of the models

    """

    logger1 = logging.getLogger('Plots')

    message = "Produce plots"
    logger1.info(message)

    expFigureSets = experiment.plots( ivText, **models)
    figureSets = list(majFigureSets) + expFigureSets
    for (handle,figure) in figureSets:
        fileName = folderName + "\\" + handle
        outputFig(figure,fileName, silent, saveFig)

    if not silent:
        if not matplotlib.is_interactive():
            plt.show()
        else:
            plt.draw()
    else:
        plt.close()


### Pickled outputs
def pickleRec(data,outputFile):

    if exists(outputFile):
        i = 1
        while exists(outputFile + "_" + str(i)):
            i += 1
        outputFile + "_" + str(i)

    outputFile += '.pkl'

    with open(outputFile,'w') as w :
        pickle.dump(data, w)

def pickleLog(results,folderName, label=""):

    if label:
        outputFile = folderName + 'Pickle\\' + results["Name"] + "-" + label
    else:
        outputFile = folderName + 'Pickle\\' + results["Name"]

    pickleRec(results,outputFile)

def simSetLog(paramText,params,paramVals,firstDecision,expName,modelName,kwargs,ivLabel,folderName):

    data = {'sim': paramText,
            'experiment': expName,
            'model': modelName,
            'ivLabel': ivLabel,
            'folder': folderName}

    data['beadTotal'] = firstDecision

    for i,n in enumerate(params):
        data[n] = paramVals[:,i]

    for k,v in kwargs.iteritems():
        data[k] = v

    record = pd.DataFrame(data)

    record = record.set_index('sim')

    outputFile = folderName + 'Pickle\\' + 'simStore'

    pickleRec(record,outputFile)

#readableLog.append((paramText, p, modelResult["firstDecision"]))

### Graphical outputs
def outputFig(fig,fileName,silent,saveFig):
    """Saves the figure to a .png file and/or displays it on the screen.

    fig:        MatPlotLib figure object
    fileName:   The name and path of the file the figure should be saved to. If ommited
                the file will be saved to a default name.
    saveFig:    If true the figure will be saved.
    silent:     If false the figure will be plotted to the screen. If true the figure
                will be closed


    outputFig(fig,fileName,silent,saveFig)
    """

    plt.figure(fig.number)

    if saveFig:
        if exists(fileName):
            i = 1
            while exists(fileName + "_" + str(i)):
                i += 1
            fileName + "_" + str(i)

        ndpi = fig.get_dpi()

        plt.savefig(fileName,dpi=ndpi)

#    if not silent and matplotlib.is_interactive():
#        plt.draw()


