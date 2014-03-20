# -*- coding: utf-8 -*-
"""

@author: Dominic
"""

import matplotlib
#matplotlib.interactive(True)
import logging

import matplotlib.pyplot as plt
import cPickle as pickle

from os.path import isfile, exists


def plots(experiment, folderName, silent, saveFig, ivText, **models):
    """ Plots the results for each of the models

    """

    logger1 = logging.getLogger('Plots')

    message = "Produce plots"
    logger1.info(message)

    figureSets = experiment.plots( ivText, **models)
    for (handle,figure) in figureSets:
        fileName = folderName + "\\" + handle
        outputFig(figure,fileName, silent, saveFig)

### Pickled outputs
def pickleLog(results,folderName, label=""):

    if label:
        outputFile = folderName + 'Pickle\\' + results["Name"] + "-" + label
    else:
        outputFile = folderName + 'Pickle\\' + results["Name"]

    if exists(outputFile):
        i = 1
        while exists(outputFile + "_" + str(i)):
            i += 1
        outputFile + "_" + str(i)

    outputFile += '.pkl'

    with open(outputFile,'w') as w :
        pickle.dump(results, w)

### Graphical outputs
def outputFig(fig,fileName,silent,saveFig):
    """Saves the figure to a .png file and/or displays it on the screen.

    fig:        MatPlotLib figure object
    fileName:    The name and path of the file the figure should be saved to. If ommited
            the file will not be saved.
    silent:    If false the figure will be plotted to the screen. If true the figure
            will be closed


    outputFig(fig,fileName,silent)
    """
    if saveFig:
        if exists(fileName):
            i = 1
            while exists(fileName + "_" + str(i)):
                i += 1
            fileName + "_" + str(i)

        ndpi = fig.get_dpi()

        plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

        plt.savefig(fileName,dpi=3*ndpi)

    if not silent:
        if matplotlib.is_interactive():
            plt.draw()
        else:
            plt.show()
    else:
        plt.close()
