# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import matplotlib
#matplotlib.use('Agg')

import logging

import matplotlib.pyplot as plt

from numpy import array, meshgrid, fromfunction
from scipy import histogram, amin, amax

### Define the different types of lines that will be plotted and their properties.
# Symbols
symbols = ['-','--','-.',':','.',',','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']
## Symbols + line
#lps = ['k' + k for k in [',','.','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']]
dots = ['.', 'o', '^', 'x', 'd', '2', 'H', ',', 'v', '+', 'D', 'p', '<', 's', '1', 'h', '4', '>', '3']
scatterdots = ['o', '^', 'x', 'd', 'v', '+', 'p', '<', 's', 'h', '>', '8']
lines = ['-', '--', ':', '-.','-']
lines_width = [1,1,2,2,2]
large = ['^', 'x', 'D', '4', 'd', 'p', '>', '2', ',', '3', 'H']
large_line_width = [2]*len(large)
lpl = lines + large
lpl_linewidth = large_line_width + lines_width
colours = ['g','r','b','m','0.85'] + ['k']*len(large)

### Define the different colour maps to be considered
#Colour plot colour map
defaultCMap = {'blue': ((0.0, 1.0, 1.0), (0.0000000001, 1.0, 1.0), (1.0, 0.5, 0.5)),
               'green': ((0.0, 1.0, 1.0), (0.0000000001, 0.0, 0.0), (1.0, 1.0, 1.0)),
                 'red': ((0.0, 1.0, 1.0), (0.0000000001, 0.0, 0.0), (1.0, 0.0, 0.0))}
#local_cmap = matplotlib.colors.LinearSegmentedColormap('local_colormap',defaultCMap)
wintermodCMap = {'blue': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 1.0), (1.0, 0.5, 0.5)),
               'green': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 0.0), (1.0, 1.0, 1.0)),
                 'red': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 0.0), (1.0, 0.0, 0.0))}
wintermod_cmap = matplotlib.colors.LinearSegmentedColormap('local_colormap_winter',wintermodCMap)
local_cmap = wintermod_cmap
#local_cmap = 'winter'

origin = 'lower'
#origin = 'upper'

### Plots
def varDynamics(params, paramVals, decisionTimes):

    lparams = len(params)

    if lparams == 1:
        fig = dim1VarDim(paramVals[0],decisionTimes, params[0])

    elif lparams == 2:
        X,Y = meshgrid(*paramVals)
        decisionTimes = array(decisionTimes)
        decisionTimes.reshape(X.shape)
        fig = dim2VarDim(X,Y,decisionTimes, params[0], params[1])

    elif lparams == 3:
        fig = None

    return fig

def dim1VarDim(X,Y, varXLabel):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    maxVal = amax(Y)
    minVal = amin(Y)
    if minVal > 0 or minVal == None:
        minVal = 0
    maxVal += (maxVal-minVal)/100.0

    plt.plot(X,Y)


    ax.set_ylim((minVal,maxVal))

    plt.xlabel(varXLabel)
    plt.ylabel("Reaction")
    plt.title("Parameter reaction times")

    return fig

def dim2VarDim(X,Y,Z, varXLabel, varYLabel):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    minZ = amin(Z)-1
    maxZ = amax(Z)+2
    levels = range(minZ,maxZ)
    CS = plt.contour(X, Y, Z, levels,
                        colors = local_cmap,
                        origin=origin,
                        linewidths = 3,
                        extend='both')
    plt.clabel(CS, fmt = '%2.1f', colors = 'w', fontsize=14)
    plt.xlabel(varXLabel)
    plt.ylabel(varYLabel)
    plt.title('Parameter reaction times')

#    CS.cmap.set_under('yellow')
#    CS.cmap.set_over('cyan')

    col = plt.colorbar()
    col.set_label("Reaction time")

    return fig

def dataVsEvents(data,events,labels,eventLabel,axisLabels):

    if len(data) > 8:
        fig = dataSpectrumVsEvents(data,events,labels,eventLabel,axisLabels)
        return fig

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xLabel = axisLabels.pop("xLabel","Event")
    yLabel = axisLabels.pop("yLabel","Value")
    y2Label = axisLabels.pop("y2Label","Event value")
    title = axisLabels.pop("title","")

    eventX = fromfunction(lambda i: i+0.5, (len(events),))


    for i, d in enumerate(data):

        pltLine = ax.plot(d, lpl[i], label = labels[i], color = colours[i], linewidth=lpl_linewidth[i],markersize = 3)#, axes=axs[0])

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label = eventLabel, color = 'k', linewidth=2,markersize = 5)
    bottom,top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01,top + 0.01))

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    axb.set_ylabel(y2Label)
    ax.set_title(title)

    lines1, pltLables1 = ax.get_legend_handles_labels()
    lines2, pltLables2 = axb.get_legend_handles_labels()
    pltLables = pltLables1 + pltLables2
    lines = lines1 + lines2
    leg = ax.legend(lines, pltLables,loc = 'best', fancybox=True)

    return fig

def dataSpectrumVsEvents(data,events,labels,eventLabel,axisLabels):
    """Normalised spectogram/histogram of values across events

    printResourcesSpectrum(data,bins)"""

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xLabel = axisLabels.pop("xLabel","Events")
    yLabel = axisLabels.pop("yLabel","Density across parameter range")
    y2Label = axisLabels.pop("y2Label","Event value")
    title = axisLabels.pop("title","")

    maxVal = amax(data)
    minVal = amin(data)

    if minVal > 0:
        minVal = 0

    bins = 25

    eventX = fromfunction(lambda i: i+0.5, (len(events),))

    data = array(data)

    # Create this histogram-like data
    histData = [histogram(d, bins, range=(minVal,maxVal), normed=True) for d in data.T]#if normed unknown try density

    plotData = array([d[0] for d in histData])

    plt.imshow(plotData.T, interpolation='nearest', cmap=wintermod_cmap, origin='lower', extent=[0,len(plotData),minVal,maxVal], aspect='auto' )
    col = plt.colorbar(orientation='horizontal')
    col.set_label("Probability density")

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label = eventLabel, color = 'k', linewidth=2,markersize = 5)
    bottom,top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01,top + 0.01))

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    axb.set_ylabel(y2Label)
    ax.set_title(title)



    return fig
