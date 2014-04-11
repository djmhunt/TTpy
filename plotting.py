# -*- coding: utf-8 -*-
"""
@author: Dominic
"""

import matplotlib
#matplotlib.use('Agg')

import logging

import matplotlib.pyplot as plt

from numpy import array, meshgrid, fromfunction, arange,linspace, sort
from scipy import histogram, amin, amax
from scipy.interpolate import griddata
from matplotlib.cm import get_cmap
from itertools import izip
from bisect import bisect_right


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
#local_cmap = get_cmap('winter')

origin = 'lower'
#origin = 'upper'

### Plots
def varDynamics(params, paramVals, decisionTimes):

    lparams = len(params)

    if lparams == 1:
        fig = dim1VarDim(paramVals[0],decisionTimes, params[0])

    elif lparams == 2:
        fig = dim2VarDim(paramVals[0],paramVals[1],decisionTimes, params[0], params[1])

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

    ax.plot(X,Y)

    if minVal != maxVal:
        ax.set_ylim((minVal,maxVal))
    else:
        logger1 = logging.getLogger('Plots')
        logger1.warning("There is no variation in the time to decision across parameters")

    ax.set_xlabel(varXLabel)
    ax.set_ylabel("Time to decision")
    plt.title("Time to decision across parameters")

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

def dim2VarDim(x,y,z, varXLabel, varYLabel):
    """

    Todo: Split out the actual plotting so contour plots do not interfere with heatmaps.

    Put in a scatter plot: plt.scatter(theta,beta,s=beadTotal)

    Look in to fatfonts
    """

    X,Y = meshgrid(x,y)
    z = array(z)
    Z = z.reshape(X.shape)

    xMin = amin(X)
    xMax = amax(X)
    yMin = amin(Y)
    yMax = amax(Y)
    if yMin == yMax:
        logger1 = logging.getLogger('Plots')
        logger1.warning("There is no variation in the time to decision across parameters")
        return plt.figure()

    minZ = 0
    maxZ = amax(z)
    if maxZ == minZ:
        maxZ += 1
    zSorted = sort(z)
    i = bisect_right(zSorted, 0)
    if i != len(zSorted):
        dz = zSorted[i]
        zJumps = (maxZ - minZ)/dz + 2
        levels = linspace(minZ,maxZ+dz,zJumps)
    else:
        levels = arange(maxZ+1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    CS = ax.contour(X,Y, Z, levels,
                     origin='lower',
                     colors = 'k',
                     linewidths=2,
                     extent=[xMin,xMax,yMin, yMax],
                     extend = 'both')
    plt.clabel(CS, levels,#[1::2],  # label every second level
               inline=1,
               fmt='%2.1f',
               fontsize=14,
               colors = 'k')

#    CS.cmap.set_under('yellow')
#    CS.cmap.set_over('cyan')
    CS.cmap.set_bad("red")

    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    CB.set_label("Time to decision")

    dx = amin(x[1:]-x[:-1])/2.0
    dy = amin(y[1:]-y[:-1])/2.0
    xJumps = (xMax - xMin)/dx + 2
    yJumps = (yMax - yMin)/dy + 2

    xMinIm = xMin - dx
    xMaxIm = xMax + dx
    yMinIm = yMin - dy
    yMaxIm = yMax + dy

    Xfleshed,Yfleshed = meshgrid(linspace(xMinIm,xMaxIm,xJumps),linspace(yMinIm,yMaxIm,yJumps))

    zPoints = [(a,b) for a,b in izip(X.flatten(),Y.flatten())]
    gridZ = griddata(zPoints, z, (Xfleshed, Yfleshed), method='nearest')

#    qm = plt.pcolormesh(gridZ, cmap = local_cmap)
    im = ax.imshow(gridZ, interpolation='nearest',
                    origin='lower',
                    cmap=local_cmap,
                    extent=[xMinIm,xMaxIm,yMinIm,yMaxIm],
                    vmin = minZ,
                    aspect='auto')
    CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
    CBI.set_label("Time to decision")

    ax.set_xlabel(varXLabel)
    ax.set_ylabel(varYLabel)
    plt.title('Time to decision across parameters')

    # Improving the position of the contour bar legend
    l,b,w,h = ax.get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

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

    yMin = axisLabels.pop("yMin",0)
    yMax = axisLabels.pop("yMax",amax(data))
    xMin = axisLabels.pop("xMin",0)
    xMax = axisLabels.pop("xMax",len(data[0]))

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

#    yLim = list(ax.get_ylim())
#    if yMin != None:
#        yLim[0] = yMin
#    if yMax != None:
#        yLim[1] = yMax
#    ax.set_ylim(yLim)
#    xLim = list(ax.get_xlim())
#    if xMin != None:
#        xLim[0] = xMin
#    if xMax != None:
#        xLim[1] = xMax
#    ax.set_xlim(xLim)

    lines1, pltLables1 = ax.get_legend_handles_labels()
    lines2, pltLables2 = axb.get_legend_handles_labels()
    pltLables = pltLables1 + pltLables2
    lines = lines1 + lines2
    leg = ax.legend(lines, pltLables,loc = 'best', fancybox=True)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

def dataSpectrumVsEvents(data,events,labels,eventLabel,axisLabels):
    """Normalised spectogram/histogram of values across events

    printResourcesSpectrum(data,bins)"""



    xLabel = axisLabels.pop("xLabel","Events")
    yLabel = axisLabels.pop("yLabel","Density across parameter range")
    y2Label = axisLabels.pop("y2Label","Event value")
    title = axisLabels.pop("title","")

    minVal = axisLabels.pop("yMin",0)
    maxVal = axisLabels.pop("yMax",amax(data))
    xMin = axisLabels.pop("xMin",0)
    xMax = axisLabels.pop("xMax",len(data[0]))

    bins = 25

    eventX = fromfunction(lambda i: i+0.5, (len(events),))

    data = array(data)

    # Create this histogram-like data
    histData = [histogram(d, bins, range=(minVal,maxVal), normed=True) for d in data.T]#if normed unknown try density

    plotData = array([d[0] for d in histData])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(plotData.T,
               interpolation='nearest',
               cmap=wintermod_cmap,
               origin='lower',
               extent=[xMin,xMax,minVal,maxVal],
               aspect='auto' )
    col = plt.colorbar(im,orientation='horizontal')
    col.set_label("Probability density")

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label = eventLabel, color = 'k', linewidth=2,markersize = 5)
    bottom,top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01,top + 0.01))

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    axb.set_ylabel(y2Label)
    ax.set_title(title)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

### Background to plotting functions
# Taking the final stages of plotting and providing a nice interface to do it all in one call
# rather than a few dozen

def axScatterSize(ax,x,y,z):
    """
    """

    ax.scatter(x,y,s=z)

    return ax

def axImage(ax,x,y,z,lables):
    """
    """

    ax.scatter(x,y,s=z)

    return ax



if __name__ == '__main__':

#    X = array([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
    x = array([1,2,4])

#    Y = array([[ 0.1,  0.1,  0.1], [ 0.2,  0.2,  0.2], [ 0.3,  0.3,  0.3]])
    y = array([0.1, 0.2, 0.3])

#    Z = array([[2, 1, 1], [0, 2, 1], [0, 0, 1]])
    z = array([1,2,3,2,3,4,3,4,5])

    fig = dim2VarDim(x,y,z, 'TestX', 'TestY')

#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#
#    minZ = amin(Z)-1
#    maxZ = amax(Z)+2
#    levels = arange(minZ,maxZ,2)
#    area = [amin(X),amax(X),amin(Y),amax(Y)]
#
#    qm = plt.pcolormesh(X, Y, Z,
#                   cmap = local_cmap)
#    CBI = plt.colorbar(qm, orientation='horizontal', shrink=0.8)
#    CBI.set_label("Time to decision")

#    plt.gcf().set_size_inches(6, 6)

    plt.show()
