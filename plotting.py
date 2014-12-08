# -*- coding: utf-8 -*-
"""
@author: Dominic
"""
from __future__ import division

import matplotlib
#matplotlib.use('Agg')

import logging

import matplotlib.pyplot as plt

from numpy import array, meshgrid, fromfunction, arange,linspace, sort
from scipy import histogram, amin, amax
from scipy.interpolate import griddata
from matplotlib.cm import get_cmap
#from mayavi import mlab #import quiver3d, flow
from itertools import izip
from bisect import bisect_right
from vtkWriter.vtkStructured import VTK_XML_Serial_Structured
from vtkWriter.vtkUnstructured import VTK_XML_Serial_Unstructured
from vtkWriter.vtkCSV import VTK_CSV


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

### Simple lineplot
def lineplot(x,Y,labels,axisLabels):

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    axPlotlines(ax,x,Y, labels, axisLabels)

    legend(ax)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

### Response changes to different parameters
def varDynamics(paramSet, decisionTimes, **kwargs):

    lparams = len(paramSet)

    params = paramSet.keys()

    if lparams == 1:
        param = params[0]
        fig = dim1VarDim(paramSet[param],decisionTimes, param, **kwargs)

    elif lparams == 2:
        fig = dim2VarDim(paramSet[params[0]],paramSet[params[1]],decisionTimes, params[0], params[1], **kwargs)

    elif lparams == 3:
        fig = dim3VarDim(paramSet[params[0]],paramSet[params[1]],paramSet[params[2]],decisionTimes, params[0], params[1], params[2], **kwargs)

    else:

        fig = None

    return fig

def dim1VarDim(X,Y, varXLabel, **kwargs):
    """
    """
    fig = plt.figure() # figsize=(8, 6), dpi=80)
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

def dim2VarDim(X,Y,z, varXLabel, varYLabel, **kwargs):
    """

    Look in to fatfonts
    """

    contour= kwargs.get("contour",False)
    heatmap = kwargs.get("heatmap",True)
    scatter = kwargs.get("scatter",False)


    zMin = amin(z)
    zMax = amax(z)
    if zMin == zMax:
        logger = logging.getLogger('Plots')
        logger.warning("There is no variation in the time to decision across parameters")
        return None

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    if contour == True:
        CS = axContour(ax,X,Y,z, minZ = 0, CB_label = "Time to decision")

    if scatter == True:
        SC = axScatter(ax,X,Y,z, minZ = 0, CB_label = "Time to decision")

    if heatmap == True:
        IM = axImage(ax,X,Y,z, minZ = 0, CB_label = "Time to decision")

    ax.set_xlabel(varXLabel)
    ax.set_ylabel(varYLabel)
    plt.title('Time to decision across parameters')

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

def dim3VarDim(X,Y,Z,f, varXLabel, varYLabel, varZLabel, **kwargs):

    paraOut = kwargs.get("paraOut", "CSV")

    xMin = amin(X)
    xMax = amax(X)
    yMin = amin(Y)
    yMax = amax(Y)
    zMin = amin(Y)
    zMax = amax(Y)

    xS = sort(list(set(X)))
    yS = sort(list(set(Y)))
    zS = sort(list(set(Z)))

    dx = amin(xS[1:]-xS[:-1])
    dy = amin(yS[1:]-yS[:-1])
    dz = amin(zS[1:]-zS[:-1])
    xJumps = (xMax - xMin)/dx + 1
    yJumps = (yMax - yMin)/dy + 1
    zJumps = (zMax - zMin)/dz + 1

#    X,Y,Z = meshgrid(x,y,z)
#    Xfleshed, Yfleshed, Zfleshed = meshgrid(linspace(xMin - dx, xMax + dx, xJumps),
#                                            linspace(yMin - dy, yMax + dy, yJumps),
#                                            linspace(zMin - dz, zMax + dz, zJumps))
    Xfleshed, Yfleshed, Zfleshed = meshgrid(linspace(xMin, xMax, xJumps),
                                            linspace(yMin, yMax, yJumps),
                                            linspace(zMin, zMax, zJumps))

#    coPoints = [(a,b,c) for a,b,c in izip(X.flatten(),Y.flatten(),Z.flatten())]
    coPoints = [(a,b,c) for a,b,c in izip(X,Y,Z)]
    coGrid = griddata(coPoints, f, (Xfleshed, Yfleshed,Zfleshed), method='nearest')

#    coGridX = coGrid[1:,:,:] - coGrid[:-1,:,:]
#    coGridY = coGrid[:,1:,:] - coGrid[:,:-1,:]
#    coGridZ = coGrid[:,:,1:] - coGrid[:,:,:-1]

    if paraOut == "structured":
        vtk_writer = VTK_XML_Serial_Structured()
    elif paraOut == "CSV":
        vtk_writer = VTK_CSV()
    elif paraOut == "unstructured":
        vtk_writer = VTK_XML_Serial_Unstructured()
    else:
        return None
    vtk_writer.snapshot(Xfleshed,
                        Yfleshed,
                        Zfleshed,
                        xLabel = varXLabel,
                        yLabel = varYLabel,
                        zLabel = varZLabel,
#                        x_force = coGridX.flatten(),
#                        y_force = coGridY.flatten(),
#                        z_force = coGridZ.flatten(),
                        radii = coGrid.flatten())

    return vtk_writer

### Inputs compared to output probabilities
def dataVsEvents(data,events,labels,eventLabel,axisLabels):

    data = array(data)

    if len(data) > 8:
        fig = dataSpectrumVsEvents(data,events,labels,eventLabel,axisLabels)
        return fig

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    y2Label = axisLabels.pop("y2Label","Event value")

    eventTimes = range(1, len(events)+1)
    eventX = [i - 0.5 for i in eventTimes]

    axPlotlines(ax,eventTimes,data, labels, axisLabels)

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label = eventLabel, color = 'k', linewidth=2,markersize = 5)
    bottom,top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01,top + 0.01))

    axb.set_ylabel(y2Label)

    legend(ax,axb)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


def dataSpectrumVsEvents(data,events,labels,eventLabel,axisLabels):
    """Normalised spectogram/histogram of values across events

    printResourcesSpectrum(data,bins)"""

    y2Label = axisLabels.pop("y2Label","Event value")

    eventX = fromfunction(lambda i: i+0.5, (len(events),))

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    axSpectrum(ax,events,data, axisLabels)

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label = eventLabel, color = 'k', linewidth=2,markersize = 5)
    bottom,top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01,top + 0.01))

    axb.set_ylabel(y2Label)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig

### Background to plotting functions
# Taking the final stages of plotting and providing a nice interface to do it all in one call
# rather than a few dozen

def axScatter(ax,X,Y,z, minZ = 0, CB_label = ""):
    """
    """

    maxZ = amax(z)

    minC = 1
    maxC = 301
    zScale = (maxC - minC) / float(maxZ - minZ)
    C = minC + (z-amin(z))*zScale

#    X,Y = meshgrid(x,y)
#    sc= ax.scatter(X.flatten(),Y.flatten(),s=C)
    sc= ax.scatter(X,Y,s=C)

    return sc

def axImage(ax,X,Y,z,minZ = 0,CB_label = ""):
    """
    """

    xMin = amin(X)
    xMax = amax(X)
    yMin = amin(Y)
    yMax = amax(Y)

    xS = sort(list(set(X)))
    yS = sort(list(set(Y)))

    dx = amin(xS[1:]-xS[:-1])
    dy = amin(yS[1:]-yS[:-1])

    xJumps = (xMax - xMin)/dx + 2
    yJumps = (yMax - yMin)/dy + 2

    xMinIm = xMin - dx
    xMaxIm = xMax + dx
    yMinIm = yMin - dy
    yMaxIm = yMax + dy

#    X,Y = meshgrid(x,y)
    Xfleshed,Yfleshed = meshgrid(linspace(xMinIm,xMaxIm,xJumps),linspace(yMinIm,yMaxIm,yJumps))

#    zPoints = [(a,b) for a,b in izip(X.flatten(),Y.flatten())]
    zPoints = [(a,b) for a,b in izip(X,Y)]
    gridZ = griddata(zPoints, z, (Xfleshed, Yfleshed), method='nearest')

#    qm = plt.pcolormesh(gridZ, cmap = local_cmap)
    im = ax.imshow(gridZ, interpolation='nearest',
                    origin='lower',
                    cmap=local_cmap,
                    extent=[xMinIm,xMaxIm,yMinIm,yMaxIm],
                    vmin = minZ,
                    aspect='auto')
    CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
    CBI.set_label(CB_label)

    return im

def axContour(ax,X,Y,z,minZ = 0,CB_label = ""):

    xMin = amin(X)
    xMax = amax(X)
    yMin = amin(Y)
    yMax = amax(Y)

    maxZ = amax(z)
    if maxZ == minZ:
        maxZ += 1
    zSorted = sort(z)
    i = bisect_right(zSorted, 0)
    if i != len(zSorted):
        diffSort = sort(zSorted[1:]-zSorted[:-1])
        dz = diffSort[bisect_right(diffSort, 0)]
        zJumps = (maxZ - minZ)/dz + 2
        levels = linspace(minZ,maxZ+dz,zJumps)
    else:
        levels = arange(maxZ+1)

#    X,Y = meshgrid(x,y)

    Xlen = len(set(X))
    Ylen = len(set(Y))

    z = array(z)
    Z = z.reshape((Ylen,Xlen))

    CS = ax.contour(X,Y, Z, levels,
                     origin='lower',
                     colors = 'k',
                     linewidths=2,
                     extent=[xMin,xMax,yMin,yMax],
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
    CB.set_label(CB_label)

#    # Improving the position of the contour bar legend
#    l,b,w,h = ax.get_position().bounds
#    ll,bb,ww,hh = CB.ax.get_position().bounds
#    CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

    return CS

def axPlotlines(ax, x, Y, labels, axisLabels):

    xLabel = axisLabels.pop("xLabel","Event")
    yLabel = axisLabels.pop("yLabel","Value")
    title = axisLabels.pop("title","")

    yMin = axisLabels.pop("yMin",amin(Y))
    yMax = axisLabels.pop("yMax",amax(Y))
    xMin = axisLabels.pop("xMin",min(x))
    Ys = Y.shape
    if len(Ys)== 1:
        xMax = axisLabels.pop("xMax",Ys[0])
        pltLines = ax.plot(x,Y, lpl[0], color = colours[0], linewidth=lpl_linewidth[0], markersize = 3)
    elif len(Ys) == 2:
        xMax = axisLabels.pop("xMax",Ys[1])
        for i, y in enumerate(Y):
            pltLines = ax.plot(x,y, lpl[i], label = labels[i], color = colours[i], linewidth=lpl_linewidth[i],markersize = 3)

    ax.set_ylim([yMin,yMax])
    ax.set_xlim([xMin,xMax])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

def axSpectrum(ax,x,Y, axisLabels):

    xLabel = axisLabels.pop("xLabel","Events")
    yLabel = axisLabels.pop("yLabel","Density across parameter range")
    title = axisLabels.pop("title","")

    minVal = axisLabels.pop("yMin",0)
    maxVal = axisLabels.pop("yMax",amax(Y))
    xMin = axisLabels.pop("xMin",0)
    xMax = axisLabels.pop("xMax",len(Y[0]))

    bins = axisLabels.pop("bins",25)
    probDensity = axisLabels.pop("probDensity",False)

    data = array(Y)

    # Create this histogram-like data
    histData = [histogram(d, bins, range=(minVal,maxVal), density=True) for d in data.T]

    if probDensity:
        plotData = array([d[0] for d in histData])
    else:
        plotData = array([d[0]*(d[1][1:]-d[1][:-1]) for d in histData])

    # Since we are calculating a probability density and the bins are less than 1 in size we needed
    # To multiply by the bin size to get the actual probability of each bin

    im = ax.imshow(plotData.T,
               interpolation='nearest',
               cmap=wintermod_cmap,
               origin='lower',
               extent=[xMin,xMax,minVal,maxVal],
               aspect='auto' )
    col = plt.colorbar(im,orientation='horizontal')
    if probDensity:
        col.set_label("Probability density")
    else:
        col.set_label("Bin probability")

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

def legend(*axis):

    lines = []
    pltLabels = []
    for ax in axis:

        line, pltLabel = ax.get_legend_handles_labels()
        lines.extend(line)
        pltLabels.extend(pltLabel)

    if not lines:
        return

    leg = ax.legend(lines, pltLabels,loc = 'best', fancybox=True)




if __name__ == '__main__':


#    x = array([1, 2, 4])
#    y = array([0.1, 0.2, 0.3])
#    z = array([0.3, 0.4, 0.5])
    x = array([1, 2, 3])
    y = array([4, 5, 6])
    z = array([7, 8, 9])
    f = array([1,2,3,2,3,4,3,4,5,2,3,4,3,4,5,4,5,6,3,4,5,4,5,6,5,6,7])

    fig = dim3VarDim(x,y,z,f, 'TestX', 'TestY','TestZ')
    fig.outputTrees("outputs/testDataStruc")

#    fig = dim2VarDim(x,y,z, 'TestX', 'TestY', contour=False, heatmap = False, scatter = True)

#    X = array([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
#    Y = array([[ 0.1,  0.1,  0.1], [ 0.2,  0.2,  0.2], [ 0.3,  0.3,  0.3]])
#    Z = array([[2, 1, 1], [0, 2, 1], [0, 0, 1]])

#    fig = plt.figure()  # figsize=(8, 6), dpi=80)
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
#
#    plt.gcf().set_size_inches(6, 6)
#
#    plt.show()
