# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

import matplotlib
#matplotlib.use('Agg')

import logging

import matplotlib.pyplot as plt
import pandas as pd

from numpy import array, meshgrid, fromfunction, arange, linspace, sort, shape
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
symbols = ['-', '--', '-.', ':', '.', ',', 'o', '^', 'v', '<', '>', 's', '+', 'x', 'D', 'd', '1', '2', '3', '4', 'h', 'H', 'p']
## Symbols + line
#lps = ['k' + k for k in [',','.','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']]
dots = ['.', 'o', '^', 'x', 'd', '2', 'H', ',', 'v', '+', 'D', 'p', '<', 's', '1', 'h', '4', '>', '3']
scatterdots = ['o', '^', 'x', 'd', 'v', '+', 'p', '<', 's', 'h', '>', '8']
lines = ['-', '--', ':', '-.', '-']
lines_width = [1, 1, 2, 2, 2]
large = ['^', 'x', 'D', '4', 'd', 'p', '>', '2', ',', '3', 'H']
large_line_width = [2]*len(large)
lpl = lines + large
lpl_linewidth = large_line_width + lines_width
colours = ['g', 'r', 'b', 'm', '0.85'] + ['k']*len(large)

### Define the different colour maps to be considered
# Colour plot colour map
defaultCMap = {'blue': ((0.0, 1.0, 1.0), (0.0000000001, 1.0, 1.0), (1.0, 0.5, 0.5)),
               'green': ((0.0, 1.0, 1.0), (0.0000000001, 0.0, 0.0), (1.0, 1.0, 1.0)),
                 'red': ((0.0, 1.0, 1.0), (0.0000000001, 0.0, 0.0), (1.0, 0.0, 0.0))}
# local_cmap = matplotlib.colors.LinearSegmentedColormap('local_colormap',defaultCMap)
wintermodCMap = {'blue': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 1.0), (1.0, 0.5, 0.5)),
                 'green': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 0.0), (1.0, 1.0, 1.0)),
                 'red': ((0.0, 1.0, 1.0), (0.0000000000000001, 1.0, 0.0), (1.0, 0.0, 0.0))}
wintermod_cmap = matplotlib.colors.LinearSegmentedColormap('local_colormap_winter',wintermodCMap)
local_cmap = wintermod_cmap
# local_cmap = get_cmap('winter')

origin = 'lower'
# origin = 'upper'


### Simple lineplot
def lineplot(x, Y, labels, axisLabels):
    """Produces a lineplot for multiple datasets witht he same x and y axis

    Uses ``axPlotlines`` for plotting

    Parameters
    ----------
    x : array of length n of floats
    Y : array of size m*n of floats
    labels : list of strings
        Description of each of the m datasets in ``Y``
    axisLabels : dictionary
        A dictionary of axis properties, all optional, used in ``axPlotlines``.
        See ``axPlotlines`` for more details.

    Returns
    -------
    fig : matplotlib figure object
    """

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    axPlotlines(ax, x, Y, labels, axisLabels)

    legend(ax)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


### Response changes to different parameters
def paramDynamics(paramSet, responses, axisLabels, **kwargs):
    """Plots response changes to different parameters

    Uses ``paramD1Dynamics``, ``paramD2Dynamics`` and ``paramD3Dynamics`` for plotting
    depending on number of parameters chosen. See their documentation for more
    detail.

    Parameters
    ----------
    paramSet : dictionary
        The keys are the parameter names. The values are a list or array of
        floats, showing the different values of the parameter looked at.
    responses : array of floats
        The measure over which the parameters are being evaluated
    axisLabels : dict of strings or floats
        Axis information such as labels or axis title. See the different subfunctions for specific parameter list
    kwargs : dictionary
        kwargs depend on number of parameters chosen. See documentation for
        ``paramD1Dynamics``, ``paramD2Dynamics`` and ``paramD3Dynamics`` for kwarg lists

    Returns
    -------
    fig : matplotlib figure object or, if ``paramD3Dynamics`` has been used, a ``vtkWriter`` object

    """

    lparams = len(paramSet)

    params = paramSet.keys()

    if lparams == 1:
        axisLabels.setdefault("xLabel", params[0])
        fig = paramD1Dynamics(paramSet[params[0]],
                              responses,
                              axisLabels,
                              **kwargs)

    elif lparams == 2:
        axisLabels.setdefault("xLabel", params[0])
        axisLabels.setdefault("yLabel", params[1])
        fig = paramD2Dynamics(paramSet[params[0]],
                              paramSet[params[1]],
                              responses,
                              axisLabels,
                              **kwargs)

    elif lparams == 3:
        axisLabels.setdefault("xLabel", params[0])
        axisLabels.setdefault("yLabel", params[1])
        axisLabels.setdefault("zLabel", params[2])
        fig = paramD3Dynamics(paramSet[params[0]],
                              paramSet[params[1]],
                              paramSet[params[2]],
                              responses,
                              axisLabels,
                              **kwargs)

    else:

        fig = None

    return fig


def paramD1Dynamics(X, Y, axisLabels, **kwargs):
    """

    Parameters
    ----------
    X : array of size m*n of floats
    Y : array of size m*n of floats
    axisLabels : dictionary
        A dictionary of axis titles, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,

    Returns
    -------
    fig : matplotlib figure object
    """

    xLabel = axisLabels.pop("xLabel", "Parameter")
    yLabel = axisLabels.pop("yLabel", "Value")
    title = axisLabels.pop("title", "")

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    maxVal = amax(Y)
    minVal = amin(Y)
    if (minVal > 0) or (minVal is None):
        minVal = 0
    maxVal += (maxVal-minVal)/100.0

    ax.plot(X, Y)

    if minVal != maxVal:
        ax.set_ylim((minVal, maxVal))
    else:
        logger = logging.getLogger('Plots')
        logger.warning("There is no variation in the " + yLabel + " across " + xLabel)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    plt.title(title)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


def paramD2Dynamics(X, Y, z, axisLabels, **kwargs):
    """
    A set of two dimensional plots superimposed on each other

    Parameters
    ----------
    X : array of size m*n of floats
    Y : array of size m*n of floats
    z : list or array of floats
        The magnitude values for the X,Y locations
    axisLabels : dictionary
        A dictionary of axis titles, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,
        ``cbLabel`` : `string`,
    contour : bool, optional
        Present the data as a contour plot using axContour. Default ``False``
    heatmap : bool, optional
        Presents the data as a square pixel heatmap using axImage. Default ``True``
    scatter : bool, optional
        Presents the data as a scatter plot using axScatter. Default ``False``
    minZ : float, optional
        Specifies what the minimum value of ``z`` should be set to. Default, ``0``

    Returns
    -------
    fig : matplotlib figure object
    """

    #Look in to fatfonts

    xLabel = axisLabels.pop("xLabel", "Parameter 1")
    yLabel = axisLabels.pop("yLabel", "Parameter 2")
    title = axisLabels.pop("title", "")
    CB_label = axisLabels.pop("cbLabel", "")

    contour = kwargs.get("contour", False)
    heatmap = kwargs.get("heatmap", True)
    scatter = kwargs.get("scatter", False)
    minZ = kwargs.get("minZ", 0)
    cmap = kwargs.get("cmap", local_cmap)

    zMin = amin(z)
    zMax = amax(z)
    if zMin == zMax:
        logger = logging.getLogger('Plots')
        logger.warning("There is no variation in the " + yLabel + " across " + xLabel)
        return None

    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(111)

    CB = None
    if scatter is True:
        SC, CB = axScatter(ax, X, Y, z, minZ=minZ, CB=CB, CB_label=CB_label, cmap=cmap)

    if contour is True:
        CS, CB = axContour(ax, X, Y, z, minZ=minZ, CB=CB, CB_label=CB_label, cmap=cmap)

    if heatmap is True:
        IM, CB = axImage(ax, X, Y, z, minZ=minZ, CB=CB, CB_label=CB_label, cmap=cmap)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    plt.title(title)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


def paramD3Dynamics(X, Y, Z, f, axisLabels, **kwargs):

    xLabel = axisLabels.pop("xLabel", "Parameter 1")
    yLabel = axisLabels.pop("yLabel", "Parameter 2")
    zLabel = axisLabels.pop("zLabel", "Parameter 3")

    paraOut = kwargs.get("paraviewInput", "CSV")

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
    coPoints = [(a, b, c) for a, b, c in izip(X, Y, Z)]
    coGrid = griddata(coPoints, f, (Xfleshed, Yfleshed, Zfleshed), method='nearest')

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
                        xLabel=xLabel,
                        yLabel=yLabel,
                        zLabel=zLabel,
                        # x_force = coGridX.flatten(),
                        # y_force = coGridY.flatten(),
                        # z_force = coGridZ.flatten(),
                        radii=coGrid.flatten())

    return vtk_writer


### Inputs compared to output probabilities
def dataVsEvents(data, events, labels, eventLabel, axisLabels, dataFormatting={}):
    """
    Line plots of data with a line plot over the top

    Uses axPlotlines unless there are more than eight datasets in ``data``,
    at which point ``dataSpectrumVsEvents`` is called.

    Parameters
    ----------
    data : array of size m*n of floats
        The main datasets to be plotted
    events : array of size n of floats
        An variable that changes alongside ``data``, but whose values are
        meaningful for all sets in data
    labels : list of strings
        Description of each of the m datasets in data
    eventLabel : string
        A description of the meaning of events
    axisLabels : dictionary
        A dictionary of axis properties, all optional, containing those in
        ``axPlotlines`` as well as:
        ``y2label`` : `string` The label of the scale of events
    dataFormatting : dictionary of string keys and values as lists of strings or floats
        A dictionary of properties for each data line to be plotted, all optional, if there are fewer than 8 datasets.
        The optional parameters are those defined in axPlotlines.

    Returns
    -------
    fig : matplotlib figure object

    """

    data = array(data)

    dataShape = shape(data)
    if len(dataShape) == 1:
        data = array([data])
    elif len(dataShape) == 2 and dataShape[1] == 1:
        data = data.T

    if len(data) > 8:
        fig = dataSpectrumVsEvents(data, events, eventLabel, axisLabels)
        return fig

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    y2Label = axisLabels.pop("y2Label", "Event value")

    eventTimes = range(1, len(events)+1)
    eventX = [i - 0.5 for i in eventTimes]

    axPlotlines(ax, eventTimes, data, labels, axisLabels, dataFormatting=dataFormatting)

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label=eventLabel, color='k', linewidth=2, markersize=5)
    bottom, top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01, top + 0.01))

    axb.set_ylabel(y2Label)

    legend(ax, axb)

    #fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


def dataSpectrumVsEvents(data, events, eventLabel, axisLabels):
    """
    Spectogram of data with a line plot over the top

    Uses axSpectrum for the spectrogram

    Parameters
    ----------
    data : array of size m*n of floats
        The main datasets to be plotted
    events : array of size n of floats
        An variable that changes alongside ``data``, but whose values are
        meaningful for all sets in data
    eventLabel : string
        A description of the meaning of events
    axisLabels : dictionary
        A dictionary of axis properties, all optional, containing those in
        ``axSpectrum`` as well as:
        ``y2label`` : `string` The label of the scale of events

    Returns
    -------
    fig : matplotlib figure object

    """

    y2Label = axisLabels.pop("y2Label", "Event value")

    eventX = fromfunction(lambda i: i+0.5, (len(events),))

    fig = plt.figure()  # figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111)

    axSpectrum(ax, data, axisLabels)

    axb = ax.twinx()
    pltLine = axb.plot(eventX, events, 'o', label=eventLabel, color='k', linewidth=2, markersize=5)
    bottom, top = axb.get_ylim()
    axb.set_ylim((bottom - 0.01, top + 0.01))

    axb.set_ylabel(y2Label)

    fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=1.0)

    return fig


### Pandas realted plotting functions
def pandasPlot(data, axisLabels = {}):
    """
    A wrapper round a pandas plotting function

    Parameters
    ----------
    data : pandas dataSeries
    axisLabels : dictionary, optional
        A dictionary of axis properties, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,
        ``yMin`` : `float`,
        ``yMax`` : `float`,
        ``xMin`` : `float`,
        ``xMax`` : `float`

    Returns
    -------
    fig : matplotlib figure object

    Notes
    -----
    Inspired by axPlotlines
    """

    xLabel = axisLabels.pop("xLabel", "Event")
    yLabel = axisLabels.pop("yLabel", "Value")
    title = axisLabels.pop("title", "")

#    yMin = axisLabels.pop("yMin", amin(Y))
#    yMax = axisLabels.pop("yMax", amax(Y))
#    xMin = axisLabels.pop("xMin", min(x))

    ax = data.plot(title=title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig = plt.gcf()

    return fig

### Background to plotting functions
# Taking the final stages of plotting and providing a nice interface to do it all in one call
# rather than a few dozen


def axScatter(ax, X, Y, z, minZ=0, CB=None, CB_label="", cmap=local_cmap):
    """
    Produces a scatter plot on the axis with the colour of the dots being a
    third axis

    Parameters
    ----------
    ax : matplotlib axis object
    X : array of size m*n of floats
    Y : array of size m*n of floats
    z : list or array of floats
        The magnitude values for the X,Y locations
    minZ : float, optional
        The lowest valid value of the colour map
    CB : matplotlib colorbar or ``None``, optional
        The colorbar object for the figure if one has already been created. Default is ``None``
    CB_label : string, optional
        The colour bar label. Unused in this instance
    cmap : matplotlib colormap object or name of known colormap
        The colormap to be used by this to represent the z data. Default local_cmap defined in this plotting module

    Returns
    -------
    sc : matplotlib scatter object
    CB : matplotlib colorbar
        The colorbar object for the figure if one has already been created
    """

    maxZ = amax(z)

    # For changing the dot size
    # minC = 1
    # maxC = 301
    # zScale = (maxC - minC) / float(maxZ - minZ)
    # C = minC + (z-amin(z))*zScale

    sc = ax.scatter(X.flatten(), Y.flatten(), s=10, c=z, cmap=cmap, edgecolors="face")

    if CB is None:
        CB = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.8, extend='both')
        CB.set_label(CB_label)
    else:
        # Check if the range needs increasing
        oldCMin, oldCMax = sc.get_clim()
        minZ = oldCMin if oldCMin < minZ else minZ
        maxZ = oldCMax if oldCMax > maxZ else maxZ
        sc.set_clim([minZ, maxZ])

    return sc, CB


def axImage(ax, X, Y, z, minZ=0, CB=None, CB_label="", cmap=local_cmap):
    """
    Produces an image on the axis

    Parameters
    ----------
    ax : matplotlib axis object
    X : array of size m*n of floats
    Y : array of size m*n of floats
    z : list or array of floats
        The magnitude values for the X,Y locations
    minZ : float, optional
        The lowest valid value of the colour map
    CB : matplotlib colorbar or ``None``, optional
        The colorbar object for the figure if one has already been created. Default is ``None``
    CB_label : string, optional
        The colour bar label
    cmap : matplotlib colormap object or name of known colormap
        The colormap to be used by this to represent the z data. Default local_cmap defined in this plotting module

    Returns
    -------
    IM : matplotlib image object
    CB : matplotlib colorbar
        The colorbar object for the figure if one has already been created
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
    Xfleshed, Yfleshed = meshgrid(linspace(xMinIm, xMaxIm, xJumps), linspace(yMinIm, yMaxIm, yJumps))

#    zPoints = [(a,b) for a,b in izip(X.flatten(),Y.flatten())]
    zPoints = [(a, b) for a, b in izip(X, Y)]
    gridZ = griddata(zPoints, z, (Xfleshed, Yfleshed), method='nearest')

#    qm = plt.pcolormesh(gridZ, cmap = local_cmap)
    im = ax.imshow(gridZ, interpolation='nearest',
                   origin='lower',
                   cmap=cmap,
                   extent=[xMinIm, xMaxIm, yMinIm, yMaxIm],
                   vmin=minZ,
                   aspect='auto')
    if CB is None:
        CB = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, extend='both')
        CB.set_label(CB_label)
    else:
        # Check if the range needs increasing
        oldCMin, oldCMax = im.get_clim()
        minZ = oldCMin if oldCMin < minZ else minZ
        maxZ = oldCMax if oldCMax > amax(z) else amax(z)
        im.set_clim([minZ, maxZ])

    return im, CB


def axContour(ax, X, Y, z, minZ=0, CB=None, CB_label="", cmap=local_cmap, maxContours=20):
    """
    Produces a contour plot on the axis

    Parameters
    ----------
    ax : matplotlib axis object
    X : array of size m*n of floats
    Y : array of size m*n of floats
    z : list or array of floats
        The magnitude values for the X,Y locations
    minZ : float, optional
        The value of the lowest contour
    CB : matplotlib colorbar or ``None``, optional
        The colorbar object for the figure if one has already been created. Default is ``None``
    CB_label : string, optional
        The colour bar label
    cmap : matplotlib colormap object or name of known colormap
        The colormap to be used by this to represent the z data. Default local_cmap defined in this plotting module
    maxContours : int, optional
        The maximum number of contours - 2 that can be shown. Default 20

    Returns
    -------
    CS : matplotlib contour object
    CB : matplotlib colorbar
        The colorbar object for the figure if one has already been created
    """

    xMin = amin(X)
    xMax = amax(X)
    yMin = amin(Y)
    yMax = amax(Y)
    maxZ = amax(z)

    xi = linspace(xMin, xMax, 50)
    yi = linspace(yMin, yMax, 50)
    ZGrid = griddata((X, Y), z, (xi[None, :], yi[:, None]), method='cubic')
    XGrid, YGrid = meshgrid(xi, yi)

    if maxZ == minZ:
        maxZ += 1
    zSorted = sort(z)
    diffSort = sort(zSorted[1:]-zSorted[:-1])
    dz = diffSort[0]
    potentialJumps = (maxZ - minZ)/dz
    if potentialJumps > maxContours:
        zJumps = maxContours
        dz = (maxZ - minZ)/(zJumps - 2)
    else:
        zJumps = potentialJumps + 2
    levels = linspace(minZ, maxZ+dz, zJumps)

    CS = ax.contour(XGrid, YGrid, ZGrid, levels,
                    origin='lower',
                    colors='k',
                    linewidths=2,
                    extent=[xMin, xMax, yMin, yMax],
                    extend='both')
    plt.clabel(CS,
               CS.levels,  # [1::2],  # label every second level
               inline=1,
               fmt='%2.1f',
               fontsize=14,
               colors='k')

#    CS.cmap.set_under('yellow')
#    CS.cmap.set_over('cyan')
    CS.cmap.set_bad("red")

    if CB is None:
        CB = plt.colorbar(CS, ax=ax, orientation='horizontal', shrink=0.8, extend='both')
        CB.set_label(CB_label)
    else:
        # Check if the range needs increasing
        oldCMin, oldCMax = CS.get_clim()
        minZ = oldCMin if oldCMin < minZ else minZ
        maxZ = oldCMax if oldCMax > maxZ else maxZ
        CS.set_clim([minZ, maxZ])

#    # Improving the position of the contour bar legend
#    l,b,w,h = ax.get_position().bounds
#    ll,bb,ww,hh = CB.ax.get_position().bounds
#    CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

    return CS, CB


def axQuiver(ax, X, Y, dX, dY, axisLabels, CB_label=""):
    """
    Quiver plot or flow plot depending on your terminology

    A work in progress

    Parameters
    ----------
    ax : matplotlib axis object
    X : array of size m*n of floats
        The horizontal axis locations of the points
    Y : array of size m*n of floats
        The vertical axis locations of the points
    dX : array of size m*n of floats
        The horizontal axis magnitudes for the arrows
    dY : array of size m*n of floats
        The vertical axis magnitudes for the arrows
    axisLabels : dictionary
        A dictionary of axis titles, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,
    CB_label : basestring
        The label of the colourbar. Default is empty

    """
    # Work in progress

    xLabel = axisLabels.pop("xLabel", "Event")
    yLabel = axisLabels.pop("yLabel", "Value")
    title = axisLabels.pop("title", "")
    
    im = ax.quiver(X, Y, dX, dY, pivot='mid', cmap=local_cmap)
    
    CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
    CBI.set_label(CB_label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid()


def axPlotlines(ax, x, Y, labels, axisLabels, dataFormatting={}):
    """
    Produces a set of lineplots on the axis

    Parameters
    ----------
    ax : matplotlib axis object
    x : array of length n of floats
    Y : array of size m*n of floats
    labels : list of strings
        Description of each of the m datasets in ``Y``
    axisLabels : dictionary
        A dictionary of axis properties, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,
        ``yMin`` : `float`,
        ``yMax`` : `float`,
        ``xMin`` : `float`,
        ``xMax`` : `float`
    dataFormatting : dictionary of string keys and values as lists of strings or floats
        A dictionary of properties for each line to be plotted, all optional. See the start of the plotting module to
        see all options for each option. Each list should be at least as long as ``m`` in ``Y``

        ``linetype`` : If the line is dotted and if so how
        ``colours`` : The colour of each line
        ``linewidth`` : The thickness of the lines
        ``markersize`` : The size of the dots at each datapoint.
    """

    xLabel = axisLabels.pop("xLabel", "Event")
    yLabel = axisLabels.pop("yLabel", "Value")
    title = axisLabels.pop("title", "")

    lineType = dataFormatting.pop('linetype', lpl)
    colourList = dataFormatting.pop('colours', colours)
    linewidth = dataFormatting.pop('linewidth', lpl_linewidth)
    markersize = dataFormatting.pop('markersize', [3 for i in xrange(len(lpl))])


    yMin = axisLabels.pop("yMin", amin(Y))
    yMax = axisLabels.pop("yMax", amax(Y))
    xMin = axisLabels.pop("xMin", min(x))
    Ys = Y.shape
    if len(Ys) == 1:
        xMax = axisLabels.pop("xMax", Ys[0])
        pltLines = ax.plot(x, Y, lineType[0], color=colourList[0], linewidth=linewidth[0], markersize=markersize[0])
    elif len(Ys) == 2:
        xMax = axisLabels.pop("xMax", Ys[1])
        for i, y in enumerate(Y):
            pltLines = ax.plot(x, y, lineType[i], label=labels[i], color=colourList[i], linewidth=linewidth[i], markersize=markersize[i])
    else:
        return

    ax.set_ylim([yMin, yMax])
    ax.set_xlim([xMin, xMax])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)


def axSpectrum(ax, Y, axisLabels):
    """
    Produces a spectrogram-like plot on the axis

    This is made by creating a histogram for each x-axis element.

    Parameters
    ----------
    ax : matplotlib axis object
    Y : array of size m*n of floats
    axisLabels : dictionary
        A dictionary of axis properties, all optional, containing:

        ``xlabel`` : `string`,
        ``ylabel`` : `string`,
        ``title`` : `string`,
        ``yMin`` : `float`,
        ``yMax`` : `float`,
        ``xMin`` : `float`,
        ``xMax`` : `float`,
        ``bins`` : `int`,
        ``probDensity`` : `binary`
        Sets if the histogram type is showing frequency or probability
        density.
    """

    xLabel = axisLabels.pop("xLabel", "Events")
    yLabel = axisLabels.pop("yLabel", "Density across parameter range")
    title = axisLabels.pop("title", "")

    minVal = axisLabels.pop("yMin", 0)
    maxVal = axisLabels.pop("yMax", amax(Y))
    xMin = axisLabels.pop("xMin", 0)
    xMax = axisLabels.pop("xMax", len(Y[0]))

    bins = axisLabels.pop("bins", 25)
    probDensity = axisLabels.pop("probDensity", False)

    data = array(Y)

    # Create this histogram-like data
    histData = [histogram(d, bins, range=(minVal, maxVal), density=True) for d in data.T]

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
                   extent=[xMin, xMax, minVal, maxVal],
                   aspect='auto')
    col = plt.colorbar(im, orientation='horizontal')
    if probDensity:
        col.set_label("Probability density")
    else:
        col.set_label("Bin probability")

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)


def legend(*axis, **kwargs):
    """
    Adds a legend to each subplot in a figure

    Parameters
    ----------
    axis : a list of matplotlib axis objects
    """

    embed = kwargs.pop('embed', False)

    lineList = []
    pltLabels = []
    for ax in axis:

        line, pltLabel = ax.get_legend_handles_labels()
        lineList.extend(line)
        pltLabels.extend(pltLabel)

        # for plotting the legend outside the box
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    if not lineList:
        return

    ax = axis[0]

    # For plotting the legend inside the plot
    if embed:
        ax.legend(lineList, pltLabels, loc='best', fancybox=True)
    else:
        # for plotting the legend outside the box
        # Put a legend below current axis
        ax.legend(lineList, pltLabels, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                  fancybox=True, shadow=True, ncol=2)


def addPoint(xPos, yPos, ax, colour="red"):
    """


    Inspired by http://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot

    Parameters
    ----------
    xPos
    yPos
    ax
    colour

    Returns
    -------

    """

    for x, y in izip(xPos, yPos):
        ax.plot(x, y, 'o', color=colour)


def addCircle(xPos, yPos, ax, circleSize=[2], colour=["red"]):

    for x, y, s, c in izip(xPos, yPos, circleSize, colour):
        circle = plt.Circle((x, y), s, color=c)
        ax.add_artist(circle)

# if __name__ == '__main__':

#    x = array([1, 2, 4])
#    y = array([0.1, 0.2, 0.3])
#    z = array([0.3, 0.4, 0.5])
#     x = array([1, 2, 3])
#     y = array([4, 5, 6])
#     z = array([7, 8, 9])
#     f = array([1, 2, 3, 2, 3, 4, 3, 4, 5, 2, 3, 4, 3, 4, 5, 4, 5, 6, 3, 4, 5, 4, 5, 6, 5, 6, 7])
#
#     fig = paramD3Dynamics(x, y, z, f, 'TestX', 'TestY', 'TestZ')
#     fig.outputTrees("outputs/testDataStruc")

#    fig = paramD2Dynamics(x,y,z, 'TestX', 'TestY', contour=False, heatmap = False, scatter = True)

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
