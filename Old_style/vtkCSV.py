#!/usr/bin/env python

"""
@author: Dominic

Based on vtktools.py
"""


from numpy import ones

from itertools import izip

from vtkAbstract import VTK_XML_Serial_General

class VTK_CSV(VTK_XML_Serial_General):
    """
    USAGE:
    vtk_writer = VTK_CSV()
    vtk_writer.snapshot(x, y, z, optional arguments...)
    vtk_writer.outputTrees("filename")
    vtk_writer.writePVD("filename")
    """
    def __init__(self):
        self.fileNames = []


    def snapshot(self, X,Y,Z, **kwargs):
        """
        ARGUMENTS:
        x               array of x coordinates of particle centers
        y               array of y coordinates of particle centers
        z               array of z coordinates of particle centers
        radii           optional array of particle radii
        """

        xLabel = kwargs.get("xLabel","x")
        yLabel = kwargs.get("yLabel","y")
        zLabel = kwargs.get("zLabel","z")

        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()

        if "radii" in kwargs:
            radii = kwargs["radii"]
        else:
            radii = ones(len(x))

        coordsList = (repr(i) + ',' + repr(j) + ',' + repr(k) + ',' + repr(f) for i,j,k,f in izip(x,y,z,radii))

        coords = '\n'.join(coordsList)

        labels = xLabel + "," + yLabel + "," + zLabel + ",radii\n"

        self.output = labels + coords

    def outputTrees(self, fileName):
        """
        ARGUMENTS
        fileName        file name and/or path/filename
        """
        self.fileNames.append(fileName)

        with open(fileName + ".csv",'w') as o:
            o.write(self.output)