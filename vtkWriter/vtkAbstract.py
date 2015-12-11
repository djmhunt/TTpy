#!/usr/bin/env python

"""
@author: Dominic

Based on vtktools.py
"""
from __future__ import division, print_function

import xml.etree.ElementTree as ET

from xml.dom import minidom
from itertools import izip

class VTK_XML_Serial_General:
    """
    Provides the general class structure that specific instances inherit
    """
    def __init__(self):
        self.fileNames = []
        self.type = ""

    def prettify(self,root):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(root)
        reparsed = minidom.parseString(rough_string)
        xml = reparsed.toprettyxml(indent="  ")
        return ET.fromstring(xml)

    def coords_to_string(self, x,y,z):
#        string = str()
#        for i in range(size(x)):
#            string = string + repr(x[i]) + ' ' + repr(y[i]) + ' ' + repr(z[i]) + ' '

        vals = (repr(i) + ' ' + repr(j) + ' ' + repr(k) for i,j,k in izip(x,y,z))
        string = ' '.join(vals)
        return string

    def array_to_string(self, a):
#        string = str()
#        for i in range(size(a)):
#            string = string + repr(a[i]) + ' '

        vals = (repr(i) for i in a)
        string = ' '.join(vals)
        return string

    def snapshot(self, X,Y,Z, **kwargs):
        """
        ARGUMENTS:
        x               array of x coordinates of particle centers
        y               array of y coordinates of particle centers
        z               array of z coordinates of particle centers
        x_jump          optional array of x components of particle jump vectors
        y_jump          optional array of y components of particle jump vectors
        z_jump          optional array of z components of particle jump vectors
        x_force         optional array of x components of force vectors
        y_force         optional array of y components of force vectors
        z_force         optional array of z components of force vectors
        radii           optional array of particle radii
        colors          optional array of scalars to use to set particle colors
                        The exact colors will depend on the color map you set up in Paraview.
        """

        # Root element
        root_element = ET.Element("VTKFile")

        # Store tree

        prettyRoot = self.prettify(root_element)
        self.mainTree = ET.ElementTree(prettyRoot)

    def outputTrees(self, fileName):
        """
        ARGUMENTS
        fileName        file name and/or path/filename
        """
        self.fileNames.append(fileName)
        self.mainTree.write(fileName + ".vt" + self.type)

    def writePVD(self, fileName):

        pvd_root = ET.Element("VTKFile")
        pvd_root.set("type", "Collection")
        pvd_root.set("version", "0.1")
        pvd_root.set("byte_order", "LittleEndian")

        collection = ET.SubElement(pvd_root, "Collection")

        for i in range(len(self.fileNames)):
            dataSet = ET.SubElement(collection, "DataSet")
            dataSet.set("timestep", str(i))
            dataSet.set("group", "")
            dataSet.set("part", "0")
            dataSet.set("file", str(self.fileNames[i]))

        pvdTree = ET.ElementTree(pvd_root)
        pvdTree.write(fileName + ".pvd" )

if __name__ == '__main__':


    # build a tree structure
    root = ET.Element("html")

    head = ET.SubElement(root, "head")

    title = ET.SubElement(head, "title")
    title.text = "Page Title"

    body = ET.SubElement(root, "body")
    body.set("bgcolor", "#ffffff")

    body.text = "Hello, World!"

    # wrap it in an ElementTree instance, and save as XML
    tree = ET.ElementTree(root)
    tree.write("page.xhtml")