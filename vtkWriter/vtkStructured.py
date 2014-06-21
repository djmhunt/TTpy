#!/usr/bin/env python

"""
@author: Dominic

Based on vtktools.py
"""

import xml.etree.ElementTree as ET

from numpy import size, amin, amax

from itertools import izip

from vtkAbstract import VTK_XML_Serial_General

class VTK_XML_Serial_Structured(VTK_XML_Serial_General):
    """
    USAGE:
    vtk_writer = VTK_XML_Serial_Structured()
    vtk_writer.snapshot(x, y, z, optional arguments...)
    vtk_writer.outputTrees("filename")
    vtk_writer.writePVD("filename")
    """
    def __init__(self):
        self.fileNames = []
        self.type = "s"

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

        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()

        datShape = X.shape

        extent = repr(0) + " " + repr(datShape[0]-1) + " " + \
                    repr(0) + " " + repr(datShape[1]-1) + " " + \
                    repr(0) + " " + repr(datShape[2]-1)

        # Root element
        root_element = ET.Element("VTKFile")
        root_element.set("type", "StructuredGrid")
        root_element.set("version", "0.1")
        root_element.set("byte_order", "LittleEndian")

        # Structured grid element
        structuredGrid = ET.SubElement(root_element, "StructuredGrid")
        structuredGrid.set("WholeExtent", extent)


        # Piece 0 (only one)
        piece = ET.SubElement(structuredGrid, "Piece")
        piece.set("NumberOfPoints", repr(size(x)))
        piece.set("NumberOfCells", "0")
        piece.set("Extent", extent)

        ### Points ####
        points = ET.SubElement(piece, "Points")

        # Point location data
        point_coords = ET.SubElement(points, "DataArray")
        point_coords.set("type", "Float32")
        point_coords.set("format", "ascii")
        point_coords.set("NumberOfComponents", "3")

        point_coords_data = self.coords_to_string(x, y, z)
        point_coords.text = point_coords_data

        #### Cells ####
        cells = ET.SubElement(piece, "Cells")

        # Cell locations
        cell_connectivity = ET.SubElement(cells, "DataArray")
        cell_connectivity.set("type", "Int32")
        cell_connectivity.set("Name", "connectivity")
        cell_connectivity.set("format", "ascii")

        # Cell location data
        cell_connectivity.text = "0"

        cell_offsets = ET.SubElement(cells, "DataArray")
        cell_offsets.set("type", "Int32")
        cell_offsets.set("Name", "offsets")
        cell_offsets.set("format", "ascii")
        offsets = "0"
        cell_offsets.text = offsets

        cell_types = ET.SubElement(cells, "DataArray")
        cell_types.set("type", "UInt8")
        cell_types.set("Name", "types")
        cell_types.set("format", "ascii")
        types = "1"
        cell_types.text = types

        #### Data at Points ####
        point_data = ET.SubElement(piece, "PointData")

        # Points
        point_coords_2 = ET.SubElement(point_data, "DataArray")
        point_coords_2.set("Name", "Points")
        point_coords_2.set("NumberOfComponents", "3")
        point_coords_2.set("type", "Float32")
        point_coords_2.set("format", "ascii")

        point_coords_2_Data = self.coords_to_string(x, y, z)
        point_coords_2.text = point_coords_2_Data

        # Particle jump vectors
        if "x_jump" in kwargs and "y_jump" in kwargs and "z_jump" in kwargs:
            x_jump = kwargs["x_jump"]
            y_jump = kwargs["y_jump"]
            z_jump = kwargs["z_jump"]
            jumps = ET.SubElement(point_data, "DataArray")
            jumps.set("Name", "jumps")
            jumps.set("NumberOfComponents", "3")
            jumps.set("type", "Float32")
            jumps.set("format", "ascii")

            jumpData = self.coords_to_string(x_jump, y_jump, z_jump)
            jumps.text = jumpData

        # Force vectors
        if "x_force" in kwargs and "y_force" in kwargs and "z_force" in kwargs:
            x_force = kwargs["x_force"]
            y_force = kwargs["y_force"]
            z_force = kwargs["z_force"]
            forces = ET.SubElement(point_data, "DataArray")
            forces.set("Name", "forces")
            forces.set("NumberOfComponents", "3")
            forces.set("type", "Float32")
            forces.set("format", "ascii")

            forceData = self.coords_to_string(x_force, y_force, z_force)
            forces.text = forceData

        # Particle radii
        if "radii" in kwargs:
            radii = kwargs["radii"]
            radiiNode = ET.SubElement(point_data, "DataArray")
            radiiNode.set("Name", "radii")
            radiiNode.set("type", "Float32")
            radiiNode.set("format", "ascii")

            radiiData = self.array_to_string(radii)
            radiiNode.text = radiiData

        # Particle colors
        if "colors" in kwargs:
            colors = kwargs["colors"]
            colorNode= ET.SubElement(point_data, "DataArray")
            colorNode.set("Name", "colors")
            colorNode.set("type", "Float32")
            colorNode.set("format", "ascii")

            color_Data = self.array_to_string(colors)
            colorNode.text = color_Data

        #### Cell data (dummy) ####
        cell_data = ET.SubElement(piece, "CellData")

        # Store tree

        prettyRoot = self.prettify(root_element)
        self.mainTree = ET.ElementTree(prettyRoot)
