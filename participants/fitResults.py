# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from plotting import dataVsEvents


def plotFitting(participant, modelData, fitQuality):
    """

    Parameters
    ----------
    participant : dict
        The data for the participant
    modelData : dict
        The data for the model instance
    fitQuality : float
        A description of the quality of a fit

    Returns
    -------
    figSets : list of matplotlib figure objects
        The figures associated with the fitting of the participant

    """
    partName = participant['Name']

    figSets = []

    fig = plotFittingSuccess(partName, fitQuality, modelData["Actions"], modelData["ActionProb"])
    figSets.append(('FittingSuccess_' + partName, fig))

    return figSets


def plotFittingSuccess(partName, fitQuality, partActions, modActProb):
    """
    The success of accurately providing the participants action as the most likely

    Parameters
    ----------
    partName : string
        The name label for the participant
    fitQuality : float
        A description of the quality of a fit
    partActions : array of floats
        The actions for each timestep
    modActProb : 2D array of floats
        The action probabilities for each model. Each row represents one model.

    Returns
    -------
    fig : matplotlib figure object

    """

    axisLabels = {"title": "Model fit values performance for participant " + partName}
    axisLabels["xLabel"] = "Time"
    axisLabels["yLabel"] = "Probability of participant action being the best one"
    axisLabels["y2Label"] = "Actions by the participant"
    axisLabels["yMax"] = max(partActions)
    axisLabels["yMin"] = min(partActions)
    modelLabels = ["Fit of quality " + str(fitQuality)]
    eventLabel = "Participant actions"

    fig = dataVsEvents(modActProb, partActions, modelLabels, eventLabel, axisLabels)

    return fig