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
    participant :
    modelData :
    fitQuality :

    Returns
    -------
    figSets :

    """
    partName = participant['Name']

    figSets = []

    fig = plotFittingSuccess(partName, fitQuality, modelData["Actions"], modelData["ActionProb"])
    figSets.append(('FittingSuccess_part_' + partName, fig))

    return figSets


def plotFittingSuccess(partName, fitQuality, partActions, modActProb):
    """
    The success of accurately providing the participants action as the most likely

    Parameters
    ----------
    partName :
    fitQuality :
    partActions :
    modActProb :

    Returns
    -------

    """

    axisLabels = {"title": "Model fit values performance for participant " + partName}
    axisLabels["xLabel"] = "Time"
    axisLabels["yLabel"] = "Probability of participant action being the best one"
    axisLabels["y2Label"] = "Actions by the participant"
    axisLabels["yMax"] = max(partActions)
    axisLabels["yMin"] = min(partActions)
    modelLabels = ["Fit of quality " + str(fitQuality)]
    eventLabel = "Participant actions"

    fig = dataVsEvents(modActProb.T, partActions, modelLabels, eventLabel, axisLabels)

    return fig