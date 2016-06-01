# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""
from __future__ import division, print_function

from numpy import amax, amin, floor, ceil, around, reshape, append

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

    modExpect = modelData["Expectations"].T
    reward = reshape(modelData["Rewards"], (1, modExpect.shape[1]))
    fig = plotExpectations(partName, fitQuality, modelData["Actions"], modExpect, reward)
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

    axisLabels = {"title": "Model fit value performance for participant " + partName}
    axisLabels["xLabel"] = "Time"
    axisLabels["yLabel"] = "Probability of participant action being the best one"
    axisLabels["y2Label"] = "Actions by the participant"
    axisLabels["yMax"] = ceil(amax(partActions))
    axisLabels["yMin"] = floor(amin(partActions))
    modelLabels = ["Fit of quality " + str(around(fitQuality, 1))]
    eventLabel = "Participant actions"

    fig = dataVsEvents(modActProb, partActions, modelLabels, eventLabel, axisLabels)

    return fig


def plotExpectations(partName, fitQuality, partActions, modExpect, reward):
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
    modExpect : 2D array of floats
        The action reward expectations for each model. Each row represents one model.
    reward : 1D array of floats shaped as a 2D array like modExpect
        The reward received by the participants at each point

    Returns
    -------
    fig : matplotlib figure object

    """

    data = append(modExpect, reward, axis=0)

    axisLabels = {"title": "Expectation values for participant " + partName + " with fit quality of " + str(around(fitQuality, 1))}
    axisLabels["xLabel"] = "Time"
    axisLabels["yLabel"] = "Expected reward for action"
    axisLabels["y2Label"] = "Actions by the participant"
    axisLabels["yMax"] = ceil(amax(modExpect))
    axisLabels["yMin"] = floor(amin(modExpect))
    modelLabels = ["Action " + str(i) for i in xrange(len(modExpect))]
    modelLabels.append("Reward")
    eventLabel = "Participant actions"

    fig = dataVsEvents(data, partActions, modelLabels, eventLabel, axisLabels)

    return fig