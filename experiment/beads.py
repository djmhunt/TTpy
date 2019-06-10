# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    `Moore, S. C., & Sellen, J. L. (2006)`.
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
from __future__ import division, print_function, unicode_literals, absolute_import

#matplotlib.interactive(True)
import logging

import pandas as pd

from numpy import array, zeros
from numpy.random import rand
from experiment.experimentTemplate import experiment
from plotting import dataVsEvents, paramDynamics
from experiment.experimentPlot import experimentPlot
from utils import varyingParams


# Bead Sequences:
beadSequences = {"MooreSellen": [1,1,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0]}
defaultBeads = beadSequences["MooreSellen"]

class Beads(experiment):
    """Based on the Moore&Sellen Beads task

    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    N : int, optional
        Number of beads that could potentially be shown
    beadSequence : list or array of {0,1}, optional
        The sequence of beads to be shown. Bead sequences can also be embedded
        in the code and then referred to by name. The only current one is
        `MooreSellen`, the default sequence.
    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to
        its documentation for more details.
    """

    Name = "beads"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        N = kwargs.pop('N',None)
        beadSequence = kwargs.pop("beadSequence",defaultBeads)

        self.plotArgs = kwargs.pop('plotArgs',{})

        if isinstance(beadSequence, basestring):
            if beadSequence in beadSequences:
                self.beads = beadSequences[beadSequence]
            else:
                raise "Unknown bead sequence"
        else:
            self.beads = beadSequence

        if N:
            self.T = N
        else:
            self.T = len(self.beads)

        self.parameters = {"Name": self.Name,
                           "N": self.T,
                           "beadSequence": self.beads}

        # Set trialstep count
        self.t = -1

        # Recording variables

        self.recBeads = [-1]*self.T
        self.recAction = [-1]*self.T
        self.firstDecision = 0

        return self

    def next(self):
        """ Produces the next bead for the iterator

        Returns
        -------
        bead : {0,1}
        nextValidActions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        self.storeState()

        nextStim = self.beads[self.t]
        nextValidActions = (0, 1)

        return nextStim, nextValidActions

    def receiveAction(self, action):
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : {1,2, None}
        """

        self.recAction[self.t] = action

        if action and not self.firstDecision:
            self.firstDecision = self.t + 1

    def outputEvolution(self):
        """
        Returns all the relevant data for this experiment run

        Returns
        -------
        results : dictionary
            The dictionary contains a series of keys including Name,
            Observables and Actions.
        """

        results = {"Name": self.Name,
                   "Observables": array(self.recBeads),
                   "Actions": self.recAction,
                   "FirstDecision": self.firstDecision}

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        self.recBeads[self.t] = self.beads[self.t]

    class experimentPlot(experimentPlot):

        def _figSets(self):

            # Create all the plots and place them in in a list to be iterated

            self.figSets = []

            fig = self.plotProbJar1()
            self.figSets.append(('Actions', fig))

            fig = self.varCategoryDynamics()
            self.figSets.append(('decisionCoM', fig))

            fig = self.varDynamicPlot()
            self.figSets.append(("firstDecision", fig))

        def varDynamicPlot(self):

            params = self.modelParams[0].keys()

            paramSet = varyingParams(self.modelStore, params)
            decisionTimes = array([exp["FirstDecision"] for exp in self.expStore])

            fig = paramDynamics(paramSet, decisionTimes, **self.plotArgs)

            return fig

        def varCategoryDynamics(self):

            params = self.modelParams[0].keys()
            # We assume that the parameters are the same for all the data to be analysed,
            # otherwise this data is meaningless

            dataSet = varyingParams(self.modelStore,params)
            dataSet["decisionTimes"] = [exp["FirstDecision"] for exp in self.expStore]

            initData = pd.DataFrame(dataSet)

            maxDecTime = max(dataSet["decisionTimes"])
            if maxDecTime == 0:
                logger = logging.getLogger('categoryDynamics')
                message = "No decisions taken, so no useful data"
                logger.info(message)
                return

            dataSets = {d:initData[initData['decisionTimes'] == d] for d in range(1, maxDecTime+1)}

            CoM = pd.DataFrame([dS.mean() for dS in dataSets.itervalues()])

            CoM = CoM.set_index('decisionTimes')

            return CoM

        def plotProbJar1(self):
            """
            Plots a set of lines for the probability of jar 1.
            """

            data = [model["Probabilities"][:, 0] for model in self.modelStore]

            events = self.expStore[0]["Observables"]

            axisLabels = {"title": "Opinion of next bead being white"}
            axisLabels["xLabel"] = "Time"
            axisLabels["yLabel"] = "Probability of Jar 1"
            axisLabels["y2Label"] = "Bead presented"
            axisLabels["yMax"] = 1
            axisLabels["yMin"] = 0
            eventLabel = "Beads drawn"

            fig = dataVsEvents(data, events, self.modelLabels, eventLabel, axisLabels)

            return fig


def generateSequence(numBeads, oneProb, switchProb):
    """
    Designed to generate a sequence of beads with a probability of switching
    jar at any time.

    Parameters
    ----------
    numBeads : int
        The number of beads in the sequence
    oneProb : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar.
    switchProb : float in ``[0,1]``
        The probability that the drawn beads change the jar they are being
        drawn from

    Returns
    -------
    sequence : list of ``{0,1}``
        The generated sequence of beads
    """

    sequence = zeros(numBeads)

    probs = rand(numBeads, 2)
    bead = 1

    for i in range(numBeads):
        if probs[i, 1] < switchProb:
            bead = 1-bead

        if probs[i, 0] < oneProb:
            sequence[i] = bead
        else:
            sequence[i] = 1-bead

    return sequence


def beadStimDirect():
    """
    Processes the beads stimuli for models expecting just the event

    Returns
    -------
    beadStim : function
        The function expects to be passed the event and a decision of ``{1,2, None}``
        and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn
    """

    def beadStim(observation, action):
        return 1, observation

    beadStim.Name = "beadStimDirect"
    return beadStim


def beadStimDualDirect():
    """
    Processes the beads stimuli for models expecting a tuple of ``[event,1-event]``

    Returns
    -------
    beadStim : function
        The function expects to be passed the event and a decision of {1,2, None}
        and then return ``[event,1-event]``, where the event is expected to be
        ``{1,0}``.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.EP
    """

    def beadStim(observation, action):
        stimulus = array([observation, 1-observation])
        return 1, stimulus

    beadStim.Name = "beadStimDualDirect"

    return beadStim


def beadStimDualInfo(oneProb):
    """
    Processes the beads stimuli for models expecting the reward information
    from two possible actions

    Parameters
    ----------
    oneProb : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar. ``event_info`` is calculated as
        ``oneProb*event + (1-oneProb)*(1-event)``

    Returns
    -------
    beadStim : function
        The function expects to be passed the event and a decision of {1,2, None}
        and then return ``[event_info,1-event_info]``.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.MS, model.MS_rev, model.BP
    """

    def beadStim(observation, action):
        stim = oneProb*observation + (1-oneProb)*(1-observation)
        stimulus = array([stim, 1-stim])
        return 1, stimulus

    beadStim.Name = "beadStimDualInfo"
    beadStim.Params = {"oneProb": oneProb}

    return beadStim

def beadRewDirect():
    """
    Processes the beads reward for models expecting just the reward

    Returns
    -------
    beadRew : function
        The function expects to be passed a tuple containing the reward and the
        last action. The function returns the reward.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.qLearn, model.qLearn2, model.decision.binary.decEta
    """

    def beadRew(reward, action, stimuli):
        return reward

    beadRew.Name = "beadRewDirect"
    return beadRew

