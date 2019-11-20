# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Value and prediction error in medial frontal cortex: integrating the single-unit and systems levels of analysis.
                `Silvetti, M., Seurinck, R., & Verguts, T. (2011)`.
                Frontiers in Human Neuroscience, 5(August), 75.
                doi:10.3389/fnhum.2011.00075
"""
from __future__ import division, print_function

import logging

import numpy as np

from collections import defaultdict

from modelTemplate import Model


class RVPM(Model):
    """The reward value and prediction model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter for updating the cue-value weights. Default 0.3
    beta : float, optional
        It is a time constant that controls how quickly the neural units
        (modeled as dynamical systems) respond to external inputs. Default 0.1
    zeta : float, optional
        Regulates the ratio between the power (amplitude) of expectation
        relative to delta units. Default 2
    w : array, optional
        The stimulus weights. Default [0.01,0.01]
    tau : float, optional
        The timing signal mean. Default 160
    z : float, optional
        The timing signal spread. Default 20
    averaging : int, optional
        The number of stimuli recorded from the beginning and end of each
        training set. Default is 3
    numActions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    numCues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    numCritics : integer, optional
        The number of different reaction learning sets.
        Default numActions*numCues
    actionCodes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((numActions, numCues)) / numCritics)``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is basicDecision
    """


    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        self.alpha = kwargRemains.pop('alpha', 0.005)
        self.beta = kwargRemains.pop('beta', 0.1)
        self.w = kwargRemains.pop('w', np.array([0.01, 0.01]))
        self.zeta = kwargRemains.pop('zeta', 2)
        self.tau = kwargRemains.pop('tau', 160)
        self.z = kwargRemains.pop('z', 100)
        self.averaging = kwargRemains.pop('averaging', 3)

        self.stimFunc = kwargRemains.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargRemains.pop('decFunc', basicDecision())

        self.T = 0  # Timing sgnal value
        self.c = 0  # The stimuli
        self.r = 0  # Reward value
        self.V = 0  # Reward prediction unit
        self.deltaP = 0  # Positive prediction error unit
        self.deltaM = 0  # Negative prediction error unit
        self.TSN = 0  # Temporally shifted neuron

        self.genStandardParameterDetails()
        self.parameters["alpha"] = self.alpha
        self.parameters["beta"] = self.beta
        self.parameters["tau"] = self.tau
        self.parameters["zeta"] = self.zeta
        self.parameters["wInit"] = self.w
        self.parameters["z"] = self.z
        self.parameters["averaging"] = self.averagings

        # Recorded information
        self._storeSetup()

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        av = self.averaging

        results = self.standardResultOutput()

        for k, v in self.generalStore.iteritems():

            results[k + '_early'] = v[:av]
            results[k + '_late'] = v[-av:]

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self._updateGeneralStore()

    def _storeSetup(self):

        self.eventStore = defaultdict(list)

        self._generalStoreSetup()

    def _generalStoreSetup(self):

        self.genStandardResultsStore()
        self.generalStore = {}

        for k in self.eventStore.iterkeys():
            self.generalStore[k] = []

    def _updateEventStore(self, event):

        self.eventStore["T"].append(self.T)
        self.eventStore["V"].append(self.V)
        self.eventStore["DP"].append(self.deltaP)
        self.eventStore["DM"].append(self.deltaM)
        self.eventStore["TSN"].append(self.TSN)
        self.eventStore["stim"].append(self.c)
        self.eventStore["rew"].append(self.r)
        self.eventStore["w"].append(self.w)
        self.eventStore["event"].append(event)
        self.eventStore["decProb"].append(self.decProbabilities)

    def _updateGeneralStore(self):

        for k, v in self.eventStore.iteritems():
            self.generalStore[k].append(np.array(v))

        for k in self.eventStore.iterkeys():
            self.eventStore[k] = []

    def rewardExpectation(self, observation):
        """Calculate the reward based on the action and stimuli

        This contains parts that are experiment dependent

        Parameters
        ----------
        observation : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        expectedReward : float
            The expected reward
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimFunc(observation, action)

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.numCues > 1:
            actionExpectations = self.actStimMerge(self.expectation, stimuli)
        else:
            actionExpectations = self.expectation

        expectedReward = actionExpectations[action]

        return expectedReward, stimuli, activeStimuli

    def delta(self, reward, expectation, action, stimuli):
        """
        Calculates the comparison between the reward and the expectation

        Parameters
        ----------
        reward : float
            The reward value
        expectation : float
            The expected reward value
        action : int
            The chosen action
        stimuli : {int | float | tuple | None}
            The stimuli received

        Returns
        -------
        delta
        """

        modReward = self.rewFunc(reward, action, stimuli)

        delta = modReward - expectation

        return delta

    def updateModel(self, delta, action, stimuliFilter):

        for t, c, r in self.stimFunc(event, self.currAction):

            self.c = c
            self.r = r
            self._processStim(t, c, r)
            self._updateEventStore(event)

    def _processStim(self, t, c, r):

        T = self._timeSigMag(t)

        dV = self._vUpdate(self.w, self.V, c)
        self.V += dV
        ddeltaP = self._deltaPUpdate(self.V, self.deltaP, r)
        self.deltaP += ddeltaP

        ddeltaM = self._deltaMUpdate(self.V, self.deltaM, T, r)
        self.deltaM += ddeltaM

        self.TSN = self._tsnUpdate(dV, ddeltaP, ddeltaM)

        self.w = self._wNew(self.w, self.V, self.deltaP, self.deltaM, c)

        self.probabilities = self.TSN

    def _wNew(self, w, V, deltaP, deltaM, c):
        new = w + self.alpha*c*V*(deltaP - deltaM)
        return new

    def _vUpdate(self, w, V, c):
        beta = self.beta
        new = -beta*V + beta*np.amax([0, np.dot(w, c)])
        return new

    def _deltaPUpdate(self, V, deltaP, r):
        beta = self.beta
        new = -beta*deltaP + beta*np.amax([0, r-self.zeta*V])
        return new

    def _timeSigMag(self, t):
        signal = np.exp((-(t-self.tau)**2)/np.square(self.z))
        return signal

    def _deltaMUpdate(self, V, deltaM, T, r):
        beta = self.beta
        new = -beta*deltaM + beta*T*np.amax([0, self.zeta*V-r])
        return new

    def _tsnUpdate(self, dV, ddeltaP, ddeltaM):
        signal = np.amax([0, self.zeta*dV]) + np.amax([0, ddeltaP]) - np.amax([0, ddeltaM])
        return signal


def blankStim():
    """ The default stimulus processor generator for RVPM

    Passes the pavlov stimuli to models that cope with stimuli and rewards
    that have a duration.

    Returns
    -------
    blankStimFunc : function
        The function yields a series of ten events ``t,0,0```, where ``t`` is
        the time. The stimulus and reward are set to 0.

    """

    def blankStimFunc(event, action):

        for t in xrange(10):

            yield t, 1, 0

    blankStimFunc.Name = "blankStim"

    return blankStimFunc


def basicDecision():
    """The default decision function for RVPM

    Returns
    -------
    beadStim : function
        The function expects to be passed the probabilities and then return
        `None`.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def basicDecisionFunc(prob):

        return None

    basicDecisionFunc.Name = "basicDecision"

    return basicDecisionFunc


def blankRew():
    """
    Default reward processor. Does nothing. Returns reward

    Returns
    -------
    blankRewFunc : function
        The function expects to be passed the reward and then return it.

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankRewFunc(reward):
        return reward

    blankRewFunc.Name = "blankRew"
    return blankRewFunc
