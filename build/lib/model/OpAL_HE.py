# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper Opponent actor learning (OpAL): Modeling
                interactive effects of striatal dopamine on reinforcement
                learning and choice incentive.
                Collins, A. G. E., & Frank, M. J. (2014).
                Psychological Review, 121(3), 337–66.
                doi:10.1037/a0037015

"""
import logging

import numpy as np

from model.modelTemplate import Model


class OpAL_HE(Model):

    """The Opponent actor learning model without Hebbian learning

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
        Learning rate parameter, used as either the
    alphaGoNogoDiff : float, optional
        The difference between ``alphaGo`` and ``alphaNogo``. Default is ``None``.
        If not ``None`` will overwrite ``alphaNogo``
        :math:`\\alpha_N = \\alpha_G - \\alpha_\\delta`
    alphaCrit : float, optional
        The critic learning rate. Default is ``alpha``
    alphaGo : float, optional
        Learning rate parameter for Go, the positive part of the actor learning
        Default is ``alpha``
    alphaNogo : float, optional
        Learning rate parameter for Nogo, the negative part of the actor learning
        Default is ``alpha``
    alphaGoDiff : float, optional
        The difference between ``alphaCrit`` and ``alphaGo``. The default is ``None``
        If not ``None``  and ``alphaNogoDiff`` is also not ``None``, it will 
        overwrite the ``alphaGo`` parameter
        :math:`\\alpha_G = \\alpha_C + \\alpha_\\deltaG`
    alphaNogoDiff : float, optional
        The difference between ``alphaCrit`` and ``alphaNogo``. The default is ``None``
        If not ``None``  and ``alphaGoDiff`` is also not ``None``, it will 
        overwrite the ``alphaNogo`` parameter
        :math:`\\alpha_N = \\alpha_C + \\alpha_\\deltaN`
    epsilon : float, optional
        Sensitivity parameter for probabilities. Also known as an exploration-
        exploitation parameter. Defined as :math:`\\epsilon` in the paper
    rho : float, optional
        The asymmetry between the actor weights. :math:`\\rho = \\epsilon_G - \\epsilon = \\epsilon_N + \\epsilon`
    number_actions : integer, optional
        The maximum number of valid actions the model can expect to receive.
        Default 2.
    number_cues : integer, optional
        The initial maximum number of stimuli the model can expect to receive.
         Default 1.
    number_critics : integer, optional
        The number of different reaction learning sets.
        Default number_actions*number_cues
    action_codes : dict with string or int as keys and int values, optional
        A dictionary used to convert between the action references used by the
        task or dataset and references used in the models to describe the order
        in which the action information is stored.
    prior : array of floats in ``[0, 1]``, optional
        The prior probability of of the states being the correct one.
        Default ``ones((number_actions, number_cues)) / number_critics)``
    expect: array of floats, optional
        The initialisation of the the expected reward.
        Default ``ones((number_actions, number_cues)) / number_critics``
    expectGo : array of floats, optional
        The initialisation of the the expected go and nogo.
        Default ``ones((number_actions, number_cues)) / number_critics``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
    rewFunc : function, optional
        The function that transforms the reward into a form the model can
        understand. Default is blankRew
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is model.decision.discrete.weightProb

    Notes
    -----
    Actor: The chosen action is updated with

    .. math::

        \\delta_{d,t} = r_t-E_{d,t}

        E_{d,t+1} = E_{d,t} + \\alpha_E \\delta_{d,t}

    Critic: The chosen action is updated with

    .. math::
        G_{d,t+1} = G_{d,t} + \\alpha_G \\delta_{d,t}

        N_{d,t+1} = N_{d,t} - \\alpha_N \\delta_{d,t}

    Probabilities: The probabilities for all actions are calculated using

    .. math::
        A_{d,t} = (1+\\rho) G_{d,t}-(1-\\rho) N_{d,t}

        P_{d,t} = \\frac{ e^{\\epsilon A_{d,t} }}{\\sum_{d \\in D}e^{\\epsilon A_{d,t}}}
    """

    def __init__(self, alpha=0.3, epsilon=0.3, rho=0, alphaCrit=None, alphaGo=None, alphaNogo=None, alphaGoDiff=None,
                 alphaNogoDiff=None, alphaGoNogoDiff=None, expect=None, expectGo=None, **kwargs):

        super(OpAL_HE, self).__init__(**kwargs)

        if alphaCrit is None:
            alphaCrit = alpha
        self.alphaCrit = alphaCrit

        if alphaGo is not None and alphaNogo is not None:
            self.alphaGo = alphaGo
            self.alphaNogo = alphaNogo
        elif alphaGoNogoDiff is not None and (alphaGo is not None or alphaNogo is not None):
            if alphaGo is not None:
                self.alphaGo = alphaGo
                self.alphaNogo = alphaGo - alphaGoNogoDiff
            elif alphaNogo is not None:
                self.alphaGo = alphaNogo + alphaGoNogoDiff
                self.alphaNogo = alphaNogo
        elif alphaGoDiff is not None and alphaNogoDiff is not None:
            self.alphaGo = alpha + alphaGoDiff
            self.alphaNogo = alpha + alphaNogoDiff
        else:
            self.alphaGo = alpha
            self.alphaNogo = alpha

        self.epsilon = epsilon
        self.rho = rho

        if expect is None:
            expect = np.ones((self.number_actions, self.number_cues)) / self.number_critics
        self.expect = expect
        if expectGo is None:
            expectGo = np.ones((self.number_actions, self.number_cues))
        self.expectGo = expectGo

        self.expectations = np.array(self.expect)
        self.go = np.array(self.expectGo)
        self.nogo = np.array(self.expectGo)
        self.actionValues = np.ones(self.expectations.shape)

        self.parameters["alphaCrit"] = self.alphaCrit
        self.parameters["alphaGo"] = self.alphaGo
        self.parameters["alphaNogo"] = self.alphaNogo
        self.parameters["epsilon"] = self.epsilon
        self.parameters["rho"] = self.rho
        self.parameters["expectation"] = self.expect
        self.parameters["expectationGo"] = self.expectGo

        # Recorded information
        self.recGo = []
        self.recNogo = []
        self.recActionValues = []

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["Go"] = np.array(self.recGo)
        results["Nogo"] = np.array(self.recNogo)
        results["ActionValues"] = np.array(self.recActionValues)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recGo.append(self.go.copy())
        self.recNogo.append(self.nogo.copy())
        self.recActionValues.append(self.actionValues.copy())

    def rewardExpectation(self, observation):
        """Calculate the estimated reward based on the action and stimuli

        This contains parts that are task dependent

        Parameters
        ----------
        observation : {int | float | tuple}
            The set of stimuli

        Returns
        -------
        actionExpectations : array of floats
            The expected rewards for each action
        stimuli : list of floats
            The processed observations
        activeStimuli : list of [0, 1] mapping to [False, True]
            A list of the stimuli that were or were not present
        """

        activeStimuli, stimuli = self.stimulus_shaper.processStimulus(observation)

        actionExpectations = self._actExpectations(self.expectations, stimuli)

        return actionExpectations, stimuli, activeStimuli

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

        modReward = self.reward_shaper.processFeedback(reward, action, stimuli)

        delta = modReward - expectation

        return delta

    def updateModel(self, delta, action, stimuli, stimuliFilter):
        """
        Parameters
        ----------
        delta : float
            The difference between the reward and the expected reward
        action : int
            The action chosen by the model in this trialstep
        stimuli : list of float
            The weights of the different stimuli in this trialstep
        stimuliFilter : list of bool
            A list describing if a stimulus cue is present in this trialstep

        """

        # Find the new activities
        self._critic(action, delta, stimuli)

        self._actor(action, delta, stimuli)

        self._actionValues(self.go, self.nogo)

        # Calculate the new probabilities
        self.probabilities = self.actorStimulusProbs()

    def _critic(self, action, delta, stimuli):

        newExpectations = self.expectations[action] + self.alphaCrit*delta*stimuli/np.sum(stimuli)
        newExpectations = newExpectations * (newExpectations >= 0)
        self.expectations[action] = newExpectations

    def _actor(self, action, delta, stimuli):

        self.go[action] += self.alphaGo * delta * stimuli/np.sum(stimuli)
        self.nogo[action] -= self.alphaNogo * delta * stimuli/np.sum(stimuli)

    def _actionValues(self, go, nogo):

        rho = self.rho

        actionValues = (1 + rho) * go - (1 - rho) * nogo

        self.actionValues = actionValues

    def _actExpectations(self, expectations, stimuli):

        # If there are multiple possible stimuli, filter by active stimuli and calculate
        # calculate the expectations associated with each action.
        if self.number_cues > 1:
            actionExpectations = self.actStimMerge(expectations, stimuli)
        else:
            actionExpectations = expectations

        return actionExpectations

    def calcProbabilities(self, actionValues):
        # type: (np.ndarray) -> np.ndarray
        """
        Calculate the probabilities associated with the actions

        Parameters
        ----------
        actionValues : 1D ndArray of floats

        Returns
        -------
        probArray : 1D ndArray of floats
            The probabilities associated with the actionValues
        """

        cbest = actionValues == max(actionValues)
        deltaEpsilon = self.epsilon * (1 / self.number_actions)
        bestEpsilon = (1 - self.epsilon) / np.sum(cbest) + deltaEpsilon
        probArray = bestEpsilon * cbest + deltaEpsilon * (1 - cbest)

        return probArray

    def actorStimulusProbs(self):
        """
        Calculates in the model-appropriate way the probability of each action.

        Returns
        -------
        probabilities : 1D ndArray of floats
            The probabilities associated with the action choices

        """

        actExpectations = self._actExpectations(self.actionValues, self.stimuli)
        probabilities = self.calcProbabilities(actExpectations)

        return probabilities
