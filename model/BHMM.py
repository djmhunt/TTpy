# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Based on the paper The role of the ventromedial prefrontal cortex
            in abstract state-based inference during decision making in humans.
            Hampton, A. N., Bossaerts, P., & O’Doherty, J. P. (2006).
            The Journal of Neuroscience : The Official Journal of the Society for Neuroscience,
            26(32), 8360–7.
            doi:10.1523/JNEUROSCI.1010-06.2006

:Notes: This proposed model is itself based on the model in the book:
        An Introduction to Hidden Markov Models and Bayesian Networks
        DOI:10.1142/S0218001401000836

"""
from __future__ import division, print_function

import logging

from numpy import exp, array, ones
from numpy.random import normal

from modelTemplate import Model
from model.decision.binary import decSingle
from utils import callableDetailsString


class BHMM(Model):
    """The Bayesian Hidden Markov Model model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    invBeta : float, optional
        Inverse of sensitivity parameter.
        Defined as :math:`\\frac{1}{\\beta+1}`. Default ``0.2``
    eta : float, optional
        Decision threshold parameter. Default ``0``
    delta : float in range ``[0,1]``, optional
        The switch probability parameter. Default ``0``
    mu : float
        The mean expected payoff
    sigma : float
        The standard deviation of the expected payoff
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
        in to a decision. Default is model.decision.binary.decSingle

    Notes
    -----
    Inputs : The participant recieves for time :math:`t` and their choice of
    action :math:`A_{t}` the payoff :math:`Y_{t}(A_{t})`

    Outputs : Their action :math:`A_{t}`. However, internal to the model and
    potentially extractable is the probability that each action is the
    "correct" choice.

    Initialisation : An arbitrary choice is made of which initial action
    :math:`A_{t=0}` to go with. The probability that a hidden state
    :math:`X_{t}` is the one given by :math:`P(X_{t}=A_{t})`
    and is initially uniform and normalised.

    Updates : The prior probabilities of the different actions being correct
    :math:`\\vec{P}_\\mathrm{Prior}(\\vec{X}_{t})` is calculated with respect to
    the probability of reversal :math:`\\delta` and the posterior probability
    :math:`\\vec{P}_\\mathrm{Post}(\\vec{X}_{t-1})`. If the participant has not
    switched then

    math::
        \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t}\\right) = \\left(\\stackrel{1-\\delta}{\\delta} \\stackrel {\\delta}{1-\\delta} \\right) \\bullet \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t-1}\\right)

    If the participant has switched, then the probabilities become

    math::
        \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t}\\right) = \\left(\\stackrel{\\delta}{1-\\delta} \\stackrel {1-\\delta}{\\delta} \\right)  \\bullet \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t-1}\\right)

    The posterior probabilities
    :math:`\\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t}\\right)` is defined for
    the action :math:`A_t` providing the payoff :math:`Y_{t}` as

    math::
        P_\\mathrm{Post}\\left(X_{t} = A_{t}\\right) = \\frac{Y_{t}\\cdot P_\\mathrm{Prior}\\left(X_{t} = A_{t}\\right) }{\\sum_{i \\in \\vec{X_{t}}} P\\left(Y_{t} | X_{t}=i\\right)\\cdot P_\\mathrm{Prior}\\left(X_{t} = i\\right) }

    and for all other actions as

    math::
        \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t} \\neq A_{t}\\right) = \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t} \\neq A_{t}\\right)

    The :math:`P\\left(Y_{t} | X_{t}=i\\right)` is the probability at time
    :math:`t` that the payoff :math:`\\vec{Y_{t}}` would be returned when the
    chosen action was the correct or incorrect. It is calculated from a normal
    distribution where the probability distribution of payoffs are

    math::
        \\vec{\\mu} = \\left(\\stackrel{\\mu_\\mathrm{Correct}}{\\mu_{Incorrect}}\\right) = \\left(\\stackrel{10}{-5}\\right)

    and a variance of :math:`\\sigma = 1`

    The probability that the subject switches their choice of action is

    math::
        P_\\mathrm{Switch}\\left(X_{t},A_{t}\\right) = \\frac{1}{1 - \\exp^{ \\beta \\left(\\mathbf{1}-P_\\mathrm{Post}\\left(X_{t}\\left(A_{t}\\right) = \\mathrm{Incorrect}\\right) - \\alpha\\right)}}

    where :math:`\\alpha` is the indecision point (when it is equiprobable to
    make either choice), and :math:`\\beta` is the degree of stochasticity in
    making the choice (i.e., the exploration/exploitation parameter). Both are
    defined in the simulation.
    """

    Name = "BHMM"

    def __init__(self, **kwargs):

        kwargRemains = self.genStandardParameters(kwargs)

        invBeta = kwargRemains.pop('invBeta', 0.2)
        self.beta = kwargRemains.pop('beta', (1 / invBeta) - 1)
        self.eta = kwargs.pop('eta', 0)
        delta = kwargs.pop('delta', 0)
        self.mu = kwargs.pop('mu', 3)
        self.sigma = kwargs.pop('sigma', 1)

        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.rewFunc = kwargRemains.pop('rewFunc', blankRew())
        self.decisionFunc = kwargs.pop('decFunc', decSingle(expResponses=tuple(range(1, self.numCritics + 1))))

        self.genStandardParameterDetails()
        self.parameters["beta"] = self.beta
        self.parameters["eta"] = self.eta
        self.parameters["delta"] = delta
        self.parameters["mu"] = self.mu
        self.parameters["sigma"] = self.sigma

        # This way for the first run you always consider that you are switching
        self.previousAction = None
        #        if len(prior) != self.numCritics:
        #            raise warning.
        self.posteriorProb = ones(self.numActions) / self.numActions
        self.switchProb = 0
        self.stayMatrix = array([[1 - delta, delta], [delta, 1 - delta]])
        self.switchMatrix = array([[delta, 1 - delta], [1 - delta, delta]])
        self.actionLoc = {k: k for k in range(0, self.numActions)}

        # Recorded information
        self.genStandardResultsStore()
        self.recSwitchProb = []
        self.recPosteriorProb = []
        self.recActionLoc = []

    def returnTaskState(self):
        """ Returns all the relevant data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.standardResultOutput()
        results["SwitchProb"] = array(self.recSwitchProb)
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["ActionLocation"] = array(self.recActionLoc)

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.storeStandardResults()
        self.recSwitchProb.append(self.switchProb)
        self.recActionLoc.append(self.actionLoc.values())
        self.recPosteriorProb.append(self.posteriorProb.copy())

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

        # # If there are multiple possible stimuli, filter by active stimuli and calculate
        # # calculate the expectations associated with each action.
        # if self.numCues > 1:
        #     actionExpectations = self.actStimMerge(self.posteriorProb, stimuli)
        # else:
        #     actionExpectations = self.posteriorProb

        actionExpectations = self.posteriorProb

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

        delta = modReward * expectation

        return delta

    def updateModel(self, event):

        currAction = self.currAction

        postProb = self._postProb(delta, currAction)
        self.posteriorProb = postProb

        # Calculate the new probabilities
        priorProb = self._prob(postProb, currAction)
        self.probabilities = priorProb

        self.switchProb = self._switch(priorProb)

    def _postProb(self, event, postProb, action):

        loc = self.actionLoc

        li = array([postProb[action], postProb[1 - action]])
        payoffs = self._payoff()

        brute = payoffs * li
        newProb = li
        newProb[0] = (event * li[0]) / sum(brute)

        loc[action] = 0
        loc[1 - action] = 1
        self.actionLoc = loc

        return newProb

    def _prob(self, postProb, action):
        """Return the new prior probability that each state is the correct one
        """

        # The probability of the current state being correct, given if the previous state was correct.
        if self.previousAction == action:
            # When the subject has stayed
            pr = self.stayMatrix.dot(postProb)
        else:
            # When the subject has switched
            pr = self.switchMatrix.dot(postProb)

        self.previousAction = action

        return pr

    def _switch(self, prob):
        """Calculate the probability that the participant switches choice

        Parameters
        ----------
        prob : array of floats
            The probabilities for the two options
        """

        pI = prob[1]
        ps = 1.0 / (1.0 - exp(-self.beta * (pI - self.eta)))

        return ps

    def _payoff(self):
        """Payoff given for a specific action and system state

        Returns
        Y : Payoff

        """
        pay = normal(self.mu, self.sigma, (self.numCritics))

        return pay


def blankStim():
    """
    Default stimulus processor. Does nothing.Returns [1,0]

    Returns
    -------
    blankStimFunc : function
        The function expects to be passed the event and then return [1,0].
    currAction : int
        The current action chosen by the model. Used to pass participant action
        to model when fitting

    Attributes
    ----------
    Name : string
        The identifier of the function

    """

    def blankStimFunc(event):
        return [1, 0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc


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