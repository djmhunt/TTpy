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
from types import NoneType

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decSingle
from utils import callableDetailsString

class BHMM(model):

    """The Beysian Hidden Markov Model model

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    beta : float, optional
        Sensitivity parameter for probabilities. Default ``4``
    eta : float, optional
        Decision threshold parameter. Default ``0``
    delta : float in range ``[0,1]``, optional
        The switch probability parameter. Default ``0``
    mu : float
        The mean expected payoff
    sigma : float
        The standard deviation of the expected payoff
    prior : array of two floats in ``[0,1]`` or just float in range, optional
        The prior probability of of the two states being the correct one.
        Default ``array([0.5,0.5])``
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can
        understand and a string to identify it later. Default is blankStim
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

    .. math::
        \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t}\\right) = \\left(\\stackrel{1-\\delta}{\\delta} \\stackrel {\\delta}{1-\\delta} \\right) \\bullet \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t-1}\\right)

    If the participant has switched, then the probabilities become

    .. math::
        \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t}\\right) = \\left(\\stackrel{\\delta}{1-\\delta} \\stackrel {1-\\delta}{\\delta} \\right)  \\bullet \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t-1}\\right)

    The posterior probabilities
    :math:`\\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t}\\right)` is defined for
    the action :math:`A_t` providing the payoff :math:`Y_{t}` as

    .. math::
        P_\\mathrm{Post}\\left(X_{t} = A_{t}\\right) = \\frac{Y_{t}\\cdot P_\\mathrm{Prior}\\left(X_{t} = A_{t}\\right) }{\\sum_{i \\in \\vec{X_{t}}} P\\left(Y_{t} | X_{t}=i\\right)\\cdot P_\\mathrm{Prior}\\left(X_{t} = i\\right) }

    and for all other actions as

    .. math::
        \\vec{P}_\\mathrm{Post}\\left(\\vec{X}_{t} \\neq A_{t}\\right) = \\vec{P}_\\mathrm{Prior}\\left(\\vec{X}_{t} \\neq A_{t}\\right)

    The :math:`P\\left(Y_{t} | X_{t}=i\\right)` is the probability at time
    :math:`t` that the payoff :math:`\\vec{Y_{t}}` would be returned when the
    chosen action was the correct or incorrect. It is calculated from a normal
    distribution where the probability distribution of payoffs are

    .. math::
        \\vec{\\mu} = \\left(\\stackrel{\\mu_\\mathrm{Correct}}{\\mu_{Incorrect}}\\right) = \\left(\\stackrel{10}{-5}\\right)

    and a variance of :math:`\\sigma = 1`

    The probability that the subject switches their choice of action is

    .. math::
        P_\\mathrm{Switch}\\left(X_{t},A_{t}\\right) = \\frac{1}{1 - \\exp^{ \\beta \\left(\\mathbf{1}-P_\\mathrm{Post}\\left(X_{t}\\left(A_{t}\\right) = \\mathrm{Incorrect}\\right) - \\alpha\\right)}}

    where :math:`\\alpha` is the indecision point (when it is equiprobable to
    make either choice), and :math:`\\beta` is the degree of stochasticity in
    making the choice (i.e., the exploration/exploitation parameter). Both are
    defined in the simulation.
    """

    Name = "BHMM"

    def __init__(self,**kwargs):

#        self.numStimuli = kwargs.pop('numStimuli', 2)
        self.numStimuli = 2
        self.beta = kwargs.pop('beta', 4)
        self.prior = kwargs.pop('prior', ones(self.numStimuli)*0.5)
        self.eta = kwargs.pop('eta', 0)
        delta = kwargs.pop('delta',0)
        self.mu = kwargs.pop('mu',3)
        self.sigma = kwargs.pop('sigma',1)


        self.stimFunc = kwargs.pop('stimFunc', blankStim())
        self.decisionFunc = kwargs.pop('decFunc', decSingle(expResponses = tuple(range(1,self.numStimuli+1))))

        self.parameters = {"Name": self.Name,
                           "beta": self.beta,
                           "eta": self.eta,
                           "delta": delta,
                           "prior": self.prior,
#                           "numStimuli": self.numStimuli,
                           "stimFunc" : callableDetailsString(self.stimFunc),
                           "decFunc" : callableDetailsString(self.decisionFunc)}

        self.currAction = 0
        # This way for the first run you always consider that you are switching
        self.previousAction = None
#        if len(prior) != self.numStimuli:
#            raise warning.
        self.posteriorProb = array(self.prior)
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.decision = None
        self.validActions = None
        self.switchProb = 0
        self.stayMatrix = array([[1-delta,delta],[delta,1-delta]])
        self.switchMatrix = array([[delta,1-delta],[1-delta,delta]])
        self.actionLoc = {k:k for k in range(0,self.numStimuli)}

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recSwitchProb = []
        self.recPosteriorProb = []
        self.recDecision = []
        self.recActionLoc = []

    def action(self):
        """
        Returns
        -------
        action : integer or None
        """
        self.currAction = self.decision

        self.storeState()

        return self.currAction

    def outputEvolution(self):
        """ Returns all the relevent data for this model

        Returns
        -------
        results : dict
            The dictionary contains a series of keys including Name,
            Probabilities, Actions and Events.
        """

        results = self.parameters.copy()

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["SwitchProb"] = array(self.recSwitchProb)
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["ActionLocation"] = array(self.recActionLoc)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            if type(events) is not NoneType:
                self._processEvent(events)
            self._processAction()


        elif instance == 'reac':
            if type(events) is not NoneType:
                self._processEvent(events)

    def _processEvent(self,events):

        currAction = self.currAction

        event = self.stimFunc(events, currAction)

        self.recEvents.append(event)

        postProb = self._postProb(event, self.posteriorProb, currAction)
        self.posteriorProb = postProb

        #Calculate the new probabilities
        priorProb = self._prob(postProb, currAction)
        self.probabilities = priorProb

        self.switchProb = self._switch(priorProb)

    def _processAction(self):

        self.decision, self.decProbs = self.decisionFunc(self.switchProb, self.currAction, validResponses = self.validActions)

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        accessed later
        """

        self.recAction.append(self.currAction)
        self.recProbabilities.append(self.probabilities.copy())
        self.recActionProb.append(self.probabilities[self.actionLoc[self.currAction]])
        self.recSwitchProb.append(self.switchProb)
        self.recActionLoc.append(self.actionLoc.values())
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)

    def _postProb(self, event, postProb, action):

        loc = self.actionLoc

        li = array([postProb[action],postProb[1-action]])
        payoffs = self._payoff()

        brute = payoffs * li
        newProb = li
        newProb[0] = (event*li[0])/sum(brute)

        loc[action] = 0
        loc[1-action] = 1
        self.actionLoc = loc

        return newProb

    def _prob(self, postProb, action):
        """Return the new prior probabilitiy that each state is the correct one
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
        pay = normal(self.mu,self.sigma,(self.numStimuli))

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
        return [1,0]

    blankStimFunc.Name = "blankStim"
    return blankStimFunc
