# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

"""
from __future__ import division

import logging

from numpy import exp, array, ones

from modelTemplate import model
from model.modelPlot import modelPlot
from model.modelSetPlot import modelSetPlot
from model.decision.binary import decSingle
from utils import callableDetailsString

class BPMS(model):

    """The Beysian Predictor with Markovian Switching model

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
    The Markovian switcher is the same as that used in BHMM and the rest is
    taken from BP. It currently does not allow more than two actions, equally
    you can only have two stimuli
    """

    Name = "BPMS"

    def __init__(self,**kwargs):

#        self.numStimuli = kwargs.pop('numStimuli', 2)
        self.numStimuli = 2
        self.beta = kwargs.pop('beta', 4)
        self.prior = kwargs.pop('prior', ones(self.numStimuli)*0.5)
        self.eta = kwargs.pop('eta', 0)
        delta = kwargs.pop('delta',0)


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

        self.currAction = 1
#        if len(prior) != self.numStimuli:
#            raise warning.
        self.posteriorProb = array(self.prior)
        self.probabilities = array(self.prior)
        self.decProbs = array(self.prior)
        self.decision = None
        self.validActions = None
        self.previousAction = None
        self.stayMatrix = array([[1-delta,delta],[delta,1-delta]])
        self.switchMatrix = array([[delta,1-delta],[1-delta,delta]])

        # Recorded information

        self.recAction = []
        self.recEvents = []
        self.recProbabilities = []
        self.recActionProb = []
        self.recSwitchProb = []
        self.recPosteriorProb = []
        self.recDecision = []

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

        results = self.parameters

        results["Probabilities"] = array(self.recProbabilities)
        results["ActionProb"] = array(self.recActionProb)
        results["SwitchProb"] = array(self.recSwitchProb)
        results["PosteriorProb"] = array(self.recPosteriorProb)
        results["Actions"] = array(self.recAction)
        results["Decsions"] = array(self.recDecision)
        results["Events"] = array(self.recEvents)

        return results

    def _update(self,events,instance):
        """Processes updates to new actions"""

        if instance == 'obs':
            if events != None:
                self._processEvent(events)
            self._processAction()


        elif instance == 'reac':
            if events != None:
                self._processEvent(events)

    def _processEvent(self,events):

        event = self.stimFunc(events, self.currAction)

        self.recEvents.append(event)

        postProb = self._postProb(event, self.posteriorProb, self.currAction)
        self.posteriorProb = postProb

        #Calculate the new probabilities
        priorProb = self._prob(postProb)
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
        self.recActionProb.append(self.decProbs[self.currAction])
        self.recSwitchProb.append(self.switchProb)
        self.recPosteriorProb.append(self.posteriorProb.copy())
        self.recDecision.append(self.decision)

    def _postProb(self, event, postProb, action):

        p = postProb * event

        li = array([p[action],p[1-action]])

        newProb = li/sum(li)

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

        return pr

    def _switch(self, prob):
        """Calculate the probability that the participant switches choice

        Parameters
        ----------
        prob : array of floats
            The probabilities for the two options
        """

        pI = prob[1]
        ps = 1.0 / (1.0 - exp(self.beta * (pI - self.eta)))

        return ps

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
