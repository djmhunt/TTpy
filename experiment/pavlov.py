# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Value and prediction error in medial frontal cortex: integrating the single-unit and systems levels of analysis.
                `Silvetti, M., Seurinck, R., & Verguts, T. (2011)`. 
                Frontiers in Human Neuroscience, 5(August), 75. 
                doi:10.3389/fnhum.2011.00075
"""

from __future__ import division

from numpy import array, ones, zeros, concatenate
from numpy.random import choice, random

from experiment import experiment
from experimentPlot import experimentPlot

class pavlov(experiment):

    """
    Based on the Silvetti et al 2011 paper `"Value and prediction error in 
    medial frontal cortex: integrating the single-unit and systems levels of 
    analysis."`
    
    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    
    Parameters
    ----------
    rewMag : float, optional
        The size of the stimulus. Default 4
    rewProb : array of floats, optional
        The probabilities of each stimulus producing a reward. 
        Default [0.85,0.33]
    stimMag : float, optional
        The size of the stimulus. Default 1
    stimDur : int, optional
        The duration, in tens of ms, that the stimulus is produced for. This
        should be longer than rewDur since rewDur is set to end when stimDur
        ends. Default 200
    rewDur : int, optional
        The duration, in tens of ms, that the reward is produced for.
        Default 40
    simDur : int, optional
        The duration, in tens of ms, that each stimulus event is run for.
        Default 300
    stimRepeats : int, optional
        The number of times a stimulus is introduced. Default 72
    
    """

    Name = "pavlov"

    def reset(self):
        """ 
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()
        
        self.rewMag = kwargs.pop('rewMag',4)
        self.rewProb = kwargs.pop('rewProb',array([0.87,0.33]))
        self.stimMag = kwargs.pop('stimMag',1)
        self.stimDur = kwargs.pop('stimDur',20)#200) # Stimulus duration
        self.rewDur = kwargs.pop('rewDur',4)#40) #duration of reward
        self.simLen = kwargs.pop('simDur',30)#300) # the length of the simulation
        self.stimRepeats = kwargs.pop('stimRepeats',7)#72) # The number of learning runs
#        simLoop = kwargs.pop('simLoopLen',100) #The number of learning loops are run
        
        self.index = -1

        self.plotArgs = kwargs.pop('plotArgs',{})

        self.parameters = {"Name": self.Name,
                           "rewMag": self.rewMag,
                           "rewProb": self.rewProb,
                           "stimMag": self.stimMag,
                           "stimDur": self.stimDur,
                           "rewDur": self.rewDur,
                           "simLen": self.simLen,
                           "stimRepeats": self.stimRepeats}
        
        self.cSet, self.stimChoice = self._getStim(self.stimRepeats, self.stimMag)
        self.rewSigSet, self.rewVals = self._getRew(self.stimChoice, self.simLen, self.stimRepeats, self.stimDur, self.rewDur, self.rewMag, self.rewProb)
        
        self.recActions = []        
        
        return self

    def next(self):
        """
        Produces the next bead for the iterator
        
        Returns
        -------
        c : list of floats
            Contains the inputs for each of the stimuli
        rewSig : list of lists of floats
            Each list contains the rewards at each time
        stimDur : int
            
        
        Raises
        ------
        StopIteration
        """

        self.index += 1

        if self.index == self.stimRepeats:
            raise StopIteration
            
        c = self.cSet[self.index]
        rewSig = self.rewSigSet[self.index]

        return c, rewSig, self.stimDur

    def receiveAction(self,action):
        """
        Receives the next action from the participant
        
        Parameters
        ----------
        action : ignored
            The action is only stored, not used.
        """

        self.action = action
        
        self.storeState()

    def outputEvolution(self):
        """ 
        Plots and saves files containing all the relavent data for this
        experiment run 
        
        Returns
        -------
        results : dictionary
            Contains the class paramters as well as the other useful data
        """

        results = self.parameters
        
        results["choices"] = self.cSet
        results["stimuli"] = self.stimChoice
        results["rewards"] = self.rewVals
        results["actions"] = self.recActions
        
        return results

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        output later 
        """
        
        self.recActions.append(self.action)

        
    def _getStim(self, stimRepeats, stimMag):
        stimChoice = choice([0,1],size = (stimRepeats,1))
        cSet = array([[1-sc[0],sc[0]] for sc in stimChoice])*stimMag
        
        return cSet, stimChoice
        
    def _getRew(self, stimChoice, simLen, stimRepeats, stimDur, rewDur, rewMag, rewProb):
    
        rewVals = (random((stimRepeats,1)) < rewProb[stimChoice])*rewMag
        rewSig1 = zeros((stimRepeats,stimDur-rewDur))
        rewSig2 = ones((stimRepeats,rewDur))*rewVals
        rewSig3 = zeros((stimRepeats,simLen-stimDur))
        rewSigSet = concatenate((rewSig1,rewSig2,rewSig3),1)
        
        return rewSigSet, rewVals
        
def pavlovStimTemporal():
    """
    Passes the pavlov stimuli to models that cope with stimuli and rewards 
    that have a duration.
        
    Returns
    -------
    pavlovStim : function
        The function expects to be passed an event with three components: 
        ``(stim,rew,stimDur)`` and yield a series of events ``t,c,r```. 
        ``stim`` is the value of the stimulus. It is expected to be a list-like
        object. ``rew`` is a list containing the reward for each timestep. 
        The reward is expected to be a float. ``stimDur`` is the duration of 
        the stimulus, an ``int``. This should be less than the length of ``rew``.
        ``c`` the stimulus. ``r`` the reward. ``t`` is the time
        
    Attributes
    ----------
    Name : string
        The identifier of the function
    
    """
    
    def pavlovStim(event):
        
        cStim = event[0]
        rewSig = event[1]
        stimDur = event[2]
    
        cStimZeros = zeros((len(cStim)))
        
        for t, r in enumerate(rewSig):
            
            if t < stimDur:
                c = cStim
            else:
                c = cStimZeros 
                
            yield t, c, r
            
    pavlovStim.Name = "pavlovStimTemporal"
            
    return pavlovStim
        
        