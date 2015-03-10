# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Value and prediction error in medial frontal cortex: integrating the single-unit and systems levels of analysis.
                `Silvetti, M., Seurinck, R., & Verguts, T. (2011)`. 
                Frontiers in Human Neuroscience, 5(August), 75. 
                doi:10.3389/fnhum.2011.00075
"""
from __future__ import division

import logging

from numpy import exp, zeros, array, amax, dot, argmax, mean, square
from random import choice
from collections import defaultdict

from model import model
from modelPlot import modelPlot
from modelSetPlot import modelSetPlot
from utils import listMerge, mergeDatasets
from plotting import lineplot

class RVPM(model):

    """The reward value and prediction model
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    
    Parameters
    ----------
    alpha : float, optional
        Learning rate parameter for updating the cue-value weights. Default 0.3
    gamma : float, optional
        It is a time constant that controls how quickly the neural units 
        (modeled as dynamical systems) respond to external inputs. Default 0.1
    theta : float, optional
        Used for gamma from the paper. Theta is used in other models for the 
        same purpose, so is added here as another option. Gamma will override
        theta.
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
    stimFunc : function, optional
        The function that transforms the stimulus into a form the model can 
        understand and a string to identify it later. Default is blankStim
    decFunc : function, optional
        The function that takes the internal values of the model and turns them
        in to a decision. Default is basicDecision
    """

    Name = "RVPM"

    def __init__(self,**kwargs):

        self.alpha = kwargs.pop('alpha',0.005)
        defaultGamma = 0.1
        self.gamma = kwargs.pop('gamma',defaultGamma)
        if self.gamma == defaultGamma:
            self.gamma = kwargs.pop('theta',defaultGamma)
        self.w = kwargs.pop('w',array([0.01,0.01]))
        self.zeta = kwargs.pop('zeta',2)
        self.tau = kwargs.pop('tau',160)
        self.z = kwargs.pop('z',100)
        self.averaging = kwargs.pop('averaging',3)
        
        self.stimFunc = kwargs.pop('stimFunc',blankStim())
        self.decisionFunc = kwargs.pop('decFunc',basicDecision())
        
        self.T = 0 # Timing sgnal value
        self.c = 0 # The stimuli
        self.r = 0 # Reward value
        self.V = 0 #Reward prediction unit
        self.deltaP = 0 # positivie prediction error unit
        self.deltaM = 0 # negative prediction error unit
        self.TSN = 0 # Temporaly shifted neuron

        self.parameters = {"Name": self.Name,
                           "theta": self.gamma,
                           "alpha": self.alpha,
                           "wInit": self.w,
                           "zeta" : self.zeta,
                           "tau" : self.tau,
                           "z" : self.z,
                           "averaging" : self.averaging,
                           "stimFunc" : self.stimFunc.Name,
                           "decFunc" : self.decisionFunc.Name}

        self.currAction = None
        self.lastObs = False

        # Recorded information

        self._storeSetup()

    def action(self):
        """
        Returns
        -------
        action : integer or None
        """
        self.currAction = self.decisionFunc(self.TSN)

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

        av = self.averaging

        results = self.parameters
        
        for k,v in self.generalStore.iteritems():
            
            results[k + '_early'] = v[:av]
            results[k + '_late'] = v[-av:]

        return results

    def _update(self,event,instance):
        """Processes updates to new actions"""

        if instance == 'obs':

            #Calculate the new information
            self._processEvent(event)

            self.lastObs = True

        elif instance == 'reac':

            if self.lastObs:

                self.lastObs = False

            else:
                
                #Calculate the new information
                self._processEvent(event)

    def storeState(self):
        """ 
        Stores the state of all the important variables so that they can be
        accessed later 
        """

        self.recAction.append(self.currAction)
        self._updateGeneralStore()

    def _processEvent(self,event):
        
        for t,c,r in self.stimFunc(event):
            
            self.c = c
            self.r = r
            self._processStim(t,c,r)
            self._updateEventStore()
            
    def _processStim(self,t,c,r):
        
        T = self._timeSigMag(t)

        dV = self._vUpdate(self.w,self.V,c)
        self.V = self.V + dV
        ddeltaP = self._deltaPUpdate(self.V,self.deltaP,r)
        self.deltaP = self.deltaP + ddeltaP

        ddeltaM = self._deltaMUpdate(self.V,self.deltaM,T,r)
        self.deltaM = self.deltaM + ddeltaM

        self.TSN = self._tsnUpdate(dV,ddeltaP,ddeltaM)
        
        self.w = self._wNew(self.w,self.V,self.deltaP,self.deltaM,c)
            
    def _wNew(self,w,V,deltaP,deltaM,c):
        new = w + self.alpha*c*V*(deltaP - deltaM)
        return new
    
    def _vUpdate(self,w,V,c):
        gamma = self.gamma
        new = -gamma*V + gamma*amax([0,dot(w,c)])
        return new
        
    def _deltaPUpdate(self,V,deltaP,r):
        gamma = self.gamma
        new = -gamma*deltaP + gamma*amax([0,r-self.zeta*V])
        return new
        
    def _timeSigMag(self,t):
        signal = exp((-(t-self.tau)**2)/square(self.z))
        return signal
        
    def _deltaMUpdate(self,V,deltaM,T,r):
        gamma = self.gamma
        new = -gamma*deltaM + gamma*T*amax([0,self.zeta*V-r])
        return new
        
    def _tsnUpdate(self,dV,ddeltaP,ddeltaM):
        signal = amax([0,self.zeta*dV]) + amax([0,ddeltaP]) - amax([0,ddeltaM])
        return signal
        
    def _storeSetup(self):
        
        self.eventStore = {}
        self.eventStore["T"] = []
        self.eventStore["V"] = []
        self.eventStore["DP"] = []
        self.eventStore["DM"] = []
        self.eventStore["TSN"] = []
        self.eventStore["stim"] = []
        self.eventStore["rew"] = []
        self.eventStore["w"] = []
        
        self._generalStoreSetup()
        
    def _generalStoreSetup(self):
        
        self.recAction = []
        self.generalStore = {}
        
        for k in self.eventStore.iterkeys():
            self.generalStore[k] = []
        
    def _updateEventStore(self):
        
        self.eventStore["T"].append(self.T)
        self.eventStore["V"].append(self.V)
        self.eventStore["DP"].append(self.deltaP)
        self.eventStore["DM"].append(self.deltaM)
        self.eventStore["TSN"].append(self.TSN)
        self.eventStore["stim"].append(self.c)
        self.eventStore["rew"].append(self.r)
        self.eventStore["w"].append(self.w)
        
        
    def _updateGeneralStore(self):
        
        for k,v in self.eventStore.iteritems():
            self.generalStore[k].append(array(v))
            
        for k in self.eventStore.iterkeys():
            self.eventStore[k] = []
            
    class modelSetPlot(modelSetPlot):
        
        """Class for the creation of plots relevant to the model set"""

        def _figSets(self):
            """ Contains all the figures """

            self.figSets = []

            # Create all the plots and place them in in a list to be iterated

            self._processData()
            fig = self.avResponse("V")
            self.figSets.append(('VResponse',fig))
            fig = self.avResponse("DP")
            self.figSets.append(('dPResponse',fig))
            fig = self.avResponse("DM")
            self.figSets.append(('dMResponse',fig))
            fig = self.avResponse("TSN")
            self.figSets.append(('TSNResponse',fig))
            fig = self.avResponse("w")
            self.figSets.append(('wResponse',fig))
            
        def _processData(self):
            
            self.modelData = mergeDatasets(self.modelStore, extend = True)
            
            averagedData = defaultdict(dict)

            for t in ["_early","_late"]:
                cmax = [c.argmax() for c in self.modelData["stim"+t]]
                
                for key in ["V","DP","DM","TSN"]:#,"w"]:
                    averagedData[key][key+t] = self._dataAverage(self.modelData[key+t], cmax)
                    
            self.modelAverages = averagedData
            
        def _dataAverage(self, data, cmax):
            
            cLists = [[] for i in xrange(max(cmax)+1)]
            
            for i, cm in enumerate(cmax):
                cLists[cm].append(data[i])
                
            meanVals = [mean(i, axis=0) for i in cLists]
            
            return meanVals
            
        def avResponse(self,key):
            """
            The averaged response from different parts of the model in the form
            of figure 1 from the paper
            """
            data = self.modelAverages[key]
            
            dataKeys = data.keys()
            dataStim = [i for i in xrange(len(data[dataKeys[0]]))]
            
            Y = array(range(len(data[dataKeys[0]][0])))
            
            labels = ["c=" + j + " " + i for i, j in listMerge(dataKeys,dataStim)]
            
            plotData = [data[i][int(j)] for i, j in listMerge(dataKeys,dataStim)]
            
            axisLabels = {"title":"Untitled"}
            axisLabels["xLabel"] = r"$t/10 \textrm{ms}$"
            axisLabels["yLabel"] = key
#            axisLabels["yMax"] = 0
#            axisLabels["yMin"] = -0.5

            fig = lineplot(Y,plotData,labels,axisLabels)

            return fig

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
    
    def blankStimFunc(event):
        
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
            


