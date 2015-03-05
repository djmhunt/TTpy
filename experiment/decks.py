# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Regulatory fit effects in a choice task
                `Worthy, D. a, Maddox, W. T., & Markman, A. B. (2007)`.
                Psychonomic Bulletin & Review, 14(6), 1125–32. 
                Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/18229485
"""
from __future__ import division

from numpy import array, zeros
from numpy.random import rand
from experiment import experiment
#from plotting import dataVsEvents, varDynamics
from experimentPlot import experimentPlot
#from utils import varyingParams

###decks
deckSets = {"WorthyMaddox": array([[ 2,  2,  1,  1,  2,  1,  1,  3,  2,  6,  2,  8,  1,  6,  2,  1,  1,
                                    5,  8,  5, 10, 10,  8,  3, 10,  7, 10,  8,  3,  4,  9, 10,  3,  6,
                                    3,  5, 10, 10, 10,  7,  3,  8,  5,  8,  6,  9,  4,  4,  4, 10,  6,
                                    4, 10,  3, 10,  5, 10,  3, 10, 10,  5,  4,  6, 10,  7,  7, 10, 10,
                                    10,  3,  1,  4,  1,  3,  1,  7,  1,  3,  1,  8],
                                    [ 7, 10,  5, 10,  6,  6, 10, 10, 10,  8,  4,  8, 10,  4,  9, 10,  8,
                                     6, 10, 10, 10,  4,  7, 10,  5, 10,  4, 10, 10,  9,  2,  9,  8, 10,
                                     7,  7,  1, 10,  2,  6,  4,  7,  2,  1,  1,  1,  7, 10,  1,  4,  2,
                                     1,  1,  1,  4,  1,  4,  1,  1,  1,  1,  3,  1,  4,  1,  1,  1,  5,
                                     1,  1,  1,  7,  2,  1,  2,  1,  4,  1,  4,  1]])}
defaultDecks = deckSets["WorthyMaddox"]

class decks(experiment):
    """
    Based on the Worthy&Maddox 2007 paper "Regulatory fit effects in a choice task.
    
    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.
    
    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.
    
    Parameters
    ----------
    draws: int, optional
        Number of cards drawn by the participant
    decks: array of floats, optional
        The decks of cards
    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to 
        its documentation for more details.        
    """

    Name = "decks"

    def reset(self):
        """ 
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        T = kwargs.pop('draws',None)
        decks = kwargs.pop("decks",defaultDecks)

        self.plotArgs = kwargs.pop('plotArgs',{})

        if isinstance(decks, str):
            if decks in deckSets:
                self.decks = deckSets[decks]
            else:
                raise "Unknown deck sets"
        else:
            self.decks = decks

        if T:
            self.T = T
        else:
            self.T = len(self.decks[0])

        self.parameters = {"Name": self.Name,
                           "Draws": self.T,
                           "Decks": self.decks}

        # Set draw count
        self.t = -1
        self.drawn = -1#[-1,-1]
        self.cardValue = None
        self.action = None

        # Recording variables

        self.recCardVal = [-1]*self.T
        self.recAction = [-1]*self.T

        return self

    def next(self):
        """
        Produces the next bead for the iterator
        
        Returns
        -------
        stimulus : None
        
        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        return None

    def receiveAction(self,action):
        """
        Receives the next action from the participant
        """

        self.action = action

    def feedback(self):
        """
        Responds to the action from the participant
        """

        deckDrawn = self.action
        cardDrawn = self.drawn + 1 #[deckDrawn] + 1

        self.cardValue = self.decks[deckDrawn,cardDrawn]

#        self.drawn[deckDrawn] = cardDrawn
        self.drawn = cardDrawn

        self.storeState()

        return self.cardValue

    def procede(self):
        """
        Updates the experiment after feedback
        """

        pass

    def outputEvolution(self):
        """
        Plots and saves files containing all the relavent data for this
        experiment run
        """

        results = {"Name": self.Name,
                   "Actions": self.recAction,
                   "cardValue": self.recCardVal,
                   "Decks": self.decks,
                   "Draws": self.T,
                   "finalDeckDraws": self.drawn}

        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action
        self.recCardVal[self.t] = self.cardValue

def deckStimDirect():
    
    def deckStim(event, decision):
        stimulus = [event*(1-decision) + event*decision, event*decision + event*(1-decision)]
        return stimulus
    return deckStim

def deckStimDualInfo(maxEventVal):
    
    def deckStim(event, decision):
        stim = (event/maxEventVal)*(1-decision) + (1-(event/maxEventVal))*decision
        stimulus = [stim,1-stim]
        return stimulus
    return deckStim