# -*- coding: utf-8 -*-
"""
@author: Dominic
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

    """Based on the Worthy&Maddox 2007 paper "Regulatory fit effects in a choice task."""

    Name = "decks"

    def __init__(self,**kwargs):

        self.kwargs = kwargs

        self.reset()

    def reset(self):
        """ Creates a new experiment instance

        T: Number of cards drawn by the participant
        decks: Array containing the decks of cards"""

        kwargs = self.kwargs.copy()

        T = kwargs.pop('Draws',None)
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
        """ Produces the next item for the iterator"""

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        return

    def receiveAction(self,action):
        """ Receives the next action from the participant"""

        self.action = action

    def feedback(self):
        """ Responds to the action from the participant"""

        deckDrawn = self.action
        cardDrawn = self.drawn + 1 #[deckDrawn] + 1

        self.cardValue = self.decks[deckDrawn,cardDrawn]

#        self.drawn[deckDrawn] = cardDrawn
        self.drawn = cardDrawn

        self._storeState()

        return self.cardValue

    def procede(self):
        """Updates the experiment after feedback"""

        pass

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this
        experiment run """

        results = {"Name": self.Name,
                   "Actions": self.recAction,
                   "cardValue": self.recCardVal,
                   "Decks": self.decks,
                   "Draws": self.T,
                   "finalDeckDraws": self.drawn}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action
        self.recCardVal[self.t] = self.cardValue

    class experimentPlot(experimentPlot):

        """Abstract class for the creation of plots relevant to a experiment"""