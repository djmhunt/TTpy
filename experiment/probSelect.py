# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Opponent actor learning (OpAL): Modeling
                interactive effects of striatal dopamine on reinforcement
                learning and choice incentive.
                Collins, A. G. E., & Frank, M. J. (2014).
                Psychological Review, 121(3), 337–66.
                doi:10.1037/a0037015

"""
from __future__ import division

import pandas as pd

from numpy import array, zeros, exp, ones
from numpy.random import rand
from experiment.experimentTemplate import experiment
from plotting import pandasPlot
from experiment.experimentPlot import experimentPlot
from experiment.experimentSetPlot import experimentSetPlot
#from utils import varyingParams


class probSelect(experiment):
    """
    Probabalistic selection task based on Genetic triple dissociation reveals multiple roles for dopamine in reinforcement learning.
                                        Frank, M. J., Moustafa, A. a, Haughey, H. M., Curran, T., & Hutchison, K. E. (2007).
                                        Proceedings of the National Academy of Sciences of the United States of America, 104(41), 16311–16316.
                                        doi:10.1073/pnas.0706111104

    Many methods are inherited from the experiment.experiment.experiment class.
    Refer to its documentation for missing methods.

    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    Parameters
    ----------
    rewardProb : float in range [0,1], optional
        The probability that a reward is given for choosing action A. Default
        is 0.7
    learningLen : int, optional
        The number of trials in the learning phase. As there is no feeback in
        the trasfer phase there is no trasfer phase. Default is 100

    plotArgs : dictionary, optional
        Any arguments that will be later used by ``experimentPlot``. Refer to
        its documentation for more details.


    Notes
    -----
    The experiment is broken up into two sections: a learning phase and a
    trasfer phase. Participants choose between pairs of four actions: A, B, M1
    and M2. Each provides a reward with a different probability: A:P>0.5,
    B:1-P<0.5, M1=M2=0.5. In the learning phase there is only A and B, but the
    transfer phase has all the action pairs but no feedback. This class only
    covers the learning phase, but models are expected to be implemented as if
    there is a transfer phase.

    """

    Name = "probSelect"

    def reset(self):
        """
        Creates a new experiment instance

        Returns
        -------
        self : The cleaned up object instance
        """

        kwargs = self.kwargs.copy()

        rewardProb = kwargs.pop('rewardProb',0.7)
        learningLen = kwargs.pop("learningLen", 100)

        self.plotArgs = kwargs.pop('plotArgs',{})

        self.parameters = {"Name": self.Name,
                           "rewardProb": rewardProb,
                           "learningLen": learningLen}

        # Set draw count
        self.t = -1
        self.rewardProb = rewardProb
        self.T = learningLen
        self.action = None
        self.stimVal = -1

        # Recording variables

        self.recStimVal = ones(learningLen)*-1
        self.recAction = ones(learningLen)*-1

        return self

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints
            The list of valid actions that the model can respond with. Set to
            [0,1], as the experiment has four actions, but only two in the
            learning phase.

        Raises
        ------
        StopIteration
        """

        self.t += 1

        if self.t == self.T:
            raise StopIteration

        nextStim = None
        nextValidActions = [0,1]

        return nextStim, nextValidActions

    def receiveAction(self,action):
        """
        Receives the next action from the participant
        """

        self.action = action

    def feedback(self):
        """
        Responds to the action from the participant
        """

        action = self.action
        rewProb = self.rewardProb

        #The probabilitiy of sucsess varies depending on if it is choice A or B
        actRewProb = (1-action)*rewProb + action*(1-rewProb)

        if actRewProb >= rand(1):
            reward = 1
        else:
            reward = 0

        self.stimVal = reward

        self.storeState()

        return reward

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

        results = self.parameters

        results["stimVals"] = array(self.recStimVal)
        results["actions"] = array(self.recAction)


        return results

    def storeState(self):
        """ Stores the state of all the important variables so that they can be
        output later """

        self.recAction[self.t] = self.action
        self.recStimVal[self.t] = self.stimVal

    class experimentPlot(experimentPlot):
        """
        Desired plots:
            :math:`\\alpha_N = 0.1, \\alpha_G \in ]0,0.2[`
            :math:`\\beta_N = 1, \\beta_G in ]0,2[`

            Plot the range of
            :math:`\\alpha_G = 0.2 - \\alpha_N` for :math:`\\alpha_N \in ]0,0.2[` and
            :math:`\\beta_G = 2 - \\beta_N` for :math:`\\beta_N in ]0,2[` with the Y-axis being
            Choose(A) = prob(A) - prob(M),
            Avoid(B) = prob(M) - prob(B),
            Bias = choose(A) - avoid(B),
        """

        def _figSets(self):

            # Create all the plots and place them in in a list to be iterated

            self.figSets = []

            self.processData()

            fig = self.biasAlpha()
            self.figSets.append(("biasAlpha",fig))

            fig = self.biasBeta()
            self.figSets.append(("biasBeta",fig))

            fig = self.chooseAlpha()
            self.figSets.append(("chooseAlpha",fig))

            fig = self.chooseBeta()
            self.figSets.append(("chooseBeta",fig))

            fig = self.avoidAlpha()
            self.figSets.append(("avoidAlpha",fig))

            fig = self.avoidBeta()
            self.figSets.append(("avoidBeta",fig))

        def processData(self):
            expStore = self.expStore
            modelStore = self.modelStore
            plotArgs = self.plotArgs

            probFinal = array([d['Probabilities'][-1] for d in modelStore])
            data = pd.DataFrame({'ProbA': probFinal[:,0],
                                 'ProbB': probFinal[:,1],
                                 'ProbM1': probFinal[:,2],
                                 'ProbM2': probFinal[:,3],
                                 'alphaGo': array([d['alphaGo'] for d in modelStore]),
                                 'beta': array([d['beta'] for d in modelStore]),
                                 'alphaNogo': array([d['alphaNogo'] for d in modelStore]),
                                 'betaDiff': array([d['betaDiff'] for d in modelStore])})

            data['chooseA'] = data['ProbA'] - data['ProbM1']
            data['avoidB'] = data['ProbM1'] - data['ProbB']
            data['bias'] = data['chooseA'] - data['avoidB']
            data['betaGo'] = data['beta'] + data['betaDiff']
            data['betaNogo'] = data['beta'] - data['betaDiff']

            self.df = data

        def _plotSubset(self, x, y, z, sort):

            data = self.df.sort(columns=sort)

            selectData = pd.DataFrame(dict(
                                (z + ' = ' + repr(v), data[data[z] == v].groupby(x).mean()[y].values)
                                for v in data[z].unique()))

            alphaGoSet = data[x].unique()
            alphaGoSet.sort()
            selectData[x] = alphaGoSet
            plotData = selectData.set_index(x)

            fig = pandasPlot(plotData, axisLabels = {'xlabel':x, 'ylabel':y, 'title':y})

            return fig


        def biasAlpha(self):

            fig = self._plotSubset('alphaGo', 'bias', 'betaGo', ['betaGo','alphaGo'])

            return fig

        def biasBeta(self):

            fig = self._plotSubset('betaGo', 'bias', 'alphaGo', ['alphaGo', 'betaGo'])

            return fig

        def chooseAlpha(self):

            fig = self._plotSubset('alphaGo', 'chooseA', 'betaGo', ['betaGo','alphaGo'])

            return fig

        def chooseBeta(self):

            fig = self._plotSubset('betaGo', 'chooseA', 'alphaGo', ['alphaGo', 'betaGo'])

            return fig

        def avoidAlpha(self):

            fig = self._plotSubset('alphaGo', 'avoidB', 'betaGo', ['betaGo','alphaGo'])

            return fig

        def avoidBeta(self):

            fig = self._plotSubset('betaGo', 'avoidB', 'alphaGo', ['alphaGo', 'betaGo'])

            return fig

    class experimentSetPlot(experimentSetPlot):
        """
        Desired plots:
            :math:`\\alpha_N = 0.1, \\alpha_G \in ]0,0.2[`
            :math:`\\beta_N = 1, \\beta_G in ]0,2[`

            Plot Positive vs negative choice bias against :math:`prob(R|A) \in ]0.5,1[`
            with:
                :math:`\\alpha_G=\\alpha_N`, varying :math:`\\beta_G` relative to :math:`\\beta_N`
                :math:`\\beta_G=\\beta_N`, varying :math:`\\alpha_G` relative to :math:`\\alpha_N`
        """

        def _figSets(self):

            # Create all the plots and place them in in a list to be iterated

            self.figSets = []

            self.processData()

            fig = self.biasVrewardAlpha()
            self.figSets.append(('biasVrewardAlpha',fig))

            fig = self.biasVrewardBeta()
            self.figSets.append(('biasVrewardBeta',fig))

        def processData(self):
            expStore = self.expStore
            modelStore = self.modelStore
            plotArgs = self.plotArgs

            probFinal = array([d['Probabilities'][-1] for d in modelStore])
            data = pd.DataFrame({'ProbA': probFinal[:,0],
                                 'ProbB': probFinal[:,1],
                                 'ProbM1': probFinal[:,2],
                                 'ProbM2': probFinal[:,3],
                                 'alphaGo': array([d['alphaGo'] for d in modelStore]),
                                 'beta': array([d['beta'] for d in modelStore]),
                                 'alphaNogo': array([d['alphaNogo'] for d in modelStore]),
                                 'betaDiff': array([d['betaDiff'] for d in modelStore]),
                                 'rewardProb': array([d['rewardProb'] for d in expStore])})

            data['chooseA'] = data['ProbA'] - data['ProbM1']
            data['avoidB'] = data['ProbM1'] - data['ProbB']
            data['bias'] = data['chooseA'] - data['avoidB']
            data['betaGo'] = data['beta'] + data['betaDiff']
            data['betaNogo'] = data['beta'] - data['betaDiff']

            self.df = data

        def _plotSubset(self, x, y, z, sort):

            data = self.df.sort(columns=sort)

            selectData = pd.DataFrame(dict(
                                (z + ' = ' + repr(v), data[data[z] == v].groupby(x).mean()[y].values)
                                for v in data[z].unique()))

            alphaGoSet = data[x].unique()
            alphaGoSet.sort()
            selectData[x] = alphaGoSet
            plotData = selectData.set_index(x)

            fig = pandasPlot(plotData, axisLabels = {'xlabel':x, 'ylabel':y, 'title':y})

            return fig

        def biasVrewardAlpha(self):

            fig = self._plotSubset('rewardProb', 'bias', 'alphaGo', ['betaGo', 'alphaGo', 'rewardProb'])

            return fig

        def biasVrewardBeta(self):

            fig = self._plotSubset('rewardProb', 'bias', 'betaGo', ['alphaGo', 'betaGo', 'rewardProb'])

            return fig

def probSelectStimDirect():
    """
    Processes the selection stimuli for models expecting just the event

    Returns
    -------
    deckStim : function
        The function expects to be passed a tuple containing the event and the
        last action. The event is an int and the action is {0,1}. The
        function returns a list of length 2.

    Attributes
    ----------
    Name : string
        The identifier of the function

    See Also
    --------
    model.OpAL

    Examples
    --------
    >>> from experiment.probSelect import probSelectStimDirect
    >>> stim = probSelectStimDirect()
    >>> stim(1,0)
    1
    >>> stim(0,0)
    0
    """


    def probSelectStim(event, action):
        return event

    probSelectStim.Name = "probSelectStimDirect"
    probSelectStim.Params = {}
    return probSelectStim