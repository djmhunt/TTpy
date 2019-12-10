# -*- coding: utf-8 -*-
"""
:Author: Dominic
"""
from __future__ import division, print_function, unicode_literals, absolute_import


class Experiment(object):
    """The abstract experiment class from which all others inherit

    Many general methods for experiments are found only here

    Parameters
    ----------


    Attributes
    ----------
    Name : string
        The name of the class used when recording what has been used.

    """

    def __init__(self, **kwargs):

        self.kwargs = kwargs.copy()

        self.Name = self.findName()

        self.parameters = {"Name": self.Name
                           }

        self.recAction = []

    def __iter__(self):
        """
        Returns the iterator for the experiment
        """

        return self

    def next(self):
        """
        Produces the next stimulus for the iterator

        Returns
        -------
        stimulus : None
        nextValidActions : Tuple of ints
            The list of valid actions that the model can respond with. Set to
            ``None``, as they never vary.

        Raises
        ------
        StopIteration
        """

        # Since there is nothing to iterate over, just return the final state

        raise StopIteration

    def __eq__(self, other):

        if self.Name == other.Name:
            return True
        else:
            return False

    def __ne__(self, other):

        if self.Name != other.Name:
            return True
        else:
            return False

    def __hash__(self):

        return hash(self.Name)

    def findName(self):
        """
        Returns the name of the class
        """

        return self.__class__.__name__

    def receiveAction(self,action):
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model
        """

        self.recAction.append(action)

    def proceed(self):
        """
        Updates the experiment before the next trialstep
        """

        pass

    def feedback(self):
        """
        Responds to the action from the participant

        Returns
        -------
        feedback : None, int or float

        """
        return None

    def returnTaskState(self):
        """
        Returns all the relevant data for this experiment run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standardResultOutput()

        results["Actions"] = self.recAction

        return results

    def storeState(self):
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        pass

    def standardResultOutput(self):

        results = self.parameters.copy()

        return results

    def params(self):
        """
        Returns the parameters of the experiment as a dictionary

        Returns
        -------
        parameters : dict
            The parameters of the experiment
        """

        return self.parameters.copy()
