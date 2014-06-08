# -*- coding: utf-8 -*-
"""
@author: Dominic
"""


from numpy import array

class model(object):

    """The documentation for the class"""

    def __init__(self,**kwargs):
        """The model class is a general template for a model"""

        self.Name = "Empty"

        self.currAction = 1

        # Recorded information

        self.recAction = []
        self.recEvents = []

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


    def action(self):
        """ Returns the action of the model"""

        self._storeState()

        return self.currAction

    def observe(self,event):
        """ Recieves the latest observation"""

        self.recEvents.append(event)

    def feedback(self,response):
        """ Recieves the reaction to the action """

    def outputEvolution(self):
        """ Plots and saves files containing all the relavent data for this model """

        results = {"Name": self.Name,
                   "Actions":array(self.recAction),
                   "Events":array(self.recEvents)}

        return results

    def _storeState(self):
        """ Stores the state of all the important variables so that they can be
            output later """

        self.recAction.append(self.currAction)

    def params(self):
        """ Returns the parameters of the model as a dictionary"""

    def plot(self):
        """Returns a plotting class relavent for this model"""





