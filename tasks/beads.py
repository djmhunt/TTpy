# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt

:Reference: Jumping to conclusions: a network model predicts schizophrenic patients’ performance on a probabilistic reasoning task.
                    `Moore, S. C., & Sellen, J. L. (2006)`.
                    Cognitive, Affective & Behavioral Neuroscience, 6(4), 261–9.
                    Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/17458441
"""
import numpy as np

from typing import Union, Tuple, List, Dict, Any, Optional, NewType

from tasks.taskTemplate import Task
from model.modelTemplate import Stimulus, Rewards

Action = NewType('Action', Union[int, str])

# Bead Sequences:
beadSequences = {"MooreSellen": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}
defaultBeads = beadSequences["MooreSellen"]


class Beads(Task):
    """Based on the Moore & Sellen Beads task

    Many methods are inherited from the tasks.taskTemplate.Task class.
    Refer to its documentation for missing methods.

    Parameters
    ----------
    task_length : int, optional
        Number of beads that could potentially be shown
    bead_sequence : list or array of {0,1}, optional
        The sequence of beads to be shown. Bead sequences can also be embedded
        in the code and then referred to by name. The only current one is
        `MooreSellen`, the default sequence.
    """

    def __init__(self, task_length: int = None, bead_sequence: Optional[Union[str, List]] = None):

        super(Beads, self).__init__()

        if isinstance(bead_sequence, str):
            if bead_sequence in beadSequences:
                self.beads = beadSequences[bead_sequence]
            else:
                raise Exception("Unknown bead sequence")
        elif isinstance(bead_sequence, list):
            self.beads = bead_sequence
        else:
            self.beads = defaultBeads

        if task_length:
            self.task_length = task_length
        else:
            self.task_length = len(self.beads)

        self.parameters["trial_duration"] = self.task_length
        self.parameters["bead_sequence"] = self.beads

        # Set trial_step count
        self.trial_step = -1

        # Recording variables

        self.recBeads = [-1]*self.task_length
        self.recAction = [-1]*self.task_length
        self.firstDecision = 0

    def __next__(self) -> Tuple[int, List[Action]]:
        """ Produces the next bead for the iterator

        Returns
        -------
        bead : {0,1}
        next_valid_actions : Tuple of ints or ``None``
            The list of valid actions that the model can respond with. Set to (0,1), as they never vary.

        Raises
        ------
        StopIteration
        """

        self.trial_step += 1

        if self.trial_step == self.task_length:
            raise StopIteration

        self.store_state()

        next_stim = self.beads[self.trial_step]
        next_valid_actions = [0, 1]

        return next_stim, next_valid_actions

    def receive_action(self, action: Action) -> None:
        """
        Receives the next action from the participant

        Parameters
        ----------
        action : int or string
            The action taken by the model
        """

        self.recAction[self.trial_step] = action

        if action and not self.firstDecision:
            self.firstDecision = self.trial_step + 1

    def return_task_state(self) -> Dict[str, Any]:
        """
        Returns all the relevant data for this task run

        Returns
        -------
        results : dictionary
            A dictionary containing the class parameters  as well as the other useful data
        """

        results = self.standard_result_output()

        results["Observables"] = np.array(self.recBeads)
        results["Actions"] = self.recAction
        results["FirstDecision"] = self.firstDecision

        return results

    def store_state(self) -> None:
        """
        Stores the state of all the important variables so that they can be
        output later
        """

        self.recBeads[self.trial_step] = self.beads[self.trial_step]


def generate_sequence(num_beads: int, one_prob: float, switch_prob: float) -> np.ndarray:
    """
    Designed to generate a sequence of beads with a probability of switching
    jar at any time.

    Parameters
    ----------
    num_beads : int
        The number of beads in the sequence
    one_prob : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar.
    switch_prob : float in ``[0,1]``
        The probability that the drawn beads change the jar they are being
        drawn from

    Returns
    -------
    sequence : list of ``{0,1}``
        The generated sequence of beads
    """

    sequence = np.zeros(num_beads)

    probabilities = np.random.rand(num_beads, 2)
    bead = 1

    for i in range(num_beads):
        if probabilities[i, 1] < switch_prob:
            bead = 1-bead

        if probabilities[i, 0] < one_prob:
            sequence[i] = bead
        else:
            sequence[i] = 1-bead

    return sequence


class StimulusBeadDirect(Stimulus):
    """
    Processes the beads stimuli for models expecting just the event

    """

    def process_stimulus(self, observation: int) -> Tuple[int, int]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
        stimuliActivity : float or list of float

        """
        return 1, observation


class StimulusBeadDualDirect(Stimulus):
    """
    Processes the beads stimuli for models expecting a tuple of ``[event,1-event]``

    """

    def process_stimulus(self, observation: int) -> Tuple[int, np.ndarray]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
            The elements present of the stimulus
        stimuliActivity : float or list of float
            The activity of each of the elements

        """
        stimulus = np.array([observation, 1-observation])
        return 1, stimulus


class StimulusBeadDualInfo(Stimulus):
    """
    Processes the beads stimuli for models expecting the reward information
    from two possible actions

    Parameters
    ----------
    one_prob : float in ``[0,1]``
        The probability of a 1 from the first jar. This is also the probability
        of a 0 from the second jar. ``event_info`` is calculated as
        ``one_prob*event + (1-one_prob)*(1-event)``
    """
    oneProb = 0.1

    def process_stimulus(self, observation: float) -> Tuple[int, np.ndarray]:
        """
        Processes the decks stimuli for models expecting just the event

        Returns
        -------
        stimuliPresent :  int or list of int
            The elements present of the stimulus
        stimuliActivity : float or list of float
            The activity of each of the elements

        """
        stim = self.oneProb*observation + (1-self.oneProb)*(1-observation)
        stimulus = np.array([stim, 1-stim])
        return 1, stimulus


class RewardBeadDirect(Rewards):
    """
    Processes the beads reward for models expecting just the reward
    """

    def process_feedback(self,
                         feedback: None,
                         last_action: Action,
                         stimuli: List[float]
                         ) -> None:
        """

        Returns
        -------
        modelFeedback:
        """
        return feedback
