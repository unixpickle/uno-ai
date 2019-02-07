"""
Action representations for Uno.
"""

from abc import ABC, abstractmethod


class Action(ABC):
    """
    An abstract action in a game of Uno.
    """
    @abstractmethod
    def index(self):
        """
        Get the index of the action in the action vector.
        """
        pass


class NopAction(Action):
    def index(self):
        return 0


class ChallengeAction(Action):
    def index(self):
        return 1


class DrawAction(Action):
    def index(self):
        return 2


class PickColorAction(Action):
    def __init__(self, color):
        self.color = color

    def index(self):
        return self.color.value + 3


class PlayCardAction(Action):
    def __init__(self, index):
        self.raw_index = index

    def index(self):
        return self.raw_index + 7
