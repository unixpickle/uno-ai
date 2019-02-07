"""
Action representations for Uno.
"""

from abc import ABC, abstractmethod

ACTION_VECTOR_SIZE = 115


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

    def __eq__(self, other):
        return type(other) == NopAction

    def __str__(self):
        return 'do nothing'


class ChallengeAction(Action):
    def index(self):
        return 1

    def __eq__(self, other):
        return type(other) == ChallengeAction

    def __str__(self):
        return 'challenge'


class DrawAction(Action):
    def index(self):
        return 2

    def __eq__(self, other):
        return type(other) == DrawAction

    def __str__(self):
        return 'draw a card'


class PickColorAction(Action):
    def __init__(self, color):
        self.color = color

    def index(self):
        return self.color.value + 3

    def __eq__(self, other):
        return type(other) == PickColorAction and other.color == self.color

    def __str__(self):
        return 'pick color %s' % self.color


class PlayCardAction(Action):
    def __init__(self, index):
        self.raw_index = index

    def index(self):
        return self.raw_index + 7

    def __eq__(self, other):
        return type(other) == PlayCardAction and other.raw_index == self.raw_index

    def __str__(self):
        return 'play card %d' % self.raw_index
