import random

from .game import Game


def test_game():
    """
    Run through a few games and make sure there's no
    exceptions.
    """
    for i in range(10):
        for n in range(2, 5):
            g = Game(n)
            while not g.winner():
                g.act(random.choice(g.options()))
