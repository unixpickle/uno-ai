"""
Measure the lengths of random games.
"""

import random

from uno_ai.game import Game


def main():
    while True:
        g = Game(4)
        num_moves = 0
        while g.winner() is not None:
            action = random.choice(g.options())
            g.act(action)
            num_moves += 1
        print(num_moves)


if __name__ == '__main__':
    main()
