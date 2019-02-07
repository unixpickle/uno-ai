from enum import Enum


CARD_VEC_SIZE = 10 + 6 + 4


def full_deck():
    """
    Create a complete Uno deck.
    """
    deck = []
    for color in [Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE]:
        for _ in range(2):
            for number in range(1, 10):
                deck.append(Card(CardType.NUMERAL, color=color, number=number))
            for card_type in [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO]:
                deck.append(Card(card_type, color=color))

        deck.append(Card(CardType.NUMERAL, color=color, number=0))
        deck.append(Card(CardType.WILD))
        deck.append(Card(CardType.WILD_DRAW))
    return deck


class CardType(Enum):
    """
    The type of a card.
    """
    NUMERAL = 0
    SKIP = 1
    REVERSE = 2
    DRAW_TWO = 3
    WILD = 4
    WILD_DRAW = 5


class Color(Enum):
    """
    The color of a card.
    """
    RED = 0
    ORANGE = 1
    GREEN = 2
    BLUE = 3


class Card:
    """
    A card in the deck.
    """

    def __init__(self, card_type, color=None, number=None):
        self.card_type = card_type
        self.color = color
        self.number = number

    def vector(self):
        """
        Convert the card into a vector.
        """
        vec = [0.0] * CARD_VEC_SIZE
        if self.number is not None:
            vec[self.number] = 1.0
        if self.color is not None:
            vec[10 + self.color.value] = 1.0
        vec[14 + self.card_type.value] = 1.0
        return vec
