"""
Core Uno game logic.
"""

from enum import Enum
import random

from .actions import NopAction, ChallengeAction, DrawAction, PickColorAction, PlayCardAction
from .cards import CardType, Color, full_deck

MAX_PLAYERS = 8
OBS_VECTOR_SIZE = 109 * 20 + MAX_PLAYERS * 3


class GameState(Enum):
    PLAY_OR_DRAW = 0
    PLAY = 1
    PLAY_DRAWN = 2
    PICK_COLOR = 3
    PICK_COLOR_INIT = 4
    CHALLENGE_VALID = 5
    CHALLENGE_INVALID = 6


class Game:
    def __init__(self, num_players):
        assert num_players <= MAX_PLAYERS
        self._num_players = num_players
        self._deck = full_deck()
        random.shuffle(self._deck)
        self._discard = []
        self._hands = []
        for _ in range(num_players):
            hand = []
            for _ in range(7):
                hand.append(self._deck.pop())
            self._hands.append(hand)
        while self._deck[-1].card_type == CardType.WILD_DRAW:
            random.shuffle(self._deck)
        self._discard.append(self._deck.pop())

        self._direction = 1
        self._turn = 0
        self._state = GameState.PLAY_OR_DRAW

        self._update_init_state()

    def hands(self):
        return self._hands

    def discard(self):
        return self._discard[-1]

    def winner(self):
        """
        If the game has ended, get the winner.
        """
        for i, hand in enumerate(self._hands):
            if not len(hand):
                return i
        return None

    def turn(self):
        """
        Get the current player.
        """
        return self._turn

    def obs(self, player):
        """
        Generate an observation vector for a player.
        """
        vec = []
        hand = self._hands[player]
        for card in hand:
            vec += card.vector()
        vec += [0.0] * (20 * (108 - len(hand)))
        vec += self._discard[-1].vector()
        vec += _player_vec(player)
        vec += _player_vec(self.turn())
        for hand in self._hands:
            vec += [float(len(hand))]
        vec += [0.0] * (MAX_PLAYERS - self._num_players)
        return vec

    def options(self):
        """
        Get the valid actions for the current player.
        """
        if self._state == GameState.PLAY_OR_DRAW:
            return [NopAction(), DrawAction()] + self._play_options()
        elif self._state == GameState.PLAY:
            return [NopAction()] + self._play_options()
        elif self._state == GameState.PLAY_DRAWN:
            res = [NopAction()]
            if self._can_play(self._current_hand()[-1]):
                res += [PlayCardAction(len(self._current_hand()) - 1)]
            return res
        elif self._state == GameState.PICK_COLOR or self._state == GameState.PICK_COLOR_INIT:
            return [PickColorAction(c) for c in [Color.RED, Color.ORANGE, Color.GREEN, Color.BLUE]]
        elif self._state == GameState.CHALLENGE_VALID or self._state == GameState.CHALLENGE_INVALID:
            return [NopAction(), ChallengeAction()]
        raise RuntimeError('invalid state')

    def act(self, action):
        """
        Take a turn by selecting the action for the
        current player.
        """
        assert action in self.options()
        if self._state == GameState.PLAY_OR_DRAW:
            if isinstance(action, NopAction):
                self._advance_turn()
            elif isinstance(action, DrawAction):
                self._state = GameState.PLAY_DRAWN
                self._current_hand().append(self._draw())
            else:
                self._play_card(action)
        elif self._state == GameState.PLAY:
            if isinstance(action, NopAction):
                self._state = GameState.PLAY_OR_DRAW
                self._advance_turn()
            else:
                self._play_card(action)
        elif self._state == GameState.PLAY_DRAWN:
            if isinstance(action, NopAction):
                self._state = GameState.PLAY_OR_DRAW
                self._advance_turn()
            else:
                self._play_card(action)
        elif self._state == GameState.PICK_COLOR or self._state == GameState.PICK_COLOR_INIT:
            disc = self._discard[-1]
            disc.color = action.color
            if self._state == GameState.PICK_COLOR:
                last_disc = self._discard[-2]
                if disc.card_type == CardType.WILD:
                    self._state = GameState.PLAY_OR_DRAW
                elif any(x.color == last_disc.color for x in self._current_hand()):
                    self._state = GameState.CHALLENGE_INVALID
                else:
                    self._state = GameState.CHALLENGE_VALID
                self._advance_turn()
            else:
                self._state = GameState.PLAY
        elif self._state == GameState.CHALLENGE_VALID or self._state == GameState.CHALLENGE_INVALID:
            if isinstance(action, NopAction):
                for _ in range(4):
                    self._current_hand().append(self._draw())
                self._advance_turn()
            elif self._state == GameState.CHALLENGE_INVALID:
                self._advance_turn(by=-1)
                for _ in range(4):
                    self._current_hand().append(self._draw())
                self._advance_turn(by=2)
            else:
                for _ in range(6):
                    self._current_hand().append(self._draw())
                self._advance_turn()
            self._state = GameState.PLAY_OR_DRAW

    def _advance_turn(self, by=1):
        self._turn += by * self._direction
        while self._turn < 0:
            self._turn += self._num_players
        while self._turn >= self._num_players:
            self._turn -= self._num_players

    def _play_card(self, action):
        card = self._current_hand()[action.raw_index]
        self._current_hand().remove(card)
        self._discard.append(card)
        if card.card_type == CardType.NUMERAL:
            self._advance_turn()
        elif card.card_type == CardType.SKIP:
            self._advance_turn(by=2)
        elif card.card_type == CardType.REVERSE:
            self._direction *= -1
            self._advance_turn()
        elif card.card_type == CardType.DRAW_TWO:
            self._advance_turn()
            for _ in range(2):
                self._current_hand().append(self._draw())
            self._advance_turn()
        elif card.card_type == CardType.WILD or card.card_type == CardType.WILD_DRAW:
            self._state = GameState.PICK_COLOR
            return
        self._state = GameState.PLAY_OR_DRAW

    def _draw(self):
        if len(self._deck):
            return self._deck.pop()
        self._deck = self._discard[:-1]
        self._discard = [self._discard[-1]]
        random.shuffle(self._deck)
        for card in self._deck:
            if card.card_type == CardType.WILD or card.card_type == CardType.WILD_DRAW:
                card.color = None
        return self._deck.pop()

    def _update_init_state(self):
        first_card = self._discard[0]
        if first_card.card_type == CardType.SKIP:
            self._turn += 1
        elif first_card.card_type == CardType.REVERSE:
            self._direction = -1
            self._turn = self._num_players - 1
        elif first_card.card_type == CardType.DRAW_TWO:
            for _ in range(2):
                self._hands[0].append(self._draw())
            self._turn = 1
        elif first_card.card_type == CardType.WILD:
            self._state = GameState.PICK_COLOR_INIT

    def _can_play(self, card):
        if card.card_type == CardType.WILD or card.card_type == CardType.WILD_DRAW:
            return True
        disc = self._discard[-1]
        if card.color == disc.color:
            return True
        if card.card_type == CardType.NUMERAL and disc.card_type == CardType.NUMERAL:
            return card.number == disc.number
        return card.card_type == disc.card_type

    def _play_options(self):
        res = []
        for i, card in enumerate(self._current_hand()):
            if self._can_play(card):
                res.append(PlayCardAction(i))
        return res

    def _current_hand(self):
        return self._hands[self._turn]


def _player_vec(idx):
    res = [0.0] * MAX_PLAYERS
    res[idx] = 1.0
    return res
