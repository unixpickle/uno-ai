class Game:
    def __init__(self, num_players):
        self.num_players = num_players

    def done(self):
        """
        Check if the game has ended.
        """
        return False

    def mask(self, player):
        """
        Get the current action mask for the player.
        """
        pass

    def obs(self, player):
        """
        Generate an observation vector for the player.
        """
        pass

    def act(self, logits):
        """
        Perform actions by providing the logits for all of
        the players.

        Args:
            logits: a list of torch Tensors, one per player.
              These will automatically be masked.
        """
        pass
