import random

from .. import player

class RandomPlayer(player.Player):
    def get_type(self):
        return player.RANDOM_PLAYER

    def get_action(self, state):
        return random.choice(state.get_legal_actions())
