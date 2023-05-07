from .. import player

class HumanPlayer(player.Player):
    def get_type(self):
        return player.HUMAN_PLAYER
