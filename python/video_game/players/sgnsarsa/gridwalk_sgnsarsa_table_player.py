from . import sgnsarsa_table_model
from ..model import model_player

class GridWalkSGNSarsaTableModel(sgnsarsa_table_model.SGNSarsaTableModel):
    pass

class GridWalkSGNSarsaTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkSGNSarsaTableModel(state)
