from . import gmcc_table_model
from ..model import model_player

class GridWalkGMCCTableModel(gmcc_table_model.GMCCTableModel):
    pass

class GridWalkGMCCTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkGMCCTableModel(state)
