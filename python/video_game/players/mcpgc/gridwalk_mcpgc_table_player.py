from . import mcpgc_table_model
from ..model import model_player

class GridWalkMCPGCTableModel(mcpgc_table_model.MCPGCTableModel):
    def get_softmax_temperature(self):
        return 3

class GridWalkMCPGCTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkMCPGCTableModel(state)
