from . import mcpgc_table_model
from ..model import model_player

class GridWalkMCPGCTableModel(mcpgc_table_model.MCPGCTableModel):
    pass

class GridWalkMCPGCTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkMCPGCTableModel(state)
