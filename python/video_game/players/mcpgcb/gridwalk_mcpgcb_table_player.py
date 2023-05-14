from . import mcpgcb_table_model
from ..model import model_player

class GridWalkMCPGCBTableModel(mcpgcb_table_model.MCPGCBTableModel):
    pass

class GridWalkMCPGCBTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkMCPGCBTableModel(state)
