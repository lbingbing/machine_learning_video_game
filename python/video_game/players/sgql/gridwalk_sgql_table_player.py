from . import sgql_table_model
from ..model import model_player

class GridWalkSGQLTableModel(sgql_table_model.SGQLTableModel):
    pass

class GridWalkSGQLTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkSGQLTableModel(state)
