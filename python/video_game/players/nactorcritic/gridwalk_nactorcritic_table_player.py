from . import nactorcritic_table_model
from ..model import model_player

class GridWalkNActorCriticTableModel(nactorcritic_table_model.NActorCriticTableModel):
    pass

class GridWalkNActorCriticTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkNActorCriticTableModel(state)
