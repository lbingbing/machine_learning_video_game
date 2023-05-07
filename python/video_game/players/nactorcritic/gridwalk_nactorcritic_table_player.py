from . import nactorcritic_table_model
from ..model import model_player

class GridWalkNActorCriticTableModel(nactorcritic_table_model.NActorCriticTableModel):
    def get_softmax_temperature(self):
        return 3

class GridWalkNActorCriticTablePlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkNActorCriticTableModel(state)
