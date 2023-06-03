from .. import player

class ModelPlayer(player.Player):
    def __init__(self, model):
        assert model.exists()
        self.model = model
        self.model.load()
        self.model.set_training(False)

    def get_type(self):
        return '{}_{}'.format(self.model.get_model_algorithm(), self.model.get_model_structure())

    def get_action(self, state):
        return self.model.get_action(state)
