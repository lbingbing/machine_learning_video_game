from . import model

class NNModel(model.Model):
    def __init__(self, state):
        super().__init__(state)

    def get_model_structure(self):
        return self.get_nn_framework() + '_nn'

    def get_nn_framework(self):
        raise NotImplementedError()
