import os

class Model:
    def __init__(self, game_name):
        self.game_name = game_name
        self.device = 'cpu'
        self.is_training = False

    def set_device(self, device):
        assert device in ('cpu', 'cuda')
        self.device = device

    def get_device(self):
        return self.device

    def set_training(self, b):
        self.is_training = b

    def get_model_structure(self):
        raise NotImplementedError()

    def get_model_algorithm(self):
        raise NotImplementedError()

    def get_model_path(self):
        return '{}_{}_{}_model'.format(self.game_name, self.get_model_algorithm(), self.get_model_structure())

    def exists(self):
        return os.path.isfile(self.get_model_path())

    def initialize(self):
        raise NotImplementedError()

    def get_parameter_number(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
