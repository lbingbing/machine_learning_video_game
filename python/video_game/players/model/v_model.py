class VModel:
    def train(self, batch, learning_rate):
        raise NotImplementedError()

    def get_V(self, state):
        raise NotImplementedError()
