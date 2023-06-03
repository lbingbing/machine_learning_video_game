class PModel:
    def train(self, batch, learning_rate):
        raise NotImplementedError()

    def get_legal_P_logit_range(self, state):
        raise NotImplementedError()

    def get_P(self, state):
        raise NotImplementedError()

    def get_legal_P_range(self, state):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()
