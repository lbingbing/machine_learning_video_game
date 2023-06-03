class PVModel:
    def train(self, batch, learning_rate, vloss_factor):
        raise NotImplementedError()

    def get_V(self, state):
        raise NotImplementedError()

    def get_legal_P_logit_range(self, state):
        raise NotImplementedError()

    def get_P(self, state):
        raise NotImplementedError()

    def get_legal_P_range(self, state):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()
