class QModel:
    def train(self, batch, learning_rate):
        raise NotImplementedError()

    def get_action_Q(self, state, action):
        raise NotImplementedError()

    def get_max_Q(self, state):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()
