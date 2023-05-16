from . import model

class TableModel(model.Model):
    def __init__(self, state):
        super().__init__(state.get_name())

        self.table = self.create_table(state)

    def get_model_structure(self):
        return 'table'

    def create_table(self, state):
        raise NotImplementedError()

    def initialize(self):
        self.table.initialize()

    def save(self):
        self.table.save(self.get_model_path())

    def load(self):
        self.table.load(self.get_model_path())
