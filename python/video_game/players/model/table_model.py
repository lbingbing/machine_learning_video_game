from . import model

class TableModel(model.Model):
    def __init__(self, game_name, table):
        super().__init__(game_name)

        self.table = table

    def get_model_structure(self):
        return 'table'

    def create_table(self, state):
        raise NotImplementedError()

    def initialize(self):
        self.table.initialize()

    def get_parameter_number(self):
        return self.table.get_entry_number()

    def save(self):
        self.table.save(self.get_model_path())

    def load(self):
        self.table.load(self.get_model_path())
