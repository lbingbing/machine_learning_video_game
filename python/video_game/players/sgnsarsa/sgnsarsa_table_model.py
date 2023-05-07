from ..model import q_table_model

class SGNSarsaTableModel(q_table_model.QTableModel):
    def get_model_algorithm(self):
        return 'sgnsarsa'
