from ..model import q_table_model

class SGQLTableModel(q_table_model.QTableModel):
    def get_model_algorithm(self):
        return 'sgql'
