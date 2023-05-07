from ..model import pv_table_model

class NActorCriticTableModel(pv_table_model.PVTableModel):
    def get_model_algorithm(self):
        return 'nactorcritic'
