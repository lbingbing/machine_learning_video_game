from ..model import pv_table_model

class MCPGCBTableModel(pv_table_model.PVTableModel):
    def get_model_algorithm(self):
        return 'mcpgcb'
