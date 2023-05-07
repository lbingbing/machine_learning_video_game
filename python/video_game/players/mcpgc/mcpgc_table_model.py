from ..model import p_table_model

class MCPGCTableModel(p_table_model.PTableModel):
    def get_model_algorithm(self):
        return 'mcpgc'
