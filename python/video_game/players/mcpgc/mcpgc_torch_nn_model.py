from ..model import p_torch_nn_model

class MCPGCTorchNNModel(p_torch_nn_model.PTorchNNModel):
    def get_model_algorithm(self):
        return 'mcpgc'
