from ..model import q_torch_nn_model

class SGNSarsaTorchNNModel(q_torch_nn_model.QTorchNNModel):
    def get_model_algorithm(self):
        return 'sgnsarsa'
