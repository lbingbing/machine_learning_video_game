from ..model import q_torch_nn_model

class SGQLTorchNNModel(q_torch_nn_model.QTorchNNModel):
    def get_model_algorithm(self):
        return 'sgql'
