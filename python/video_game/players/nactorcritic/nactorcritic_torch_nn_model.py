from ..model import pv_torch_nn_model

class NActorCriticTorchNNModel(pv_torch_nn_model.PVTorchNNModel):
    def get_model_algorithm(self):
        return 'nactorcritic'
