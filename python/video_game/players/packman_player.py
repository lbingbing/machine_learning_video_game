from . import player

def create_player(state, player_type):
    p = None
    if player_type == player.HUMAN_PLAYER:
        from .human import human_player
        p = human_player.HumanPlayer()
    elif player_type == player.RANDOM_PLAYER:
        from .random import random_player
        p = random_player.RandomPlayer()
    elif player_type == player.GMCC_TORCH_NN_PLAYER:
        from .gmcc import packman_gmcc_torch_nn_player
        p = packman_gmcc_torch_nn_player.PackManGMCCTorchNNPlayer(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import packman_sgnsarsa_torch_nn_player
        p = packman_sgnsarsa_torch_nn_player.PackManSGNSarsaTorchNNPlayer(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import packman_sgql_torch_nn_player
        p = packman_sgql_torch_nn_player.PackManSGQLTorchNNPlayer(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import packman_mcpgc_torch_nn_player
        p = packman_mcpgc_torch_nn_player.PackManMCPGCTorchNNPlayer(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import packman_mcpgcb_torch_nn_player
        p = packman_mcpgcb_torch_nn_player.PackManMCPGCBTorchNNPlayer(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import packman_nactorcritic_torch_nn_player
        p = packman_nactorcritic_torch_nn_player.PackManNActorCriticTorchNNPlayer(state)
    else:
        assert False
    return p
