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
        from .gmcc import bombman_gmcc_torch_nn_player
        p = bombman_gmcc_torch_nn_player.BombManGMCCTorchNNPlayer(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import bombman_sgnsarsa_torch_nn_player
        p = bombman_sgnsarsa_torch_nn_player.BombManSGNSarsaTorchNNPlayer(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import bombman_sgql_torch_nn_player
        p = bombman_sgql_torch_nn_player.BombManSGQLTorchNNPlayer(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import bombman_mcpgc_torch_nn_player
        p = bombman_mcpgc_torch_nn_player.BombManMCPGCTorchNNPlayer(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import bombman_mcpgcb_torch_nn_player
        p = bombman_mcpgcb_torch_nn_player.BombManMCPGCBTorchNNPlayer(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import bombman_nactorcritic_torch_nn_player
        p = bombman_nactorcritic_torch_nn_player.BombManNActorCriticTorchNNPlayer(state)
    else:
        assert False
    return p
