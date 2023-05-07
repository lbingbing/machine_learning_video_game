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
        from .gmcc import tetris_gmcc_torch_nn_player
        p = tetris_gmcc_torch_nn_player.TetrisGMCCTorchNNPlayer(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import tetris_sgnsarsa_torch_nn_player
        p = tetris_sgnsarsa_torch_nn_player.TetrisSGNSarsaTorchNNPlayer(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import tetris_sgql_torch_nn_player
        p = tetris_sgql_torch_nn_player.TetrisSGQLTorchNNPlayer(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import tetris_mcpgc_torch_nn_player
        p = tetris_mcpgc_torch_nn_player.TetrisMCPGCTorchNNPlayer(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import tetris_mcpgcb_torch_nn_player
        p = tetris_mcpgcb_torch_nn_player.TetrisMCPGCBTorchNNPlayer(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import tetris_nactorcritic_torch_nn_player
        p = tetris_nactorcritic_torch_nn_player.TetrisNActorCriticTorchNNPlayer(state)
    else:
        assert False
    return p
