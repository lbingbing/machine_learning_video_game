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
        from .gmcc import snake_gmcc_torch_nn_player
        p = snake_gmcc_torch_nn_player.create_player(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import snake_sgnsarsa_torch_nn_player
        p = snake_sgnsarsa_torch_nn_player.create_player(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import snake_sgql_torch_nn_player
        p = snake_sgql_torch_nn_player.create_player(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import snake_mcpgc_torch_nn_player
        p = snake_mcpgc_torch_nn_player.create_player(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import snake_mcpgcb_torch_nn_player
        p = snake_mcpgcb_torch_nn_player.create_player(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import snake_nactorcritic_torch_nn_player
        p = snake_nactorcritic_torch_nn_player.create_player(state)
    else:
        assert False
    return p
