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
        from .gmcc import flappybird_gmcc_torch_nn_player
        p = flappybird_gmcc_torch_nn_player.FlappyBirdGMCCTorchNNPlayer(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import flappybird_sgnsarsa_torch_nn_player
        p = flappybird_sgnsarsa_torch_nn_player.FlappyBirdSGNSarsaTorchNNPlayer(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import flappybird_sgql_torch_nn_player
        p = flappybird_sgql_torch_nn_player.FlappyBirdSGQLTorchNNPlayer(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import flappybird_mcpgc_torch_nn_player
        p = flappybird_mcpgc_torch_nn_player.FlappyBirdMCPGCTorchNNPlayer(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import flappybird_mcpgcb_torch_nn_player
        p = flappybird_mcpgcb_torch_nn_player.FlappyBirdMCPGCBTorchNNPlayer(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import flappybird_nactorcritic_torch_nn_player
        p = flappybird_nactorcritic_torch_nn_player.FlappyBirdNActorCriticTorchNNPlayer(state)
    else:
        assert False
    return p
