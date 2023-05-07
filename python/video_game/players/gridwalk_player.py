from . import player

def create_player(state, player_type):
    p = None
    if player_type == player.HUMAN_PLAYER:
        from .human import human_player
        p = human_player.HumanPlayer()
    elif player_type == player.RANDOM_PLAYER:
        from .random import random_player
        p = random_player.RandomPlayer()
    elif player_type == player.GMCC_TABLE_PLAYER:
        from .gmcc import gridwalk_gmcc_table_player
        p = gridwalk_gmcc_table_player.GridWalkGMCCTablePlayer(state)
    elif player_type == player.GMCC_TORCH_NN_PLAYER:
        from .gmcc import gridwalk_gmcc_torch_nn_player
        p = gridwalk_gmcc_torch_nn_player.GridWalkGMCCTorchNNPlayer(state)
    elif player_type == player.SGNSARSA_TABLE_PLAYER:
        from .sgnsarsa import gridwalk_sgnsarsa_table_player
        p = gridwalk_sgnsarsa_table_player.GridWalkSGNSarsaTablePlayer(state)
    elif player_type == player.SGNSARSA_TORCH_NN_PLAYER:
        from .sgnsarsa import gridwalk_sgnsarsa_torch_nn_player
        p = gridwalk_sgnsarsa_torch_nn_player.GridWalkSGNSarsaTorchNNPlayer(state)
    elif player_type == player.SGQL_TABLE_PLAYER:
        from .sgql import gridwalk_sgql_table_player
        p = gridwalk_sgql_table_player.GridWalkSGQLTablePlayer(state)
    elif player_type == player.SGQL_TORCH_NN_PLAYER:
        from .sgql import gridwalk_sgql_torch_nn_player
        p = gridwalk_sgql_torch_nn_player.GridWalkSGQLTorchNNPlayer(state)
    elif player_type == player.MCPGC_TABLE_PLAYER:
        from .mcpgc import gridwalk_mcpgc_table_player
        p = gridwalk_mcpgc_table_player.GridWalkMCPGCTablePlayer(state)
    elif player_type == player.MCPGC_TORCH_NN_PLAYER:
        from .mcpgc import gridwalk_mcpgc_torch_nn_player
        p = gridwalk_mcpgc_torch_nn_player.GridWalkMCPGCTorchNNPlayer(state)
    elif player_type == player.MCPGCB_TABLE_PLAYER:
        from .mcpgcb import gridwalk_mcpgcb_table_player
        p = gridwalk_mcpgcb_table_player.GridWalkMCPGCBTablePlayer(state)
    elif player_type == player.MCPGCB_TORCH_NN_PLAYER:
        from .mcpgcb import gridwalk_mcpgcb_torch_nn_player
        p = gridwalk_mcpgcb_torch_nn_player.GridWalkMCPGCBTorchNNPlayer(state)
    elif player_type == player.NACTORCRITIC_TABLE_PLAYER:
        from .nactorcritic import gridwalk_nactorcritic_table_player
        p = gridwalk_nactorcritic_table_player.GridWalkNActorCriticTablePlayer(state)
    elif player_type == player.NACTORCRITIC_TORCH_NN_PLAYER:
        from .nactorcritic import gridwalk_nactorcritic_torch_nn_player
        p = gridwalk_nactorcritic_torch_nn_player.GridWalkNActorCriticTorchNNPlayer(state)
    else:
        assert False
    return p
