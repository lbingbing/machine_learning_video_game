HUMAN_PLAYER                 = 'human'
RANDOM_PLAYER                = 'random'
GMCC_TABLE_PLAYER            = 'gmcc_table'
GMCC_TORCH_NN_PLAYER         = 'gmcc_torch_nn'
SGNSARSA_TABLE_PLAYER        = 'sgnsarsa_table'
SGNSARSA_TORCH_NN_PLAYER     = 'sgnsarsa_torch_nn'
SGQL_TABLE_PLAYER            = 'sgql_table'
SGQL_TORCH_NN_PLAYER         = 'sgql_torch_nn'
MCPGC_TABLE_PLAYER           = 'mcpgc_table'
MCPGC_TORCH_NN_PLAYER        = 'mcpgc_torch_nn'
MCPGCB_TABLE_PLAYER          = 'mcpgcb_table'
MCPGCB_TORCH_NN_PLAYER       = 'mcpgcb_torch_nn'
NACTORCRITIC_TABLE_PLAYER    = 'nactorcritic_table'
NACTORCRITIC_TORCH_NN_PLAYER = 'nactorcritic_torch_nn'

PLAYER_TYPES = [
    HUMAN_PLAYER,
    RANDOM_PLAYER,
    GMCC_TABLE_PLAYER,
    GMCC_TORCH_NN_PLAYER,
    SGNSARSA_TABLE_PLAYER,
    SGNSARSA_TORCH_NN_PLAYER,
    SGQL_TABLE_PLAYER,
    SGQL_TORCH_NN_PLAYER,
    MCPGC_TABLE_PLAYER,
    MCPGC_TORCH_NN_PLAYER,
    MCPGCB_TABLE_PLAYER,
    MCPGCB_TORCH_NN_PLAYER,
    NACTORCRITIC_TABLE_PLAYER,
    NACTORCRITIC_TORCH_NN_PLAYER,
    ]

class Player:
    def get_type(self):
        raise NotImplementedError()

    def get_action(self, state):
        raise NotImplementedError()

def is_human(player):
    return player.get_type() == HUMAN_PLAYER

def add_player_options(parser):
    parser.add_argument('player_type', choices=PLAYER_TYPES, help='player type')
