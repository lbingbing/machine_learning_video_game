import os
import platform
import subprocess

os.chdir('machine_learning_video_game_release')

def run(cmd, timeout=None):
    if platform.system() in ('Linux', 'Darwin'):
        os.environ['PATH'] = os.environ.get('PATH', '') + ':.'
    print(' '.join(cmd))
    res = subprocess.run(cmd, timeout=timeout)
    return res.returncode == 0

def test_gridwalk_game_regression_random():
    assert run(['python', '-m', 'video_game.games.gridwalk_game_regression', 'random', '100'])

def test_flappybird_game_regression_random():
    assert run(['python', '-m', 'video_game.games.flappybird_game_regression', 'random', '100'])

def test_snake_game_regression_random():
    assert run(['python', '-m', 'video_game.games.snake_game_regression', 'random', '100'])

def test_tetris_game_regression_random():
    assert run(['python', '-m', 'video_game.games.tetris_game_regression', 'random', '100'])

def test_bombman_game_regression_random():
    assert run(['python', '-m', 'video_game.games.bombman_game_regression', 'random', '100'])

def test_packman_game_regression_random():
    assert run(['python', '-m', 'video_game.games.packman_game_regression', 'random', '100'])

def test_gridwalk_gmcc_table_train():
    assert run(['python', '-m', 'video_game.players.gmcc.gridwalk_gmcc_table_train', '--iteration_num', '5'])

def test_gridwalk_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.gridwalk_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.flappybird_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_snake_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.snake_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_tetris_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.tetris_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_bombman_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.bombman_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_packman_gmcc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.gmcc.packman_gmcc_torch_nn_train', '--iteration_num', '5'])

def test_gridwalk_sgnsarsa_table_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.gridwalk_sgnsarsa_table_train', '--iteration_num', '5'])

def test_gridwalk_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.gridwalk_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.flappybird_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_snake_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.snake_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_tetris_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.tetris_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_bombman_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.bombman_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_packman_sgnsarsa_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgnsarsa.packman_sgnsarsa_torch_nn_train', '--iteration_num', '5'])

def test_gridwalk_sgql_table_train():
    assert run(['python', '-m', 'video_game.players.sgql.gridwalk_sgql_table_train', '--iteration_num', '5'])

def test_gridwalk_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.gridwalk_sgql_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.flappybird_sgql_torch_nn_train', '--iteration_num', '5'])

def test_snake_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.snake_sgql_torch_nn_train', '--iteration_num', '5'])

def test_tetris_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.tetris_sgql_torch_nn_train', '--iteration_num', '5'])

def test_bombman_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.bombman_sgql_torch_nn_train', '--iteration_num', '5'])

def test_packman_sgql_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.sgql.packman_sgql_torch_nn_train', '--iteration_num', '5'])

def test_gridwalk_mcpgc_table_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.gridwalk_mcpgc_table_train', '--iteration_num', '5'])

def test_gridwalk_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.gridwalk_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.flappybird_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_snake_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.snake_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_tetris_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.tetris_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_bombman_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.bombman_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_packman_mcpgc_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgc.packman_mcpgc_torch_nn_train', '--iteration_num', '5'])

def test_gridwalk_mcpgcb_table_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.gridwalk_mcpgcb_table_train', '--iteration_num', '5'])

def test_gridwalk_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.gridwalk_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.flappybird_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_snake_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.snake_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_tetris_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.tetris_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_bombman_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.bombman_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_packman_mcpgcb_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.mcpgcb.packman_mcpgcb_torch_nn_train', '--iteration_num', '5'])

def test_gridwalk_nactorcritic_table_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.gridwalk_nactorcritic_table_train', '--iteration_num', '5'])

def test_gridwalk_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.gridwalk_nactorcritic_torch_nn_train', '--iteration_num', '5'])

def test_flappybird_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.flappybird_nactorcritic_torch_nn_train', '--iteration_num', '5'])

def test_snake_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.snake_nactorcritic_torch_nn_train', '--iteration_num', '5'])

def test_tetris_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.tetris_nactorcritic_torch_nn_train', '--iteration_num', '5'])

def test_bombman_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.bombman_nactorcritic_torch_nn_train', '--iteration_num', '5'])

def test_packman_nactorcritic_torch_nn_train():
    assert run(['python', '-m', 'video_game.players.nactorcritic.packman_nactorcritic_torch_nn_train', '--iteration_num', '5'])
