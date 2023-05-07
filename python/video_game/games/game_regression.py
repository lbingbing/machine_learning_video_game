import argparse
import sys

from ..players import player

def print_progress(total_game_num, done_game_num, total_score, total_age):
    print('\rtotal/done/total_score/total_age: {}/{}/{}/{}'.format(total_game_num, done_game_num, total_score, total_age), end='', file=sys.stderr, flush=True)

def print_finish():
    print(file=sys.stderr)

def main(state, create_player):
    parser = argparse.ArgumentParser('{} game regression'.format(state.get_name()))
    player.add_player_options(parser)
    parser.add_argument('game_num', type=int, help='game num')
    args = parser.parse_args()

    p = create_player(state, args.player_type)

    total_game_num = args.game_num
    done_game_num = 0
    total_score = 0
    total_age = 0
    print_progress(total_game_num, done_game_num, total_score, total_age)
    for game_id in range(args.game_num):
        state.reset()
        while not state.is_end():
            action = p.get_action(state)
            state.do_action(action)
            state.update()
        total_score += state.get_score()
        total_age += state.get_age()
        done_game_num += 1
        print_progress(total_game_num, done_game_num, total_score, total_age)
    print_finish()
