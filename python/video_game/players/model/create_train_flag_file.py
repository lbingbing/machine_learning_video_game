import argparse

from . import train_flags

parser = argparse.ArgumentParser()
parser.add_argument('flags', nargs='+', choices=['model_save', 'train_stop', 'train_configs'], help='flags')
parser.add_argument('--model_path', help='model path')
args = parser.parse_args()

if args.model_path:
    if 'model_save' in args.flags:
        train_flags.create_save_model_flag_file(args.model_path)
    if 'train_stop' in args.flags:
        train_flags.create_stop_train_flag_file(args.model_path)
    if 'train_configs' in args.flags:
        train_flags.create_train_configs_flag_file(args.model_path)
else:
    if 'model_save' in args.flags:
        train_flags.create_save_all_model_flag_file()
    if 'train_stop' in args.flags:
        train_flags.create_stop_all_train_flag_file()
