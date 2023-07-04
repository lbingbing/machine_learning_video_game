import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--release', action='store_true', help='clean for release')
args = parser.parse_args()

def remove_file(file_path):
    if os.path.isfile(file_path):
        print('removing', file_path)
        os.remove(file_path)

def remove_tree(dir_path):
    if os.path.isdir(dir_path):
        print('removing {}/'.format(dir_path))
        shutil.rmtree(dir_path)

remove_tree('__pycache__')
remove_tree('python/__pycache__')
remove_tree('.pytest_cache')
remove_tree('build')
release_dir = 'machine_learning_video_game_release'
if args.release:
    remove_tree(os.path.join(release_dir, '__pycache__'))
    remove_file(os.path.join(release_dir, 'train_configs_update'))
    for name in os.listdir(release_dir):
        path = os.path.join(release_dir, name)
        if os.path.isdir(path) and name.endswith('_model'):
            remove_tree(path)
else:
    remove_tree(release_dir)
