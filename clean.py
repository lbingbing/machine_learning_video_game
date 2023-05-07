import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--release', action='store_true', help='clean for release')
args = parser.parse_args()

def rmtree(dir_path):
    if os.path.isdir(dir_path):
        print('removing', dir_path)
        shutil.rmtree(dir_path)

rmtree('__pycache__')
rmtree('python/__pycache__')
rmtree('.pytest_cache')
rmtree('build')
rmtree('machine_learning_video_game_release/__pycache__')
if not args.release:
    rmtree('machine_learning_video_game_release')
