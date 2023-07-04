import os
import json

from . import train_utils

model_save_flag_file = 'model_save.flag'
model_saved_flag_file = 'model_saved.flag'
train_stop_flag_file = 'train_stop.flag'
train_stopped_flag_file = 'train_stopped.flag'
train_configs_update_flag_file = 'train_configs_update.flag'
train_configs_updated_flag_file = 'train_configs_updated.flag'

def create_save_model_flag_file(model_dir_path):
    if os.path.isfile(os.path.join(model_dir_path, model_saved_flag_file)):
        os.rename(os.path.join(model_dir_path, model_saved_flag_file), os.path.join(model_dir_path, model_save_flag_file))
    else:
        with open(os.path.join(model_dir_path, model_save_flag_file), 'w', encoding='utf-8') as f:
            pass
        
def create_stop_train_flag_file(model_dir_path):
    if os.path.isfile(os.path.join(model_dir_path, train_stopped_flag_file)):
        os.rename(os.path.join(model_dir_path, train_stopped_flag_file), os.path.join(model_dir_path, train_stop_flag_file))
    else:
        with open(os.path.join(model_dir_path, train_stop_flag_file), 'w', encoding='utf-8') as f:
            pass

def create_train_configs_flag_file(model_dir_path):
    if os.path.isfile(os.path.join(model_dir_path, train_configs_updated_flag_file)):
        os.rename(os.path.join(model_dir_path, train_configs_updated_flag_file), os.path.join(model_dir_path, train_configs_update_flag_file))
    else:
        with open(os.path.join(model_dir_path, train_configs_update_flag_file), 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
def check_and_clear_save_model_flag_file(model_dir_path):
    if os.path.isfile(model_save_flag_file):
        return True
    elif os.path.isfile(os.path.join(model_dir_path, model_save_flag_file)):
        os.rename(os.path.join(model_dir_path, model_save_flag_file), os.path.join(model_dir_path, model_saved_flag_file))
        return True
    else:
        return False
        
def check_and_clear_stop_train_flag_file(model_dir_path):
    if os.path.isfile(train_stop_flag_file):
        return True
    elif os.path.isfile(os.path.join(model_dir_path, train_stop_flag_file)):
        os.rename(os.path.join(model_dir_path, train_stop_flag_file), os.path.join(model_dir_path, train_stopped_flag_file))
        return True
    else:
        return False

def check_and_update_train_configs(model_dir_path, configs):
    if os.path.isfile(train_configs_update_flag_file):
        success = train_utils.try_update_train_configs(train_configs_update_flag_file, configs)
        return success
    elif os.path.isfile(os.path.join(model_dir_path, train_configs_update_flag_file)):
        success = train_utils.try_update_train_configs(os.path.join(model_dir_path, train_configs_update_flag_file), configs)
        if success:
            os.rename(os.path.join(model_dir_path, train_configs_update_flag_file), os.path.join(model_dir_path, train_configs_updated_flag_file))
        return success
    else:
        return False
