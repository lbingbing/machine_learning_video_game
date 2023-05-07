import os
import json

all_model_save_flag_file = 'model_save'
model_save_flag_file_suffix = '.model_save'
model_saved_flag_file_suffix = '.model_saved'
all_train_stop_flag_file = 'train_stop'
train_stop_flag_file_suffix = '.train_stop'
train_stopped_flag_file_suffix = '.train_stopped'
train_configs_update_file_suffix = '.train_configs_update'
train_configs_updated_file_suffix = '.train_configs_updated'

def create_save_all_model_flag_file():
    with open(all_model_save_flag_file, 'w', encoding='utf-8') as f:
        pass

def create_save_model_flag_file(model_path):
    if os.path.isfile(model_path+model_saved_flag_file_suffix):
        os.rename(model_path+model_saved_flag_file_suffix, model_path+model_save_flag_file_suffix)
    else:
        with open(model_path+model_save_flag_file_suffix, 'w', encoding='utf-8') as f:
            pass
        
def create_stop_all_train_flag_file():
    with open(all_train_stop_flag_file, 'w', encoding='utf-8') as f:
        pass

def create_stop_train_flag_file(model_path):
    if os.path.isfile(model_path+train_stopped_flag_file_suffix):
        os.rename(model_path+train_stopped_flag_file_suffix, model_path+train_stop_flag_file_suffix)
    else:
        with open(model_path+train_stop_flag_file_suffix, 'w', encoding='utf-8') as f:
            pass

def create_train_configs_flag_file(model_path):
    if os.path.isfile(model_path+train_configs_updated_file_suffix):
        os.rename(model_path+train_configs_updated_file_suffix, model_path+train_configs_update_file_suffix)
    else:
        with open(model_path+train_configs_update_file_suffix, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
def check_and_clear_save_model_flag_file(model_path):
    if os.path.isfile(all_model_save_flag_file):
        return True
    elif os.path.isfile(model_path+model_save_flag_file_suffix):
        os.rename(model_path+model_save_flag_file_suffix, model_path+model_saved_flag_file_suffix)
        return True
    else:
        return False
        
def check_and_clear_stop_train_flag_file(model_path):
    if os.path.isfile(all_train_stop_flag_file):
        return True
    elif os.path.isfile(model_path+train_stop_flag_file_suffix):
        os.rename(model_path+train_stop_flag_file_suffix, model_path+train_stopped_flag_file_suffix)
        return True
    else:
        return False

def print_train_configs(configs):
    for k, v in configs.items():
        print('{}: {}'.format(k, v))

def check_and_update_train_configs(model_path, configs):
    if os.path.isfile(model_path+train_configs_update_file_suffix):
        try:
            with open(model_path+train_configs_update_file_suffix, encoding='utf-8') as f:
                new_configs = json.load(f)
        except json.decoder.JSONDecodeError:
            return False
        print('old train configs:')
        print_train_configs(configs)
        configs.update(new_configs)
        print('new train configs:')
        print_train_configs(configs)
        os.rename(model_path+train_configs_update_file_suffix, model_path+train_configs_updated_file_suffix)
        return False
    else:
        return False
