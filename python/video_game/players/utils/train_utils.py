import argparse
import os
import sys
import logging
import json
import itertools
import shutil
import random
import math
import time

from . import train_flags

def add_train_arguments(parser):
    parser.add_argument('--iteration_num', type=int, default=0, help='iteration num')
    parser.add_argument('--check_interval', type=int, default=1000, help='check interval')
    parser.add_argument('--save_model_interval', type=int, default=100000, help='save model interval')
    parser.add_argument('--checkpoint_interval', type=int, default=100000, help='checkpoint interval')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--monitor_port', type=int, help='monitor port')

def init_training(model, device):
    if device is not None:
        model.set_device(device)
    if model.exists():
        model.load()
        init_action = 'loaded'
    else:
        model.initialize()
        init_action = 'created'
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(model.get_model_dir_path(), 'train.log'))])
    log('model {} {}, {} parameters, use {} device'.format(model.get_model_dir_path(), init_action, model.get_parameter_number(), model.get_device()))

def log(msg):
    logging.info(msg)

def get_training_context_file_path(model_dir_path):
    return os.path.join(model_dir_path, 'context.json')

def create_training_context(model_dir_path, configs):
    training_context_file_path = get_training_context_file_path(model_dir_path)
    if os.path.isfile(training_context_file_path):
        with open(training_context_file_path, encoding='utf-8') as f:
            context = json.load(f)
    else:
        context = {
            'done_iteration_num': 0,
            'configs': configs,
            }
    log("done_iteration_num: {}".format(context['done_iteration_num']))
    log('train configs:')
    for k, v in context['configs'].items():
        log('{}: {}'.format(k, v))
    return context

def save_training_context(model_dir_path, context):
    training_context_file_path = get_training_context_file_path(model_dir_path)
    with open(training_context_file_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=4)

def save_checkpoint(model):
    model_dir_path = model.get_model_dir_path()
    model_path = model.get_model_path()
    training_context_file_path = get_training_context_file_path(model_dir_path)
    for checkpoint_id in itertools.count(1):
        checkpoint_model_path = '{}.{}'.format(model_path, checkpoint_id)
        checkpoint_training_context_file_path = '{}.{}'.format(training_context_file_path, checkpoint_id)
        if not os.path.isfile(checkpoint_model_path) and not os.path.isfile(checkpoint_training_context_file_path):
            shutil.copy(model_path, checkpoint_model_path)
            shutil.copy(training_context_file_path, checkpoint_training_context_file_path)
            return checkpoint_id

def pre_iteration(iteration_id, check_interval, model, configs):
    if iteration_id % check_interval == 1:
        train_flags.check_and_update_train_configs(model.get_model_dir_path(), configs)

def post_iteration(iteration_id, iteration_num, check_interval, save_model_interval, checkpoint_interval, model, training_context, check_fn):
    training_context['done_iteration_num'] = iteration_id
    need_check = iteration_id % check_interval == 0
    need_save_model = iteration_id % save_model_interval == 0
    need_checkpoint = iteration_id % checkpoint_interval == 0
    stop = False
    if need_check:
        check_fn(iteration_id)
    if need_save_model or need_checkpoint or (need_check and train_flags.check_and_clear_save_model_flag_file(model.get_model_dir_path())):
        model.save()
        save_training_context(model.get_model_dir_path(), training_context)
        log('model saved')
    if need_checkpoint:
        checkpoint_id = save_checkpoint(model)
        log('checkpoint {} saved'.format(checkpoint_id))
    if need_check and train_flags.check_and_clear_stop_train_flag_file(model.get_model_dir_path()):
        log('stopped')
        stop = True
    if iteration_num > 0 and iteration_id >= iteration_num:
        log('finish')
        stop = True
    return stop

def try_update_train_configs(config_file, configs):
    try:
        with open(config_file, encoding='utf-8') as f:
            new_configs = json.load(f)
    except json.decoder.JSONDecodeError:
        return False
    log('update train configs:')
    for k in configs:
        if k in new_configs and new_configs[k] != configs[k]:
            log('{}: {} -> {}'.format(k, configs[k], new_configs[k]))
        else:
            log('{}: {}'.format(k, configs[k]))
    for k in new_configs:
        if k not in configs:
            log('{}: - -> {}'.format(k, new_configs[k]))
    configs.update(new_configs)
    return True

def get_exploring_starts(exploring_starts, smemory):
    exploring_starts_ratio, exploring_starts_age = exploring_starts
    is_exploring_starts = exploring_starts_ratio > 0 and random.random() < exploring_starts_ratio and smemory.has_age(exploring_starts_age)
    if is_exploring_starts:
        start_state = smemory.sample(exploring_starts_age).clone()
    else:
        start_state = None
    return is_exploring_starts, start_state

def get_dynamic_epsilon(t, dynamic_epsilon):
    if isinstance(dynamic_epsilon, (int, float)):
        return dynamic_epsilon
    else:
        epsilon_t_pairs = [[dynamic_epsilon[0], 0]] + dynamic_epsilon[1:]
        for (epsilon1, t1), (epsilon2, t2) in zip(epsilon_t_pairs[:-1], epsilon_t_pairs[1:]):
            if t1 <= t < t2:
                return (epsilon2 - epsilon1) * (t - t1) / (t2 - t1) + epsilon1
        return epsilon_t_pairs[-1][0]

def get_dynamic_learning_rate(iteration_id, dynamic_learning_rate):
    if isinstance(dynamic_learning_rate, (int, float)):
        return dynamic_learning_rate
    else:
        learning_rate_t_pairs = [[dynamic_learning_rate[0], 0]] + dynamic_learning_rate[1:]
        for (learning_rate1, iteration_id1), (learning_rate2, iteration_id2) in zip(learning_rate_t_pairs[:-1], learning_rate_t_pairs[1:]):
            if iteration_id1 <= iteration_id < iteration_id2:
                return learning_rate1 * math.pow(learning_rate2 / learning_rate1, (iteration_id - iteration_id1) / (iteration_id2 - iteration_id1))
        return learning_rate_t_pairs[-1][0]

def get_current_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')
