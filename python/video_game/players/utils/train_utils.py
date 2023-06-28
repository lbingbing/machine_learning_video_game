import argparse
import os
import sys
import logging
import json
import math
import time

def add_train_arguments(parser):
    parser.add_argument('--iteration_num', type=int, default=0, help='iteration num')
    parser.add_argument('--check_interval', type=int, default=500, help='check interval')
    parser.add_argument('--save_model_interval', type=int, default=100000, help='save model interval')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--monitor_port', type=int, help='monitor port')

def init_model_log(model_path):
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('{}.log'.format(model_path))])

def log(msg):
    logging.info(msg)

def init_model(model, device):
    if device is not None:
        model.set_device(device)
    if model.exists():
        model.load()
        init_action = 'loaded'
    else:
        model.initialize()
        init_action = 'created'
    log('model {} {}, {} parameters, use {} device'.format(model.get_model_path(), init_action, model.get_parameter_number(), model.get_device()))

def get_training_context_file(model_path):
    return model_path + '.context'

def create_training_context(model_path, configs):
    training_context_file = get_training_context_file(model_path)
    if os.path.isfile(training_context_file):
        with open(training_context_file, encoding='utf-8') as f:
            context = json.load(f)
    else:
        context = {
            'start_iteration_id': 1,
            'configs': configs,
            }
    log("start_iteration_id: {}".format(context['start_iteration_id']))
    log('train configs:')
    for k, v in context['configs'].items():
        log('{}: {}'.format(k, v))
    return context

def save_training_context(model_path, context):
    training_context_file = get_training_context_file(model_path)
    with open(training_context_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=4)

def try_update_train_configs(config_file, configs):
    try:
        with open(config_file, encoding='utf-8') as f:
            new_configs = json.load(f)
    except json.decoder.JSONDecodeError:
        return False
    log('train configs:')
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

def get_dynamic_learning_rate(iteration_id, dynamic_learning_rate):
    if isinstance(dynamic_learning_rate, (int, float)):
        return dynamic_learning_rate
    else:
        learning_rate_t_pairs = [[dynamic_learning_rate[0], 0]] + dynamic_learning_rate[1:]
        for (learning_rate1, iteration_id1), (learning_rate2, iteration_id2) in zip(learning_rate_t_pairs[:-1], learning_rate_t_pairs[1:]):
            if iteration_id1 <= iteration_id < iteration_id2:
                return learning_rate1 * math.pow(learning_rate2 / learning_rate1, (iteration_id - iteration_id1) / (iteration_id2 - iteration_id1))
        return learning_rate_t_pairs[-1][0]

def get_dynamic_epsilon(t, dynamic_epsilon):
    if isinstance(dynamic_epsilon, (int, float)):
        return dynamic_epsilon
    else:
        epsilon_t_pairs = [[dynamic_epsilon[0], 0]] + dynamic_epsilon[1:]
        for (epsilon1, t1), (epsilon2, t2) in zip(epsilon_t_pairs[:-1], epsilon_t_pairs[1:]):
            if t1 <= t < t2:
                return (epsilon2 - epsilon1) * (t - t1) / (t2 - t1) + epsilon1
        return epsilon_t_pairs[-1][0]

def get_current_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')
