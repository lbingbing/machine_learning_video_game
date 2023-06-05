import argparse
import math
import time

def add_train_arguments(parser):
    parser.add_argument('--iteration_num', type=int, default=0, help='iteration num')
    parser.add_argument('--check_interval', type=int, default=500, help='check interval')
    parser.add_argument('--save_model_interval', type=int, default=100000, help='save model interval')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--monitor_port', type=int, help='monitor port')

def init_model(model, device):
    if device is not None:
        model.set_device(device)
    if model.exists():
        model.load()
        init_action = 'loaded'
    else:
        model.initialize()
        init_action = 'created'
    print('model {} {}, {} parameters, use {} device'.format(model.get_model_path(), init_action, model.get_parameter_number(), model.get_device()))

def get_dynamic_learning_rate(iteration_id, dynamic_learning_rate):
    if isinstance(dynamic_learning_rate, (int, float)):
        return dynamic_learning_rate
    else:
        start_learning_rate, decay_iteration = dynamic_learning_rate
        return start_learning_rate * math.exp(-iteration_id / decay_iteration)

def get_dynamic_epsilon(t, dynamic_epsilon):
    if isinstance(dynamic_epsilon, (int, float)):
        return dynamic_epsilon
    else:
        start_epsilon, end_epsilon, transition_t = dynamic_epsilon
        return (end_epsilon - start_epsilon) * t / transition_t + start_epsilon if t <= transition_t else end_epsilon

def get_current_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')
