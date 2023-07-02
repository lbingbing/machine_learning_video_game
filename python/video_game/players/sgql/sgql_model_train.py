import argparse
import random
import itertools

from ..utils import train_utils
from ..utils import state_memory
from ..utils import replay_memory
from ..utils import train_flags
from ..utils import train_monitor

def get_default_configs():
    return {
        'episode_num_per_iteration': 1,
        'exploring_starts': [0, 0],
        'state_memory_size': 4096,
        'dynamic_epsilon': 0.1,
        'discount': 0.99,
        'replay_memory_size': 4096,
        'batch_num_per_iteration': 1,
        'batch_size': 32,
        'dynamic_learning_rate': 0.001,
        }

def sample(state, model, smemory, rmemory, configs, monitor):
    model.set_training(False)

    episode_num_per_iteration = configs['episode_num_per_iteration']
    exploring_starts = configs['exploring_starts']
    dynamic_epsilon = configs['dynamic_epsilon']
    discount = configs['discount']

    smemory.resize(configs['state_memory_size'])
    rmemory.resize(configs['replay_memory_size'])

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        is_exploring_starts, start_state = train_utils.get_exploring_starts(exploring_starts, smemory)
        if is_exploring_starts:
            state = start_state
        else:
            state.reset()
        if monitor:
            monitor.send_state(state.clone())
        for t in itertools.count(state.get_age()):
            state1 = state.clone()
            smemory.record(state1)
            epsilon = train_utils.get_dynamic_epsilon(t, dynamic_epsilon)
            if random.random() > epsilon:
                action = model.get_action(state)
            else:
                legal_actions = state.get_legal_actions()
                action = random.choice(legal_actions)
            state.do_action(action)
            state.update()
            R = state.get_reward()
            is_end = state.is_end()
            max_Q = 0 if is_end else model.get_max_Q(state)
            rmemory.record((state1, action, R+max_Q))
            if monitor:
                monitor.send_state(state.clone())
            if is_end:
                break
        if not is_exploring_starts:
            scores.append(state.get_score())
            ages.append(state.get_age())
    return scores, ages

def train(model, rmemory, configs, iteration_id):
    model.set_training(True)

    batch_num_per_iteration = configs['batch_num_per_iteration']
    batch_size = configs['batch_size']
    learning_rate = train_utils.get_dynamic_learning_rate(iteration_id, configs['dynamic_learning_rate'])

    losses = []
    for i in range(batch_num_per_iteration):
        batch = rmemory.sample(batch_size)
        loss = model.train(batch, learning_rate)
        losses.append(loss)
    return losses

def main(state, model, configs):
    parser = argparse.ArgumentParser('train {} sgql model'.format(state.get_name()))
    train_utils.add_train_arguments(parser)
    args = parser.parse_args()

    train_utils.init_train_log(model.get_model_path())

    train_utils.init_model(model, args.device)

    training_context = train_utils.create_training_context(model.get_model_path(), configs)
    start_iteration_id = training_context['start_iteration_id']
    configs = training_context['configs']

    smemory = state_memory.StateMemory(configs['state_memory_size'])
    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    losses = []
    scores = []
    ages = []
    with train_monitor.create_training_monitor(args.monitor_port) as monitor:
        for iteration_id in itertools.count(1):
            if iteration_id % args.check_interval == 1:
                train_flags.check_and_update_train_configs(model.get_model_path(), configs)
            scores1, ages1 = sample(state, model, smemory, rmemory, configs, monitor)
            losses1 = train(model, rmemory, configs, iteration_id)
            losses += losses1
            scores += scores1
            ages += ages1
            need_check = iteration_id % args.check_interval == 0
            if need_check:
                state.reset()
                model.set_training(False)
                max_Q = model.get_max_Q(state)
                avg_loss = sum(losses) / len(losses)
                avg_score = sum(scores) / len(scores)
                avg_age = sum(ages) / len(ages)
                train_utils.log('{} iter: {} loss: {:.2f} max_Q: {:.2f} score: {:.2f} age: {:.2f}'.format(train_utils.get_current_time_str(), iteration_id, avg_loss, max_Q, avg_score, avg_age))
                losses.clear()
                scores.clear()
                ages.clear()
            if iteration_id % args.save_model_interval == 0 or (need_check and train_flags.check_and_clear_save_model_flag_file(model.get_model_path())):
                model.save()
                training_context['start_iteration_id'] = iteration_id + 1
                train_utils.save_training_context(model.get_model_path(), training_context)
                train_utils.log('model {} saved'.format(model.get_model_path()))
            if need_check and train_flags.check_and_clear_stop_train_flag_file(model.get_model_path()):
                train_utils.log('stopped')
                break
            if args.iteration_num > 0 and iteration_id >= args.iteration_num:
                train_utils.log('finish')
                break
