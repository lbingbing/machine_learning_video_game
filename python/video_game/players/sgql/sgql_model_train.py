import argparse
import random
import itertools

from ..utils import train_utils
from ..utils import replay_memory
from ..model import train_flags

def select_action(state, model, epsilon):
    if random.random() > epsilon:
        action = model.get_opt_action(state)
    else:
        legal_actions = state.get_legal_actions()
        action = random.choice(legal_actions)
    return action

def sample(state, model, rmemory, configs):
    episode_num_per_iteration = configs['episode_num_per_iteration']
    dynamic_epsilon = configs['dynamic_epsilon']
    discount = configs['discount']

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        state.reset()
        for t in itertools.count():
            state1 = state.clone()
            epsilon = train_utils.get_dynamic_epsilon(t, dynamic_epsilon)
            if random.random() > epsilon:
                action = model.get_opt_action(state)
            else:
                legal_actions = state.get_legal_actions()
                action = random.choice(legal_actions)
            state.do_action(action)
            state.update()
            R = state.get_reward()
            is_end = state.is_end()
            max_Q = 0 if is_end else model.get_max_Q(state)
            rmemory.record((state1, action, R+max_Q))
            if is_end:
                break
        scores.append(state.get_score())
        ages.append(state.get_age())
    return scores, ages

def train(model, rmemory, configs):
    batch_num_per_iteration = configs['batch_num_per_iteration']
    batch_size = configs['batch_size']
    learning_rate = configs['learning_rate']

    losses = []
    for i in range(batch_num_per_iteration):
        batch = rmemory.sample(batch_size)
        loss = model.train(batch, learning_rate)
        losses.append(loss)
    return losses

def main(state, model, configs):
    parser = argparse.ArgumentParser('train {} sgql model'.format(state.get_name()))
    args = parser.parse_args()

    for k, v in configs.items():
        print('{}: {}'.format(k, v))

    check_interval = configs['check_interval']
    save_model_interval = configs['save_model_interval']

    if model.exists():
        model.load()
        print('model {} loaded'.format(model.get_model_path()))
    else:
        print('model {} created'.format(model.get_model_path()))
    print('use {} device'.format(model.get_device()))

    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    losses = []
    scores = []
    ages = []
    for iteration_id in itertools.count(1):
        scores1, ages1 = sample(state, model, rmemory, configs)
        losses1 = train(model, rmemory, configs)
        losses += losses1
        scores += scores1
        ages += ages1
        need_check = iteration_id % check_interval == 0
        if need_check:
            state.reset()
            max_Q = model.get_max_Q(state)
            avg_loss = sum(losses) / len(losses)
            avg_score = sum(scores) / len(scores)
            avg_age = sum(ages) / len(ages)
            losses.clear()
            scores.clear()
            ages.clear()
            print('iteration: {} avg_loss: {:.8f} max_Q: {:.8f} avg_score: {:.2f} avg_age: {:.2f}'.format(iteration_id, avg_loss, max_Q, avg_score, avg_age))
            train_flags.check_and_update_train_configs(model.get_model_path(), configs)
        if iteration_id % save_model_interval == 0 or (need_check and train_flags.check_and_clear_save_model_flag_file(model.get_model_path())):
            model.save()
            print('model {} saved'.format(model.get_model_path()))
        if need_check and train_flags.check_and_clear_stop_train_flag_file(model.get_model_path()):
            print('stopped')
            break
