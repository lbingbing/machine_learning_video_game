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
    step_num = configs['step_num']
    dynamic_epsilon = configs['dynamic_epsilon']
    discount = configs['discount']

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        samples = []
        state.reset()
        epsilon = train_utils.get_dynamic_epsilon(0, dynamic_epsilon)
        action = select_action(state, model, epsilon)
        is_end = False
        for t in itertools.count():
            if not is_end:
                state1 = state.clone()
                state.do_action(action)
                state.update()
                R = state.get_reward()
                samples.append((state1, action, R))
                if state.is_end():
                    is_end = True
                    scores.append(state.get_score())
                    ages.append(state.get_age())
                else:
                    epsilon = train_utils.get_dynamic_epsilon(t+1, dynamic_epsilon)
                    action = select_action(state, model, epsilon)
            if len(samples) >= step_num or is_end:
                G = 0
                discount_factor = 1
                for S, A, R in samples:
                    G += R * discount_factor
                    discount_factor *= discount
                if not is_end:
                    G += model.get_action_Q(state, action) * discount_factor
                S, A, R = samples[0]
                rmemory.record((S, A, G))
                del samples[0]
                if not samples:
                    break
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
    parser = argparse.ArgumentParser('train {} sgnsarse model'.format(state.get_name()))
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
