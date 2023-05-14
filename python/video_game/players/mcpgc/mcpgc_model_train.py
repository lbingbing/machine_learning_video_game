import argparse
import itertools

from ..utils import train_utils
from ..utils import replay_memory
from ..utils import train_flags

def sample(state, model, rmemory, configs):
    episode_num_per_iteration = configs['episode_num_per_iteration']
    discount = configs['discount']

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        samples = []
        state.reset()
        while not state.is_end():
            state1 = state.clone()
            action = model.get_action(state)
            state.do_action(action)
            state.update()
            R = state.get_reward()
            samples.append((state1, action, R))
        scores.append(state.get_score())
        ages.append(state.get_age())
        Gs = []
        Fs = []
        G = 0
        F = 1
        for S, A, R in reversed(samples):
            G = G * discount + R
            Gs.append(G)
            Fs.append(F)
            F *= discount
        for (S, A, R), G, F in zip(samples, Gs, Fs):
            rmemory.record((S, A, F*G))
    return scores, ages

def train(model, rmemory, configs, iteration_id):
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
    parser = argparse.ArgumentParser('train {} mcpgc model'.format(state.get_name()))
    args = parser.parse_args()

    if not train_flags.check_and_update_train_configs(model.get_model_path(), configs):
        print('train configs:')
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
    model.set_training(True)

    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    losses = []
    scores = []
    ages = []
    for iteration_id in itertools.count(1):
        scores1, ages1 = sample(state, model, rmemory, configs)
        losses1 = train(model, rmemory, configs, iteration_id)
        losses += losses1
        scores += scores1
        ages += ages1
        need_check = iteration_id % check_interval == 0
        if need_check:
            state.reset()
            P_logit_range = model.get_P_logit_range(state)
            P_range = model.get_P_range(state)
            avg_loss = sum(losses) / len(losses)
            avg_score = sum(scores) / len(scores)
            avg_age = sum(ages) / len(ages)
            print('{} iteration: {} avg_loss: {:.8f} P_logit_range: [{:.4f}, {:.4f}] P_range: [{:.4f}, {:.4f}] avg_score: {:.2f} avg_age: {:.2f}'.format(train_utils.get_current_time_str(), iteration_id, avg_loss, *P_logit_range, *P_range, avg_score, avg_age))
            losses.clear()
            scores.clear()
            ages.clear()
            train_flags.check_and_update_train_configs(model.get_model_path(), configs)
        if iteration_id % save_model_interval == 0 or (need_check and train_flags.check_and_clear_save_model_flag_file(model.get_model_path())):
            model.save()
            print('model {} saved'.format(model.get_model_path()))
        if need_check and train_flags.check_and_clear_stop_train_flag_file(model.get_model_path()):
            print('stopped')
            break
