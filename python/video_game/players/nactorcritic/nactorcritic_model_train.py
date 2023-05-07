import argparse
import itertools

from ..utils import replay_memory
from ..model import train_flags

def sample(state, model, rmemory, configs):
    episode_num_per_iteration = configs['episode_num_per_iteration']
    step_num = configs['step_num']
    discount = configs['discount']

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        samples = []
        state.reset()
        is_end = False
        F = 1
        for t in itertools.count():
            if not is_end:
                state1 = state.clone()
                action = model.get_action(state)
                V = model.get_V(state)
                state.do_action(action)
                state.update()
                R = state.get_reward()
                samples.append((state1, action, R, V))
                if state.is_end():
                    is_end = True
                    scores.append(state.get_score())
                    ages.append(state.get_age())
            if len(samples) >= step_num or is_end:
                G = 0
                discount_factor = 1
                for S, A, R, V in samples:
                    G += R * discount_factor
                    discount_factor *= discount
                if not is_end:
                    G += model.get_V(state) * discount_factor
                S, A, R, V = samples[0]
                D = G - V
                rmemory.record((S, A, G, F*D))
                F *= discount
                del samples[0]
                if not samples:
                    break
    return scores, ages

def train(model, rmemory, configs):
    batch_num_per_iteration = configs['batch_num_per_iteration']
    batch_size = configs['batch_size']
    learning_rate = configs['learning_rate']
    vloss_factor = configs['vloss_factor']

    vlosses = []
    plosses = []
    for i in range(batch_num_per_iteration):
        batch = rmemory.sample(batch_size)
        vloss, ploss = model.train(batch, learning_rate, vloss_factor)
        vlosses.append(vloss)
        plosses.append(ploss)
    return vlosses, plosses

def main(state, model, configs):
    parser = argparse.ArgumentParser('train {} nactorcritic model'.format(state.get_name()))
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

    vlosses = []
    plosses = []
    scores = []
    ages = []
    for iteration_id in itertools.count(1):
        scores1, ages1 = sample(state, model, rmemory, configs)
        vlosses1, plosses1 = train(model, rmemory, configs)
        vlosses += vlosses1
        plosses += plosses1
        scores += scores1
        ages += ages1
        need_check = iteration_id % check_interval == 0
        if need_check:
            state.reset()
            V = model.get_V(state)
            P_logit_range = model.get_P_logit_range(state)
            P_range = model.get_P_range(state)
            avg_vloss = sum(vlosses) / len(vlosses)
            avg_ploss = sum(plosses) / len(plosses)
            avg_score = sum(scores) / len(scores)
            avg_age = sum(ages) / len(ages)
            vlosses.clear()
            plosses.clear()
            scores.clear()
            ages.clear()
            print('iteration: {} avg_vloss: {:.8f} avg_ploss: {:.8f} V: {:.8f} P_logit_range: [{:.4f}, {:.4f}] P_range: [{:.4f}, {:.4f}] avg_score: {:.2f} avg_age: {:.2f}'.format(iteration_id, avg_vloss, avg_ploss, V, *P_logit_range, *P_range, avg_score, avg_age))
            train_flags.check_and_update_train_configs(model.get_model_path(), configs)
        if iteration_id % save_model_interval == 0 or (need_check and train_flags.check_and_clear_save_model_flag_file(model.get_model_path())):
            model.save()
            print('model {} saved'.format(model.get_model_path()))
        if need_check and train_flags.check_and_clear_stop_train_flag_file(model.get_model_path()):
            print('stopped')
            break
