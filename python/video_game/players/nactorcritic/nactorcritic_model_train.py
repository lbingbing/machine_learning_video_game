import argparse
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
        'step_num': 4,
        'discount': 0.99,
        'replay_memory_size': 4096,
        'batch_num_per_iteration': 1,
        'batch_size': 32,
        'dynamic_learning_rate': 0.001,
        'vloss_factor': 1,
        }

def sample(state, model, smemory, rmemory, configs, monitor):
    model.set_training(False)

    episode_num_per_iteration = configs['episode_num_per_iteration']
    exploring_starts = configs['exploring_starts']
    step_num = configs['step_num']
    discount = configs['discount']

    smemory.resize(configs['state_memory_size'])
    rmemory.resize(configs['replay_memory_size'])

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        samples = []
        is_exploring_starts, start_state = train_utils.get_exploring_starts(exploring_starts, smemory)
        if is_exploring_starts:
            state = start_state
        else:
            state.reset()
        if monitor:
            monitor.send_state(state.clone())
        is_end = False
        F = 1
        for t in itertools.count(state.get_age()):
            if not is_end:
                state1 = state.clone()
                smemory.record(state1)
                action = model.get_action(state)
                V = model.get_V(state)
                state.do_action(action)
                state.update()
                R = state.get_reward()
                samples.append((state1, action, R, V))
                if monitor:
                    monitor.send_state(state.clone())
                if state.is_end():
                    is_end = True
                    if not is_exploring_starts:
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
                p_factor = F * D
                P = [0] * state.get_action_dim()
                P[state.action_to_action_index(A)] = 1
                if p_factor < 0:
                    P = [1 - e for e in P]
                rmemory.record((S, P, abs(p_factor), G))
                F *= discount
                del samples[0]
                if not samples:
                    break
    return scores, ages

def train(model, rmemory, configs, iteration_id):
    model.set_training(True)

    batch_num_per_iteration = configs['batch_num_per_iteration']
    batch_size = configs['batch_size']
    learning_rate = train_utils.get_dynamic_learning_rate(iteration_id, configs['dynamic_learning_rate'])
    vloss_factor = configs['vloss_factor']

    plosses = []
    vlosses = []
    for i in range(batch_num_per_iteration):
        batch = rmemory.sample(batch_size)
        ploss, vloss = model.train(batch, learning_rate, vloss_factor)
        plosses.append(ploss)
        vlosses.append(vloss)
    return plosses, vlosses

def main(state, model, configs):
    parser = argparse.ArgumentParser('train {} nactorcritic model'.format(state.get_name()))
    train_utils.add_train_arguments(parser)
    args = parser.parse_args()

    train_utils.init_training(model, args.device)

    training_context = train_utils.create_training_context(model.get_model_dir_path(), configs)
    start_iteration_id = training_context['done_iteration_num'] + 1
    configs = training_context['configs']

    smemory = state_memory.StateMemory(configs['state_memory_size'])
    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    plosses = []
    vlosses = []
    scores = []
    ages = []

    def check_fn(iteration_id):
        state.reset()
        model.set_training(False)
        V = model.get_V(state)
        legal_P_logit_range = model.get_legal_P_logit_range(state)
        legal_P_range = model.get_legal_P_range(state)
        avg_ploss = sum(plosses) / len(plosses)
        avg_vloss = sum(vlosses) / len(vlosses)
        avg_score = sum(scores) / len(scores)
        avg_age = sum(ages) / len(ages)
        train_utils.log('{} iter: {} ploss: {:.2f} vloss: {:.2f} V: {:.2f} P_logit_range: [{:.2f}, {:.2f}] P_range: [{:.2f}, {:.2f}] score: {:.2f} age: {:.2f}'.format(train_utils.get_current_time_str(), iteration_id, avg_ploss, avg_vloss, V, *legal_P_logit_range, *legal_P_range, avg_score, avg_age))
        plosses.clear()
        vlosses.clear()
        scores.clear()
        ages.clear()

    with train_monitor.create_training_monitor(args.monitor_port) as monitor:
        for iteration_id in itertools.count(start_iteration_id):
            train_utils.pre_iteration(iteration_id, args.check_interval, model, configs)
            scores1, ages1 = sample(state, model, smemory, rmemory, configs, monitor)
            plosses1, vlosses1 = train(model, rmemory, configs, iteration_id)
            plosses += plosses1
            vlosses += vlosses1
            scores += scores1
            ages += ages1
            stop = train_utils.post_iteration(iteration_id, args.iteration_num, args.check_interval, args.save_model_interval, args.checkpoint_interval, model, training_context, check_fn)
            if stop:
                break
