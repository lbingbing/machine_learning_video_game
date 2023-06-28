import argparse
import itertools

from ..utils import train_utils
from ..utils import replay_memory
from ..utils import train_flags
from ..utils import train_monitor

def sample(state, model, rmemory, configs, monitor):
    model.set_training(False)

    episode_num_per_iteration = configs['episode_num_per_iteration']
    step_num = configs['step_num']
    discount = configs['discount']

    rmemory.resize(configs['replay_memory_size'])

    scores = []
    ages = []
    for episode_id in range(episode_num_per_iteration):
        samples = []
        state.reset()
        if monitor:
            monitor.send_state(state.clone())
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
                if monitor:
                    monitor.send_state(state.clone())
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

    train_utils.init_model_log(model.get_model_path())

    train_utils.init_model(model, args.device)

    training_context = train_utils.create_training_context(model.get_model_path(), configs)
    start_iteration_id = training_context['start_iteration_id']
    configs = training_context['configs']

    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    plosses = []
    vlosses = []
    scores = []
    ages = []
    with train_monitor.create_training_monitor(args.monitor_port) as monitor:
        for iteration_id in itertools.count(1):
            if iteration_id % args.check_interval == 1:
                train_flags.check_and_update_train_configs(model.get_model_path(), configs)
            scores1, ages1 = sample(state, model, rmemory, configs, monitor)
            plosses1, vlosses1 = train(model, rmemory, configs, iteration_id)
            plosses += plosses1
            vlosses += vlosses1
            scores += scores1
            ages += ages1
            need_check = iteration_id % args.check_interval == 0
            if need_check:
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
