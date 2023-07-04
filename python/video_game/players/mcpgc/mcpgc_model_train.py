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
        while not state.is_end():
            state1 = state.clone()
            smemory.record(state1)
            action = model.get_action(state)
            state.do_action(action)
            state.update()
            R = state.get_reward()
            samples.append((state1, action, R))
            if monitor:
                monitor.send_state(state.clone())
        if not is_exploring_starts:
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
            p_factor = F * G
            P = [0] * state.get_action_dim()
            P[state.action_to_action_index(A)] = 1
            if p_factor < 0:
                P = [1 - e for e in P]
            rmemory.record((S, P, abs(p_factor)))
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
    parser = argparse.ArgumentParser('train {} mcpgc model'.format(state.get_name()))
    train_utils.add_train_arguments(parser)
    args = parser.parse_args()

    train_utils.init_training(model, args.device)

    training_context = train_utils.create_training_context(model.get_model_dir_path(), configs)
    start_iteration_id = training_context['done_iteration_num'] + 1
    configs = training_context['configs']

    smemory = state_memory.StateMemory(configs['state_memory_size'])
    rmemory = replay_memory.ReplayMemory(configs['replay_memory_size'])

    losses = []
    scores = []
    ages = []

    def check_fn(iteration_id):
        state.reset()
        model.set_training(False)
        legal_P_logit_range = model.get_legal_P_logit_range(state)
        legal_P_range = model.get_legal_P_range(state)
        avg_loss = sum(losses) / len(losses)
        avg_score = sum(scores) / len(scores)
        avg_age = sum(ages) / len(ages)
        train_utils.log('{} iter: {} loss: {:.2f} P_logit_range: [{:.2f}, {:.2f}] P_range: [{:.2f}, {:.2f}] score: {:.2f} age: {:.2f}'.format(train_utils.get_current_time_str(), iteration_id, avg_loss, *legal_P_logit_range, *legal_P_range, avg_score, avg_age))
        losses.clear()
        scores.clear()
        ages.clear()

    with train_monitor.create_training_monitor(args.monitor_port) as monitor:
        for iteration_id in itertools.count(start_iteration_id):
            train_utils.pre_iteration(iteration_id, args.check_interval, model, configs)
            scores1, ages1 = sample(state, model, smemory, rmemory, configs, monitor)
            losses1 = train(model, rmemory, configs, iteration_id)
            losses += losses1
            scores += scores1
            ages += ages1
            stop = train_utils.post_iteration(iteration_id, args.iteration_num, args.check_interval, args.save_model_interval, args.checkpoint_interval, model, training_context, check_fn)
            if stop:
                break
