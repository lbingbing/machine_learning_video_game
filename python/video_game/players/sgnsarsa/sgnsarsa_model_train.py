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
        'step_num': 4,
        'dynamic_epsilon': 0.1,
        'discount': 0.99,
        'replay_memory_size': 4096,
        'batch_num_per_iteration': 1,
        'batch_size': 32,
        'dynamic_learning_rate': 0.001,
        }

def select_action(state, model, epsilon):
    if random.random() > epsilon:
        action = model.get_action(state)
    else:
        legal_actions = state.get_legal_actions()
        action = random.choice(legal_actions)
    return action

def sample(state, model, smemory, rmemory, configs, monitor):
    model.set_training(False)

    episode_num_per_iteration = configs['episode_num_per_iteration']
    exploring_starts = configs['exploring_starts']
    step_num = configs['step_num']
    dynamic_epsilon = configs['dynamic_epsilon']
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
        epsilon = train_utils.get_dynamic_epsilon(state.get_age(), dynamic_epsilon)
        action = select_action(state, model, epsilon)
        is_end = False
        for t in itertools.count(state.get_age()):
            if not is_end:
                state1 = state.clone()
                smemory.record(state1)
                state.do_action(action)
                state.update()
                R = state.get_reward()
                samples.append((state1, action, R))
                if monitor:
                    monitor.send_state(state.clone())
                if state.is_end():
                    is_end = True
                    if not is_exploring_starts:
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
    parser = argparse.ArgumentParser('train {} sgnsarse model'.format(state.get_name()))
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
        max_Q = model.get_max_Q(state)
        avg_loss = sum(losses) / len(losses)
        avg_score = sum(scores) / len(scores)
        avg_age = sum(ages) / len(ages)
        train_utils.log('{} iter: {} loss: {:.2f} max_Q: {:.2f} score: {:.2f} age: {:.2f}'.format(train_utils.get_current_time_str(), iteration_id, avg_loss, max_Q, avg_score, avg_age))
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
