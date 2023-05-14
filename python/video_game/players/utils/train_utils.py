import time

def get_dynamic_learning_rate(iteration_id, dynamic_learning_rate):
    start_learning_rate, end_learning_rate, transition_iteration = dynamic_learning_rate
    return (end_learning_rate - start_learning_rate) * iteration_id / transition_iteration + start_learning_rate if iteration_id <= transition_iteration else end_learning_rate

def get_dynamic_epsilon(t, dynamic_epsilon):
    start_epsilon, end_epsilon, transition_t = dynamic_epsilon
    return (end_epsilon - start_epsilon) * t / transition_t + start_epsilon if t <= transition_t else end_epsilon

def get_current_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')
