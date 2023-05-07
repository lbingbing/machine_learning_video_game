def get_dynamic_epsilon(t, dynamic_epsilon):
    epsilon_low, epsilon_high, epsilon_high_t = dynamic_epsilon
    return (epsilon_high - epsilon_low) * t / epsilon_high_t + epsilon_low if t <= epsilon_high_t else epsilon_high
