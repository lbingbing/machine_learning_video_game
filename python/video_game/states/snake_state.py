import random
import copy

import numpy as np

BACKGROUND = 0
SNAKE_HEAD = 1
SNAKE_BODY = 2
TARGET     = -1

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_DIM = 4

HISTORY_LENGTH = 1

class SnakeState:
    def __init__(self, canvas_shape):
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.max_no_eat_action_num = self.height * self.width * 2
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        self.snake_positions = [((self.height - 1) // 2, self.width // 2), ((self.height - 1) // 2, self.width // 2 - 1)]
        self.background_positions = [(i, j) for i in range(self.height) for j in range(self.width) if (i, j) not in self.snake_positions]
        self.target_position = None
        self.direction = RIGHT
        self.action_done = False
        self.no_eat_action_num = 0
        self.end = False
        self.last_score = 0
        self.score = 0
        self.age = 0

        self.generate_target()
        self.render()

        self.canvas_history = []
        for i in range(HISTORY_LENGTH):
            self.canvas_history.append(copy.deepcopy(self.canvas))

    def get_name(self):
        return 'snake_{}_{}'.format(self.height, self.width)

    def get_canvas_shape(self):
        return self.height, self.width

    def is_end(self):
        return self.end

    def get_last_score(self):
        return self.last_score

    def get_score(self):
        return self.score

    def get_age(self):
        return self.age

    def get_action_done(self):
        return self.action_done

    def get_legal_actions(self):
        if self.direction == UP:
            legal_actions = [UP, LEFT, RIGHT]
        elif self.direction == DOWN:
            legal_actions = [DOWN, LEFT, RIGHT]
        elif self.direction == LEFT:
            legal_actions = [UP, DOWN, LEFT]
        elif self.direction == RIGHT:
            legal_actions = [UP, DOWN, RIGHT]
        return legal_actions

    def do_action(self, action):
        assert not self.action_done
        self.action_done = True
        if action == UP and self.direction not in (UP, DOWN):
            self.direction = UP
        elif action == DOWN and self.direction not in (UP, DOWN):
            self.direction = DOWN
        elif action == LEFT and self.direction not in (LEFT, RIGHT):
            self.direction = LEFT
        elif action == RIGHT and self.direction not in (LEFT, RIGHT):
            self.direction = RIGHT

    def update(self):
        del self.canvas_history[0]
        self.canvas_history.append(copy.deepcopy(self.canvas))

        self.last_score = 0
        next_position = self.get_next_position()
        tail_position = self.snake_positions.pop(-1)
        if next_position[0] < 0 or next_position[0] >= self.height or next_position[1] < 0 or next_position[1] >= self.width or next_position in self.snake_positions:
            self.end = True
        else:
            if next_position == self.target_position:
                self.no_eat_action_num = 0
                self.last_score = 1
                self.score += 1
                self.snake_positions.append(tail_position)
                if self.background_positions:
                    self.generate_target()
                else:
                    self.target_position = None
                    self.end = True
            else:
                if next_position != tail_position:
                    self.background_positions.remove(next_position)
                    self.background_positions.append(tail_position)
                self.no_eat_action_num += 1
                if self.no_eat_action_num == self.max_no_eat_action_num:
                    self.end = True
            self.snake_positions.insert(0, next_position)
            if not self.end:
                self.action_done = False
            self.age += 1
            assert len(self.background_positions) + len(self.snake_positions) + int(self.target_position is not None) == self.height * self.width
        self.render()

    def get_next_position(self):
        next_y, next_x = self.snake_positions[0]
        if self.direction == UP:
            next_y -= 1
        elif self.direction == DOWN:
            next_y += 1
        elif self.direction == LEFT:
            next_x -= 1
        elif self.direction == RIGHT:
            next_x += 1
        else:
            assert False
        return next_y, next_x

    def generate_target(self):
        self.target_position = random.choice(self.background_positions)
        self.background_positions.remove(self.target_position)

    def render(self):
        self.render_background()
        self.render_snake()
        self.render_target()

    def render_background(self):
        self.canvas = [[BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_snake(self):
        self.canvas[self.snake_positions[0][0]][self.snake_positions[0][1]] = SNAKE_HEAD
        for position in self.snake_positions[1:]:
            self.canvas[position[0]][position[1]] = SNAKE_BODY

    def render_target(self):
        if self.target_position is not None:
            self.canvas[self.target_position[0]][self.target_position[1]] = TARGET

    def get_reward(self):
        if self.is_end():
            return -2
        elif self.get_last_score() > 0:
            return 1
        else:
            return 0

    def get_state_numpy_shape(self):
        return 1, HISTORY_LENGTH+1, self.height, self.width

    def to_state_numpy(self):
        state_numpy = np.array(self.canvas, dtype=np.float32).reshape(1, self.height, self.width)
        history_numpy = np.array(self.canvas_history, dtype=np.float32)
        return np.concatenate((history_numpy, state_numpy), axis=0).reshape(1, HISTORY_LENGTH+1, self.height, self.width)

    def get_action_dim(self):
        return ACTION_DIM

    def action_to_action_index(self, action):
        return action

    def action_index_to_action(self, action_index):
        return action_index

    def action_to_action_numpy(self, action):
        action_index = self.action_to_action_index(action)
        action_numpy = np.zeros((self.get_action_dim(),), dtype=bool)
        action_numpy[action_index] = True
        return action_numpy.reshape(1, self.get_action_dim())

    def get_legal_action_mask_numpy(self):
        legal_action_mask_numpy = np.zeros((self.get_action_dim(),), dtype=bool)
        legal_action_indexes = [self.action_to_action_index(action) for action in self.get_legal_actions()]
        legal_action_mask_numpy[legal_action_indexes] = True
        return legal_action_mask_numpy.reshape(1, self.get_action_dim())

    def get_equivalent_num(self):
        return 4

    def get_equivalent_state_numpy(self, state_numpy):
        state_numpy_flip_y = np.flip(state_numpy, axis=2)
        state_numpy_flip_x = np.flip(state_numpy, axis=3)
        state_numpy_flip_yx = np.flip(state_numpy_flip_y, axis=3)
        return np.concatenate(
            [
            state_numpy,
            state_numpy_flip_y,
            state_numpy_flip_x,
            state_numpy_flip_yx,
            ],
            axis=0)

    def get_equivalent_action_numpy(self, action_numpy):
        action_numpy_flip_y = action_numpy.copy()
        action_numpy_flip_y[:,UP] = action_numpy[:,DOWN]
        action_numpy_flip_y[:,DOWN] = action_numpy[:,UP]
        action_numpy_flip_x = action_numpy.copy()
        action_numpy_flip_x[:,LEFT] = action_numpy[:,RIGHT]
        action_numpy_flip_x[:,RIGHT] = action_numpy[:,LEFT]
        action_numpy_flip_yx = action_numpy_flip_y.copy()
        action_numpy_flip_yx[:,LEFT] = action_numpy_flip_y[:,RIGHT]
        action_numpy_flip_yx[:,RIGHT] = action_numpy_flip_y[:,LEFT]
        return np.concatenate(
            [
            action_numpy,
            action_numpy_flip_y,
            action_numpy_flip_x,
            action_numpy_flip_yx,
            ],
            axis=0).reshape(action_numpy.shape[0]*self.get_equivalent_num(), self.get_action_dim())

def create_state():
    return SnakeState((5, 5))
