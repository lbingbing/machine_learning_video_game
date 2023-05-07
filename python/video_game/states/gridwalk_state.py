import random
import copy

import numpy as np

BACKGROUND = 0
WALKER     = 1
TARGET     = -1

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_DIM = 4

class GridWalkState:
    def __init__(self, canvas_shape):
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.max_age = self.height * self.width * 2
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        self.walker_position = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
        self.target_position = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
        self.action = None
        self.action_done = False
        self.end = False
        self.last_score = 0
        self.score = 0
        self.age = 0

        self.render()

    def get_name(self):
        return 'gridwalk_{}_{}'.format(self.height, self.width)

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
        legal_actions = [UP, LEFT, DOWN, RIGHT]
        return legal_actions

    def do_action(self, action):
        assert not self.action_done
        self.action = action
        self.action_done = True

    def update(self):
        self.last_score = 0
        if self.action is not None:
            if self.action == UP:
                self.walker_position[0] -= 1
            elif self.action == DOWN:
                self.walker_position[0] += 1
            elif self.action == LEFT:
                self.walker_position[1] -= 1
            elif self.action == RIGHT:
                self.walker_position[1] += 1
            else:
                assert False
            self.action = None
        if self.walker_position[0] < 0 or self.walker_position[0] >= self.height or self.walker_position[1] < 0 or self.walker_position[1] >= self.width:
            self.end = True
        else:
            if self.walker_position == self.target_position:
                self.end = True
                self.last_score = 1
                self.score = 1
            self.age += 1
            if self.age == self.max_age:
                self.end = True
            if not self.end:
                self.action_done = False
        self.render()

    def render(self):
        self.render_background()
        self.render_target()
        self.render_walker()

    def render_background(self):
        self.canvas = [[BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_target(self):
        self.canvas[self.target_position[0]][self.target_position[1]] = TARGET

    def render_walker(self):
        if self.walker_position[0] >= 0 and self.walker_position[0] < self.height and self.walker_position[1] >= 0 and self.walker_position[1] < self.width:
            self.canvas[self.walker_position[0]][self.walker_position[1]] = WALKER

    def get_reward(self):
        if self.is_end():
            if self.get_last_score() > 0:
                return 1
            else:
                return -1
        else:
            return 0

    def get_state_dim(self):
        return (self.height * self.width) ** 2

    def to_state_index(self):
        return self.walker_position[0] + self.height * (self.walker_position[1] + self.width * (self.target_position[0] + self.height * self.target_position[1]))

    def get_state_numpy_shape(self):
        return 1, 1, self.height, self.width

    def to_state_numpy(self):
        return np.array(self.canvas, dtype=np.float32).reshape(1, 1, self.height, self.width)

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

    def get_legal_action_indexes(self):
        return [self.action_to_action_index(action) for action in self.get_legal_actions()]

    def get_legal_action_mask_numpy(self):
        legal_action_mask_numpy = np.zeros((self.get_action_dim(),), dtype=bool)
        legal_action_indexes = [self.action_to_action_index(action) for action in self.get_legal_actions()]
        legal_action_mask_numpy[legal_action_indexes] = True
        return legal_action_mask_numpy.reshape(1, self.get_action_dim())

    def get_equivalent_num(self):
        return 4

    def get_equivalent_state_indexes(self, state_index):
        return [state_index] * self.get_equivalent_num()

    def get_equivalent_action_indexes(self, action_index):
        return [action_index] * self.get_equivalent_num()

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
    return GridWalkState((5, 5))
