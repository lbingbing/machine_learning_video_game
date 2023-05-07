import random
import copy

import numpy as np

BACKGROUND = 0
PACKMAN    = 1
BEAN       = 2
WALL       = -1
ENEMY      = -2

NOP   = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4
ACTION_DIM = 5

HISTORY_LENGTH = 1

class PackManState:
    def __init__(self, canvas_shape):
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.wall_num = 0
        self.bean_num = max(round(self.height * self.width / 9), 1)
        self.enemy_num = max(round(self.height * self.width / 25), 1)
        self.max_age = self.height * self.width * 2
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        blanks = [(i, j) for i in range(self.height) for j in range(self.width)]
        random.shuffle(blanks)
        self.wall_positions = blanks[:self.wall_num]
        blanks = blanks[self.wall_num:]
        self.bean_positions = blanks[:self.bean_num]
        blanks = blanks[self.bean_num:]
        self.enemy_positions = [list(e) for e in blanks[:self.enemy_num]]
        blanks = blanks[self.enemy_num:]
        self.packman_position = list(random.choice(blanks))
        self.action = None
        self.action_done = False
        self.end = False
        self.win = False
        self.last_score = 0
        self.score = 0
        self.age = 0

        self.render()

        self.canvas_history = []
        for i in range(HISTORY_LENGTH):
            self.canvas_history.append(copy.deepcopy(self.canvas))

    def get_name(self):
        return 'packman_{}_{}'.format(self.height, self.width)

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
        legal_actions = [NOP]
        if self.can_move_up(self.packman_position):
            legal_actions.append(UP)
        if self.can_move_down(self.packman_position):
            legal_actions.append(DOWN)
        if self.can_move_left(self.packman_position):
            legal_actions.append(LEFT)
        if self.can_move_right(self.packman_position):
            legal_actions.append(RIGHT)
        return legal_actions

    def can_move_up(self, pos):
        return pos[0]-1 >= 0 and not self.is_wall(pos[0]-1, pos[1])

    def can_move_down(self, pos):
        return pos[0]+1 < self.height and not self.is_wall(pos[0]+1, pos[1])

    def can_move_left(self, pos):
        return pos[1]-1 >= 0 and not self.is_wall(pos[0], pos[1]-1)

    def can_move_right(self, pos):
        return pos[1]+1 < self.width and not self.is_wall(pos[0], pos[1]+1)

    def is_wall(self, y, x):
        return (y, x) in self.wall_positions

    def do_action(self, action):
        assert not self.action_done
        self.action = action
        self.action_done = True

    def update(self):
        self.last_score = 0
        packman_position_trace = [self.packman_position[:]]
        if self.action == UP:
            assert self.can_move_up(self.packman_position)
            self.move_up(self.packman_position)
        elif self.action == DOWN:
            assert self.can_move_down(self.packman_position)
            self.move_down(self.packman_position)
        elif self.action == LEFT:
            assert self.can_move_left(self.packman_position)
            self.move_left(self.packman_position)
        elif self.action == RIGHT:
            assert self.can_move_right(self.packman_position)
            self.move_right(self.packman_position)
        self.action = None
        packman_position_trace.append(self.packman_position[:])
        for pos in self.enemy_positions:
            enemy_position_trace = [pos[:]]
            enemy_legal_actions = [NOP]
            if self.can_move_up(pos):
                enemy_legal_actions.append(UP)
            if self.can_move_down(pos):
                enemy_legal_actions.append(DOWN)
            if self.can_move_left(pos):
                enemy_legal_actions.append(LEFT)
            if self.can_move_right(pos):
                enemy_legal_actions.append(RIGHT)
            enemy_action = random.choice(enemy_legal_actions)
            if enemy_action == UP:
                self.move_up(pos)
            elif enemy_action == DOWN:
                self.move_down(pos)
            elif enemy_action == LEFT:
                self.move_left(pos)
            elif enemy_action == RIGHT:
                self.move_right(pos)
            enemy_position_trace.append(pos[:])
            if packman_position_trace[1] == enemy_position_trace[1] or \
               (packman_position_trace[0] == enemy_position_trace[1] and packman_position_trace[1] == enemy_position_trace[0]):
                self.end = True
        if tuple(self.packman_position) in self.bean_positions:
            self.bean_positions.remove(tuple(self.packman_position))
            self.last_score = 1
            self.score += 1
            if not self.bean_positions:
                self.end = True
                self.win = True
        self.age += 1
        if self.age == self.max_age:
            self.end = True
        if not self.end:
            self.action_done = False
        self.render()

    def move_up(self, pos):
        pos[0] -= 1

    def move_down(self, pos):
        pos[0] += 1

    def move_left(self, pos):
        pos[1] -= 1

    def move_right(self, pos):
        pos[1] += 1

    def render(self):
        self.render_background()
        self.render_walls()
        self.render_beans()
        self.render_enemies()
        self.render_packman()

    def render_background(self):
        self.canvas = [[BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_walls(self):
        for y, x in self.wall_positions:
            self.canvas[y][x] = WALL

    def render_beans(self):
        for y, x in self.bean_positions:
            self.canvas[y][x] = BEAN

    def render_enemies(self):
        for y, x in self.enemy_positions:
            self.canvas[y][x] = ENEMY

    def render_packman(self):
        self.canvas[self.packman_position[0]][self.packman_position[1]] = PACKMAN

    def get_reward(self):
        if self.is_end():
            if self.win:
                return 0
            else:
                return -self.bean_num
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
    return PackManState((10, 10))
