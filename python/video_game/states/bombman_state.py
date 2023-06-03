import random
import copy

import numpy as np

BOMB_TRIGGER_TIME = 5

BACKGROUND        = 0
BOMBMAN           = 1
BOMB_BASE         = BOMBMAN + 1
BOMBMAN_BOMB_BASE = BOMB_BASE + BOMB_TRIGGER_TIME
STEEL             = -1
WALL              = -2
GATE              = -3

NOP   = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4
PLANT_BOMB = 5
ACTION_DIM = 6

HISTORY_LENGTH = 1

class BombManState:
    def __init__(self, canvas_shape):
        assert canvas_shape[0] % 2 == 1
        assert canvas_shape[1] % 2 == 1
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.wall_num_low = max(round(self.height * self.width / 18), 1)
        self.wall_num_high = self.wall_num_low * 2
        self.max_age = self.height * self.width * 2
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        blanks = [(i, j) for i in range(self.height) for j in range(self.width) if i % 2 == 0 or j % 2 == 0]
        random.shuffle(blanks)
        wall_num = random.randint(self.wall_num_low, self.wall_num_high)
        self.wall_positions = blanks[:wall_num]
        self.gate_position = random.choice(self.wall_positions)
        self.bombman_position = list(random.choice(blanks[wall_num:]))
        self.bombs = {}
        self.bomb_radius = 1
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
        return 'bombman_{}_{}'.format(self.height, self.width)

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
        if self.can_move_up():
            legal_actions.append(UP)
        if self.can_move_down():
            legal_actions.append(DOWN)
        if self.can_move_left():
            legal_actions.append(LEFT)
        if self.can_move_right():
            legal_actions.append(RIGHT)
        if self.can_plant_bomb():
            legal_actions.append(PLANT_BOMB)
        return legal_actions

    def can_move_up(self):
        y, x = self.bombman_position
        return y-1 >= 0 and not self.is_steel(y-1, x) and not self.is_wall(y-1, x) and not self.is_bomb(y-1, x)

    def can_move_down(self):
        y, x = self.bombman_position
        return y+1 < self.height and not self.is_steel(y+1, x) and not self.is_wall(y+1, x) and not self.is_bomb(y+1, x)

    def can_move_left(self):
        y, x = self.bombman_position
        return x-1 >= 0 and not self.is_steel(y, x-1) and not self.is_wall(y, x-1) and not self.is_bomb(y, x-1)

    def can_move_right(self):
        y, x = self.bombman_position
        return x+1 < self.width and not self.is_steel(y, x+1) and not self.is_wall(y, x+1) and not self.is_bomb(y, x+1)

    def can_plant_bomb(self):
        y, x = self.bombman_position
        return not self.is_bomb(y, x)

    def is_bombman(self, y, x):
        return (y, x) == tuple(self.bombman_position)

    def is_steel(self, y, x):
        return y % 2 == 1 and x % 2 == 1

    def is_wall(self, y, x):
        return (y, x) in self.wall_positions

    def is_bomb(self, y, x):
        return (y, x) in self.bombs

    def do_action(self, action):
        assert not self.action_done
        self.action = action
        self.action_done = True

    def update(self):
        self.last_score = 0
        if self.action == UP:
            assert self.can_move_up()
            self.move_up()
        elif self.action == DOWN:
            assert self.can_move_down()
            self.move_down()
        elif self.action == LEFT:
            assert self.can_move_left()
            self.move_left()
        elif self.action == RIGHT:
            assert self.can_move_right()
            self.move_right()
        elif self.action == PLANT_BOMB:
            assert self.can_plant_bomb()
            self.plant_bomb()
        self.action = None
        if tuple(self.bombman_position) == self.gate_position:
            self.end = True
            self.win = True
            self.last_score = 1
            self.score += 1
        else:
            bombman_killed, wall_destroyed = self.handle_bomb_explode()
            self.handle_bomb_timer()
            if bombman_killed:
                self.end = True
            elif wall_destroyed:
                self.last_score = 0.01
                self.score += 0.01
            self.age += 1
            if self.age == self.max_age:
                self.end = True
            if not self.end:
                self.action_done = False
        self.render()

    def move_up(self):
        self.bombman_position[0] -= 1

    def move_down(self):
        self.bombman_position[0] += 1

    def move_left(self):
        self.bombman_position[1] -= 1

    def move_right(self):
        self.bombman_position[1] += 1

    def plant_bomb(self):
        pos = tuple(self.bombman_position)
        assert pos not in self.bombs
        self.bombs[pos] = 0

    def handle_bomb_explode(self):
        bomb_positions = []
        for pos, timer in self.bombs.items():
            if timer == BOMB_TRIGGER_TIME:
                bomb_positions.append(pos)
        bomb_explode_context = {
            'bombman_killed': False,
            'wall_destroyed': False,
            'bomb_positions': bomb_positions,
            'checked_positions': set(),
            }
        self.bomb_explode(bomb_explode_context)
        return bomb_explode_context['bombman_killed'], bomb_explode_context['wall_destroyed']

    def bomb_explode(self, bomb_explode_context):
        while bomb_explode_context['bomb_positions']:
            pos = bomb_explode_context['bomb_positions'][0]
            y, x = pos
            if self.is_bombman(y, x):
                bomb_explode_context['checked_positions'].add(pos)
                bomb_explode_context['bombman_killed'] = True
            for i in range(1, self.bomb_radius+1):
                if y-i >= 0:
                    wall_destroyed = self.check_bomb_explode_position(bomb_explode_context, y-i, x)
                    if wall_destroyed:
                        break
                else:
                    break
            for i in range(1, self.bomb_radius+1):
                if y+i < self.height:
                    wall_destroyed = self.check_bomb_explode_position(bomb_explode_context, y+i, x)
                    if wall_destroyed:
                        break
                else:
                    break
            for i in range(1, self.bomb_radius+1):
                if x-i >= 0:
                    wall_destroyed = self.check_bomb_explode_position(bomb_explode_context, y, x-i)
                    if wall_destroyed:
                        break
                else:
                    break
            for i in range(1, self.bomb_radius+1):
                if x+i < self.width:
                    wall_destroyed = self.check_bomb_explode_position(bomb_explode_context, y, x+i)
                    if wall_destroyed:
                        break
                else:
                    break
            del self.bombs[pos]
            del bomb_explode_context['bomb_positions'][0]

    def check_bomb_explode_position(self, bomb_explode_context, y, x):
        wall_destroyed = False
        if (y, x) not in bomb_explode_context['checked_positions']:
            bomb_explode_context['checked_positions'].add((y, x))
            if self.is_bombman(y, x):
                bomb_explode_context['bombman_killed'] = True
            if self.is_wall(y, x):
                self.wall_positions.remove((y, x))
                bomb_explode_context['wall_destroyed'] = True
                wall_destroyed = True
            if self.is_bomb(y, x):
                bomb_explode_context['bomb_positions'].append((y, x))
        return wall_destroyed

    def handle_bomb_timer(self):
        for pos in self.bombs:
            self.bombs[pos] += 1

    def render(self):
        self.render_background()
        self.render_walls()
        self.render_gate()
        self.render_bombman_and_bombs()

    def render_background(self):
        self.canvas = [[STEEL if self.is_steel(i, j) else BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_walls(self):
        for y, x in self.wall_positions:
            self.canvas[y][x] = WALL

    def render_gate(self):
        if not self.is_wall(*self.gate_position):
            self.canvas[self.gate_position[0]][self.gate_position[1]] = GATE

    def render_bombman_and_bombs(self):
        for (y, x), timer in self.bombs.items():
            if not self.is_bombman(y, x):
                self.canvas[y][x] = BOMB_BASE + timer - 1
        if self.is_bomb(*self.bombman_position):
            self.canvas[self.bombman_position[0]][self.bombman_position[1]] = BOMBMAN_BOMB_BASE + timer - 1
        else:
            self.canvas[self.bombman_position[0]][self.bombman_position[1]] = BOMBMAN

    def get_reward(self):
        if self.is_end():
            if self.win:
                return 1
            else:
                return -1
        elif self.get_last_score() > 0:
            return 1 / self.max_age
        else:
            return -1 / self.max_age

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

    def get_legal_action_indexes(self):
        return [self.action_to_action_index(action) for action in self.get_legal_actions()]

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
    return BombManState((9, 9))
