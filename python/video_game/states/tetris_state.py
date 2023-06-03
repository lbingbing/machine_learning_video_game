import random
import copy
import itertools

import numpy as np

BACKGROUND   = 0
LANDED_UNIT  = 1
FALLING_UNIT = -1

NORMAL_TYPE  = 0
DOT_TYPE     = 1
NEG_GUN_TYPE = 3
POS_GUN_TYPE = 2
BOMB_TYPE    = 4

BRICK_TYPES = (
    ( NORMAL_TYPE, 1, ((0, 0), (1, 0), (2, 0), (3, 0))),
    ( NORMAL_TYPE, 1, ((0, 0), (0, 1), (1, 0), (1, 1))),
    ( NORMAL_TYPE, 1, ((0, 0), (0, 1), (0, 2), (1, 1))),
    ( NORMAL_TYPE, 1, ((0, 0), (0, 1), (1, 1), (1, 2))),
    ( NORMAL_TYPE, 1, ((0, 0), (0, 1), (0, 2), (1, 0))),
    ( NORMAL_TYPE, 1, ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
    ( NORMAL_TYPE, 1, ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2))),
    (    DOT_TYPE, 1, ((0, 0), )),
    (NEG_GUN_TYPE, 1, ((0, 0), (1, 0))),
    (POS_GUN_TYPE, 1, ((0, 0), (1, 0), (2, 0))),
    (   BOMB_TYPE, 1, ((0, 0), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2))),
    )

BRICK_CUMULATIVE_WEIGHTS = list(itertools.accumulate(w for t, w, coords in BRICK_TYPES))

NOP    = 0
LEFT   = 1
RIGHT  = 2
ROTATE = 3
FIRE   = 4
FALL   = 5
LAND   = 6
ACTION_DIM = 7

class TetrisState:
    def __init__(self, canvas_shape):
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        self.landed_map = [[False for j in range(self.width)] for i in range(self.height)]
        self.action = None
        self.action_done = False
        self.end = False
        self.last_score = 0
        self.score = 0
        self.age = 0

        self.generate_falling_brick()
        self.render()

    def get_name(self):
        return 'tetris_{}_{}'.format(self.height, self.width)

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
        if self.can_move_left():
            legal_actions.append(LEFT)
        if self.can_move_right():
            legal_actions.append(RIGHT)
        if self.falling_brick_type == NORMAL_TYPE and self.can_rotate():
            legal_actions.append(ROTATE)
        if self.falling_brick_type == NEG_GUN_TYPE or self.falling_brick_type == POS_GUN_TYPE:
            legal_actions.append(FIRE)
        if self.can_fall():
            legal_actions.append(FALL)
            legal_actions.append(LAND)
        return legal_actions

    def can_move_left(self):
        for y, x in self.falling_brick_positions:
            if x-1 < 0 or self.falling_brick_type != BOMB_TYPE and self.landed_map[y][x-1]:
                return False
        return True

    def can_move_right(self):
        for y, x in self.falling_brick_positions:
            if x+1 >= self.width or self.falling_brick_type != BOMB_TYPE and self.landed_map[y][x+1]:
                return False
        return True

    def can_rotate(self):
        for y, x in self.get_rotated_positions(self.falling_brick_positions):
            if self.landed_map[y][x]:
                return False
        return True

    def can_fall(self):
        if self.falling_brick_type == DOT_TYPE:
            y, x = self.falling_brick_positions[0]
            if any(not self.landed_map[y1][x] for y1 in range(y+1, self.height)):
                return True
            else:
                return False
        else:
            for y, x in self.falling_brick_positions:
                if y+1 >= self.height or self.landed_map[y+1][x]:
                    return False
            return True

    def do_action(self, action):
        assert not self.action_done
        self.action = action
        self.action_done = True

    def update(self):
        self.last_score = 0
        if self.action == LEFT:
            assert self.can_move_left()
            self.move_left()
        elif self.action == RIGHT:
            assert self.can_move_right()
            self.move_right()
        elif self.action == ROTATE:
            assert self.can_rotate()
            self.rotate()
        elif self.action == FIRE:
            assert self.falling_brick_type == NEG_GUN_TYPE or self.falling_brick_type == POS_GUN_TYPE
            self.fire()
        elif self.action == FALL:
            assert self.can_fall()
            self.fall()
        elif self.action == LAND:
            assert self.can_fall()
            self.land()
        self.action = None
        if self.can_fall():
            self.fall()
        else:
            self.land_brick()
            self.generate_falling_brick()
        if not self.end:
            self.action_done = False
        self.age += 1
        self.render()

    def move_left(self):
        for pos in self.falling_brick_positions:
            pos[1] -= 1

    def move_right(self):
        for pos in self.falling_brick_positions:
            pos[1] += 1

    def rotate(self):
        self.falling_brick_positions = self.get_rotated_positions(self.falling_brick_positions)

    def get_rotated_positions(self, positions):
        y_min = min(y for y, x in positions)
        y_max = max(y for y, x in positions)
        x_min = min(x for y, x in positions)
        x_max = max(x for y, x in positions)
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        rotated_positions = []
        for y, x in positions:
            rotated_y = center_y - (x - center_x)
            rotated_x = center_x + (y - center_y)
            rotated_positions.append([rotated_y, rotated_x])
        new_y_max = max(y for y, x in rotated_positions)
        for pos in rotated_positions:
            pos[0] += y_max - new_y_max
        new_y_min = min(y for y, x in rotated_positions)
        if new_y_min < 0:
            for pos in rotated_positions:
                pos[0] -= new_y_min
        new_y_max = max(y for y, x in rotated_positions)
        if new_y_max >= self.height:
            for pos in rotated_positions:
                pos[0] -= new_y_max - (self.height - 1)
        new_x_min = min(x for y, x in rotated_positions)
        if new_x_min < 0:
            for pos in rotated_positions:
                pos[1] -= new_x_min
        new_x_max = max(x for y, x in rotated_positions)
        if new_x_max >= self.width:
            for pos in rotated_positions:
                pos[1] -= new_x_max - (self.width - 1)
        return rotated_positions

    def fire(self):
        y, x = self.falling_brick_positions[-1]
        if self.falling_brick_type == NEG_GUN_TYPE:
            for y1 in range(y+1, self.height):
                if self.landed_map[y1][x]:
                    self.landed_map[y1][x] = False
                    break
        elif self.falling_brick_type == POS_GUN_TYPE:
            y1 = y
            while y1+1 < self.height and not self.landed_map[y1+1][x]:
                y1 += 1
            if y1 > y:
                self.landed_map[y1][x] = True
                self.check_full_row()

    def fall(self):
        for pos in self.falling_brick_positions:
            pos[0] += 1

    def land(self):
        while self.can_fall():
            self.fall()

    def land_brick(self):
        if self.falling_brick_type == NORMAL_TYPE or self.falling_brick_type == DOT_TYPE:
            for y, x in self.falling_brick_positions:
                self.landed_map[y][x] = True
            self.check_full_row()
        elif self.falling_brick_type == BOMB_TYPE:
            self.explode()

    def check_full_row(self):
        full_row_ids = [row_id for row_id, row in enumerate(self.landed_map) if all(row)]
        if full_row_ids:
            for row_id in reversed(full_row_ids):
                del self.landed_map[row_id]
            for i in range(len(full_row_ids)):
                self.landed_map.insert(0, [False for j in range(self.width)])
            self.last_score = len(full_row_ids) ** 2
            self.score += self.last_score

    def explode(self):
        y_min = min(y for y, x in self.falling_brick_positions)
        x_min = min(x for y, x in self.falling_brick_positions)
        for y in range(y_min, y_min+4):
            if y < self.height:
                for x in range(x_min, x_min+4):
                    if x < self.width:
                        self.landed_map[y][x] = False

    def generate_falling_brick(self):
        brick_type, weight, positions = random.choices(BRICK_TYPES, cum_weights=BRICK_CUMULATIVE_WEIGHTS, k=1)[0]
        self.falling_brick_type = brick_type
        w = max(x for y, x in positions) - min(x for y, x in positions) + 1
        offset = (self.width - w) // 2
        self.falling_brick_positions = [[y, offset+x] for y, x in positions]
        if self.falling_brick_type == NORMAL_TYPE:
            if random.random() < 0.5:
                self.flip_x()
            for i in range(random.randint(0, 3)):
                self.rotate()
            y_min = min(y for y, x in self.falling_brick_positions)
            for pos in self.falling_brick_positions:
                pos[0] -= y_min
        if any(self.landed_map[y][x] for y, x in self.falling_brick_positions):
            if self.falling_brick_type == NORMAL_TYPE or self.falling_brick_type == DOT_TYPE:
                self.end = True
            elif self.falling_brick_type == NEG_GUN_TYPE or self.falling_brick_type == POS_GUN_TYPE:
                self.generate_falling_brick()
            elif self.falling_brick_type == BOMB_TYPE:
                self.explode()
                self.generate_falling_brick()

    def flip_x(self):
        center_x = (max(x for y, x in self.falling_brick_positions) + min(x for y, x in self.falling_brick_positions)) // 2
        for pos in self.falling_brick_positions:
            pos[1] = center_x * 2 - pos[1]

    def render(self):
        self.render_background()
        self.render_landed_bricks()
        self.render_falling_bricks()

    def render_background(self):
        self.canvas = [[BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_landed_bricks(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.landed_map[y][x]:
                    self.canvas[y][x] = LANDED_UNIT

    def render_falling_bricks(self):
        for y, x in self.falling_brick_positions:
            self.canvas[y][x] = FALLING_UNIT

    def get_reward(self):
        if self.is_end():
            return -2
        elif self.get_last_score() > 0:
            return self.get_last_score()
        else:
            return -1 / (self.height * 10)

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
        return 2

    def get_equivalent_state_numpy(self, state_numpy):
        state_numpy_flip_x = np.flip(state_numpy, axis=3)
        return np.concatenate(
            [
            state_numpy,
            state_numpy_flip_x,
            ],
            axis=0)

    def get_equivalent_action_numpy(self, action_numpy):
        action_numpy_flip_x = action_numpy.copy()
        action_numpy_flip_x[:,LEFT] = action_numpy[:,RIGHT]
        action_numpy_flip_x[:,RIGHT] = action_numpy[:,LEFT]
        return np.concatenate(
            [
            action_numpy,
            action_numpy_flip_x,
            ],
            axis=0).reshape(action_numpy.shape[0]*self.get_equivalent_num(), self.get_action_dim())

def create_state():
    return TetrisState((20, 10))
