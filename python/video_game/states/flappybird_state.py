import random
import copy

import numpy as np

BIRD_HEIGHT = 2
BIRD_WIDTH  = 2

BACKGROUND = 0
BIRD       = 1
WALL       = -1
TARGET     = 2

NOP = 0
FLY = 1
ACTION_DIM = 2

GRAVITY = 1
FLY_SPEED = -2

HISTORY_LENGTH = 2

class FlappyBirdState:
    def __init__(self, canvas_shape):
        self.height = canvas_shape[0]
        self.width = canvas_shape[1]
        self.window_height_low = round(self.height / 4)
        self.window_height_high = round(self.height / 2)
        self.wall_width_low = round(self.width / 7)
        self.wall_width_high = round(self.width / 5)
        self.space_width_low = round(self.width / 3)
        self.space_width_high = round(self.width * 2 / 3)
        self.max_no_eat_action_num = self.width * 10
        self.reset()

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        self.bird_y = self.height // 2 - 1
        self.bird_x = 2
        self.bird_speed = 0
        self.walls = []
        self.targets = []
        self.generate_wall_space_target_counter = 0
        self.action_done = False
        self.no_eat_action_num = 0
        self.end = False
        self.last_score = 0
        self.score = 0
        self.age = 0

        self.render()

        self.canvas_history = []
        for i in range(HISTORY_LENGTH):
            self.canvas_history.append(copy.deepcopy(self.canvas))

    def get_name(self):
        return 'flappybird_{}_{}'.format(self.height, self.width)

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
        return [NOP, FLY]

    def do_action(self, action):
        assert not self.action_done
        self.action_done = True
        if action == FLY:
            self.bird_speed = FLY_SPEED

    def update(self):
        del self.canvas_history[0]
        self.canvas_history.append(copy.deepcopy(self.canvas))

        self.update_bird()
        self.update_walls()
        self.update_targets()
        self.update_bird_speed()
        if self.is_bird_hit_boundary() or self.is_bird_hit_wall():
            self.end = True
        else:
            self.handle_hit_target()
            self.remove_walls()
            self.remove_targets()
            self.generate_wall_space_target()
            if self.no_eat_action_num == self.max_no_eat_action_num:
                self.end = True
            if not self.end:
                self.action_done = False
            self.age += 1
        self.render()

    def update_bird(self):
        self.bird_y += self.bird_speed

    def update_walls(self):
        for wall in self.walls:
            wall[1] -= 1
            wall[3] -= 1

    def update_targets(self):
        for target in self.targets:
            target[1] -= 1

    def update_bird_speed(self):
        self.bird_speed += GRAVITY

    def is_bird_hit_boundary(self):
        return self.bird_y < 0 or self.bird_y+BIRD_HEIGHT > self.height

    def is_bird_hit_wall(self):
        for wall_y0, wall_x0, wall_y1, wall_x1 in self.walls:
            if wall_y0 < self.bird_y+BIRD_HEIGHT and self.bird_y < wall_y1 and \
               wall_x0 < self.bird_x+BIRD_WIDTH and self.bird_x < wall_x1:
                return True
        return False

    def handle_hit_target(self):
        self.last_score = 0
        target_to_delete = None
        for target in self.targets:
            if self.is_bird_hit_target(target):
                target_to_delete = target
                break
        if target_to_delete is not None:
            self.no_eat_action_num = 0
            self.last_score = 1
            self.score += 1
            self.targets.remove(target_to_delete)

    def is_bird_hit_target(self, target):
        return target[0] >= self.bird_y and target[0] < self.bird_y + BIRD_HEIGHT and \
               target[1] >= self.bird_x and target[1] < self.bird_x + BIRD_WIDTH

    def remove_walls(self):
        while True:
            if self.walls and self.walls[0][3] < 0:
                del self.walls[0]
            else:
                break

    def remove_targets(self):
        while True:
            if self.targets and self.targets[0][1] < 0:
                del self.targets[0]
            else:
                break

    def generate_wall_space_target(self):
        if self.generate_wall_space_target_counter > 0:
            self.generate_wall_space_target_counter -= 1
        else:
            wall_width = random.randint(self.wall_width_low, self.wall_width_high)
            space_width = random.randint(self.space_width_low, self.space_width_high)
            self.generate_wall_space_target_counter = wall_width + space_width

            window_height = random.randint(self.window_height_low, self.window_height_high)
            wall_y1 = random.randint(0, self.height - window_height)
            wall_y2 = wall_y1 + window_height
            self.walls.append([0, self.width, wall_y1, self.width+wall_width])
            self.walls.append([wall_y2, self.width, self.height, self.width+wall_width])

            target_y = random.randint(0, self.height-1)
            target_x = self.width + random.randint(wall_width, self.generate_wall_space_target_counter-1)
            self.targets.append([target_y, target_x])

    def render(self):
        self.render_background()
        self.render_bird()
        self.render_walls()
        self.render_targets()

    def render_background(self):
        self.canvas = [[BACKGROUND for j in range(self.width)] for i in range(self.height)]

    def render_bird(self):
        for i in range(self.bird_y, self.bird_y+BIRD_HEIGHT):
            if i >= 0 and i < self.height:
                for j in range(self.bird_x, self.bird_x+BIRD_WIDTH):
                    if j >= 0 and j < self.width:
                        self.canvas[i][j] = BIRD

    def render_walls(self):
        for wall_y0, wall_x0, wall_y1, wall_x1 in self.walls:
            for i in range(wall_y0, wall_y1):
                if i >= 0 and i < self.height:
                    for j in range(wall_x0, wall_x1):
                        if j >= 0 and j < self.width:
                            self.canvas[i][j] = WALL

    def render_targets(self):
        for target_y, target_x in self.targets:
            if target_y >= 0 and target_y < self.height and \
               target_x >= 0 and target_x < self.width:
                self.canvas[target_y][target_x] = TARGET

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
        return action_numpy.reshape(1, -1)

    def get_legal_action_mask_numpy(self):
        legal_action_mask_numpy = np.zeros((self.get_action_dim(),), dtype=bool)
        legal_action_indexes = [self.action_to_action_index(action) for action in self.get_legal_actions()]
        legal_action_mask_numpy[legal_action_indexes] = True
        return legal_action_mask_numpy.reshape(1, -1)

    def get_equivalent_num(self):
        return 1

    def get_equivalent_state_numpy(self, state_numpy):
        return state_numpy

    def get_equivalent_action_numpy(self, action_numpy):
        return action_numpy

def create_state():
    return FlappyBirdState((20, 20))
