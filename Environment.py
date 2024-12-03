import torch
from Object import Object, decay_func
import math
import random
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def get_distance_between_locations(x1, y1, x2, y2, with_direction=False):
    """
    :param with_direction: if True, an additional list is returned that indicates the directions
    :return: number of diagonal and straight steps to go from [x1, y1] to [x2, y2], and directions
    """
    directions = [[0, 0], [0, 0]]
    dx = x2 - x1  # self._agent_location[0]
    dy = y2 - y1  # self._agent_location[1]
    # min_dis = min(abs(dx), abs(dy))
    min_dis = np.minimum(abs(dx), abs(dy))
    # max_dis = max(abs(dx), abs(dy))
    max_dis = np.maximum(abs(dx), abs(dy))
    diagonal_steps = min_dis
    straight_steps = max_dis - min_dis

    if with_direction:
        x_diff = x2 - x1
        y_diff = y2 - y1
        if diagonal_steps > 0:
            directions[0] = [np.sign(x_diff),
                             np.sign(y_diff)]
            x1 += diagonal_steps * np.sign(x_diff)
            y1 += diagonal_steps * np.sign(y_diff)
        if straight_steps > 0:
            x_diff = x2 - x1
            y_diff = y2 - y1
            directions[1] = [np.sign(x_diff),
                             np.sign(y_diff)]
        return diagonal_steps, straight_steps, directions

    return diagonal_steps, straight_steps


def get_pairwise_distance(P1, P2):
    distances = np.zeros((P1.shape[0], P2.shape[0]))
    for i in range(P1.shape[0]):
        n_diagonal, n_straight = get_distance_between_locations(P1[i, 0], P1[i, 1], P2[:, 0], P2[:, 1])
        row_i = math.sqrt(2) * n_diagonal + n_straight
        distances[i, :] = row_i
    return distances


class Environment(gym.Env):
    def __init__(self, params, few_many_objects, object_reappears=True):
        self.each_type_object_num = None
        self._agent_location = None
        self._object_pool = []
        self.params = params
        self.height = params.HEIGHT
        self.width = params.WIDTH
        self.few_many_objects = few_many_objects
        self.metadata = {"render_modes": None}
        self.object_type_num = params.OBJECT_TYPE_NUM
        self.object_reappears = object_reappears
        self._no_reward_threshold = params.NO_REWARD_THRESHOLD
        self._env_map = np.zeros((1 + self.object_type_num, self.height, self.width), dtype=int)
        self._env_map_dict = {}
        self._mental_states = np.empty((self.object_type_num,), dtype=np.float64)
        self._mental_states_slope = np.empty((self.object_type_num,), dtype=np.float64)
        self._object_coefficients = np.empty((self.object_type_num,
                                              self.object_type_num), dtype=int)
        self._object_rewards = None
        self._environment_states_parameters = [self._mental_states_slope, self._object_coefficients]
        self._environment_states_parameters_range = [self.params.MENTAL_STATES_SLOPE_RANGE,
                                                     self.params.ENVIRONMENT_OBJECT_COEFFICIENT_RANGE]
        self.each_type_invisible_object_num = np.zeros((self.object_type_num), dtype=int)
        self._object_coefficients_min = self.params.MIN_MAX_OBJECT_REWARD[0] / self.params.MIN_MAX_OBJECT_REWARD[1]
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 1, shape=(1 + self.object_type_num,
                                     self.height, self.width), dtype=int),  # 'env_map'
             spaces.Box(self.params.INITIAL_MENTAL_STATES_RANGE[0], 2 ** 63 - 2,
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states'
             spaces.Box(self.params.MENTAL_STATES_SLOPE_RANGE[0],
                        self.params.MENTAL_STATES_SLOPE_RANGE[1],
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states_slope'
             spaces.Box(self.params.ENVIRONMENT_OBJECT_COEFFICIENT_RANGE[0],
                        self.params.ENVIRONMENT_OBJECT_COEFFICIENT_RANGE[1],
                        shape=(self.object_type_num,), dtype=float))  # 'environment_object_reward'
        )
        self.action_space = spaces.Box(-2 ** 63, 2 ** 63 - 2,
                                       shape=(self.params.WIDTH * self.params.HEIGHT,), dtype=float)

    def sample(self):
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        self.each_type_object_num = self._init_object_num_on_map()
        self._init_objects()
        self._init_random_map()
        # self._object_rewards = np.zeros((self.object_type_num, sum(self.each_type_object_num)), dtype=np.float)
        self._init_random_mental_states()
        self._init_random_parameters()
        observation = self.get_observation()
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        sample_observation = self.sample()
        return sample_observation, dict()

    def step(self, goal_map, update_objects_status=False):
        # possible solution: Put everything here in a for loop, then at each step, update object probability
        # to see if anything appears/disappears. if an event is initiated (an object appeared/disappeared,
        # break the loop,
        assert goal_map.shape == self._env_map[0].shape, "invalid goal_map size"
        info = dict()  # look for this bug: env_map has a 1 in layers 1 or 2, that the corresponding object is not in the env_map_dict
        goal_location = np.argwhere(goal_map).flatten()  # location of 1 in map
        on_object = self._env_map[:, goal_location[0], goal_location[1]].sum() == 2  # agent on an object

        diagonal_steps, straight_steps = get_distance_between_locations(self._agent_location[0],
                                                                        self._agent_location[1],
                                                                        goal_location[0],
                                                                        goal_location[1])
        step_length = math.sqrt(2) * diagonal_steps + straight_steps
        self._update_agent_locations(goal_location)
        dt = np.array(1) if step_length < 1.4 else step_length
        reached_type = deepcopy(self._env_map[1:,
                                self._agent_location[0],
                                self._agent_location[1]])
        which_type = np.argwhere(reached_type)
        objects_reward = np.zeros((self.object_type_num,))
        if which_type.shape[0] > 0:  # Agent did not stay
            reached_object = self._env_map_dict[(which_type.item(), goal_location[0], goal_location[1])]
            objects_reward[np.argwhere(reached_type).item()] = reached_object.reward  # 2 x

        object_reward = self._object_coefficients @ objects_reward
        self._update_object_locations()
        mental_states_cost = self._get_mental_states_cost(diagonal_steps, straight_steps)
        self._update_mental_state_after_step(dt=dt)

        positive_mental_states_before_reward = self._positive_mental_states()
        self._update_mental_states_after_object(u=object_reward)
        positive_mental_states_after_reward = self._positive_mental_states()
        # mental_states_reward = np.maximum(0,
        #                                   positive_mental_states_before_reward - positive_mental_states_after_reward)
        time_mental_state_reward, fixed_mental_state_reward = self._get_mental_states_reward(
            positive_mental_states_before_reward,
            positive_mental_states_after_reward,
            reached_type)
        # reward = [sigma (slope) > 0] * mental_states_reward + [sigma (slope) == 0] * mental_states_reward
        # time_reward = np.array(time_mental_state_reward - step_length - mental_states_cost)
        # fixed_reward = np.array(fixed_mental_state_reward - step_length)
        reward = float(time_mental_state_reward + fixed_mental_state_reward - mental_states_cost - step_length)
        terminated = False
        truncated = False
        info['dt'] = dt
        info['rewarding'] = time_mental_state_reward > 0 or fixed_mental_state_reward > 0
        if (which_type.shape[0] == 0 or diagonal_steps + straight_steps == 0) and not on_object:
            info['object'] = False
        else:
            info['object'] = True

        info['environment_changed'] = False
        if update_objects_status:
            info['environment_changed'] = self._update_objects_status(object_event=info['object'])
        observation = self.get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def _get_positive_auc(self, n_0, time):
        x = np.zeros_like(self._mental_states_slope)
        for i in range(self._mental_states_slope.shape[0]):
            if self._mental_states_slope[i] > 0:
                x[i] = -n_0[i] / self._mental_states_slope[i]
        x = np.maximum(x, 0)
        y = np.maximum(self._mental_states_slope * time + n_0, 0) - np.maximum(n_0, 0)
        auc = np.zeros_like(x)
        for i in range(x.shape[0]):
            if x[i] < time and y[i] > 0:
                auc[i] = (time - x[i]) * y[i] / 2
        return auc

    def _get_mental_states_cost(self, diagonal_steps, straight_steps):
        is_time_varying = self._mental_states_slope > 0
        mental_states = self._mental_states.copy()
        if diagonal_steps == 0 and straight_steps == 0:  # stays
            # cost = (self._positive_mental_states() * is_time_varying).sum()
            base_carried_need = self._positive_mental_states(mental_states)
            cost = (base_carried_need + self._get_positive_auc(mental_states, time=1)).sum()
            return cost
        else:
            cost = 0

        for step in range(diagonal_steps):
            # Bug: Only take the positive part of the slope_carried_need. Right now it doesn't matter if the needs are negative.
            base_carried_need = self._positive_mental_states(mental_states) * math.sqrt(2)
            slope_carried_need = self._get_positive_auc(mental_states, math.sqrt(2))
            cost += (is_time_varying * (base_carried_need + slope_carried_need)).sum()
            mental_states += math.sqrt(2) * self._mental_states_slope

        for step in range(straight_steps):
            base_carried_need = self._positive_mental_states(mental_states)
            slope_carried_need = self._get_positive_auc(mental_states, 1.)
            cost += (is_time_varying * (base_carried_need + slope_carried_need)).sum()
            mental_states += self._mental_states_slope

        return cost

    def get_observation(self):
        observation = [self._env_map.copy(), self._mental_states.copy()]
        for i in range(len(self._environment_states_parameters)):
            observation.append(self._environment_states_parameters[i].copy())
        return observation

    def _update_agent_locations(self, new_location):
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 0
        self._agent_location = new_location
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

    def _update_object_locations(self):
        if self._env_map[1:, self._agent_location[0], self._agent_location[1]].sum() == 0:  # not reached an object
            return

        reached_object_type = np.argwhere(self._env_map[1:, self._agent_location[0], self._agent_location[1]]).item()
        grabbed_obj = self._env_map_dict.pop(
            (reached_object_type, self._agent_location[0], self._agent_location[1]))
        grabbed_obj.visible = False
        self._env_map[reached_object_type + 1, self._agent_location[0], self._agent_location[1]] = 0
        if self.object_reappears:
            self._init_random_map(object_num_on_map=self.each_type_object_num)  # argument is kind of redundant

    def _update_objects_status(self, object_event):
        to_pop = []
        environment_changed = False
        # check the plots  and see the bug
        visible_objects = {}
        can_make_invisible = True
        for key in self._env_map_dict.keys():
            visible_objects[self._env_map_dict[key].index] = deepcopy(key)

        for obj_type in range(self.object_type_num):
            if self.each_type_invisible_object_num[obj_type] == 1: # Only allow one invisible object of each type (for now)
                can_make_invisible = False
            for at in range(self.each_type_object_num[obj_type]):
                at_object = self._object_pool[obj_type][at]
                if at_object.visible:
                    visible_prob = decay_func(at_object.visible_dur)
                    sample = random.random()
                    if sample > visible_prob and not object_event and can_make_invisible:  # make it invisible, only when no other object event happened
                        object_key = visible_objects[at_object.index]
                        object_type = object_key[0]
                        object_loc = object_key[1:]
                        at_object.visible = False
                        at_object.to_appear = False
                        at_object.invisible_dur = random.randint(at_object.invisible_steps_range[0],
                                                                 at_object.invisible_steps_range[1])

                        to_pop.append(deepcopy(object_key))
                        self._env_map[object_type + 1, object_loc[0], object_loc[1]] = 0
                        object_event = True
                        environment_changed = True
                        self.each_type_invisible_object_num[obj_type] += 1
                    else:
                        at_object.visible_dur += 1
                elif at_object.invisible_dur <= 1 and not object_event:
                    at_object.visible = False
                    at_object.to_appear = True
                    object_event = True
                    environment_changed = True
                    self.each_type_invisible_object_num[obj_type] -= 1
                else:
                    at_object.invisible_dur -= 1
        for key in to_pop:
            self._env_map_dict.pop(key)
        self._init_random_map()
        return environment_changed

    def _positive_mental_states(self, mental_states=None):
        if mental_states is None:
            mental_states = self._mental_states.copy()
        positive_mental_state = np.maximum(0, mental_states)  # * (self._mental_states_slope > 0))
        return positive_mental_state

    def _update_mental_state_after_step(self, dt):
        dz = (self._mental_states_slope * dt)
        self._mental_states += dz

    def _update_mental_states_after_object(self, u):  # u > 0
        mental_states_threshold = np.empty_like(self._mental_states)
        for i, state in enumerate(self._mental_states):
            if state < self._no_reward_threshold:
                mental_states_threshold[i] = state
            else:
                mental_states_threshold[i] = self._no_reward_threshold
        self._mental_states += u
        self._mental_states = np.maximum(self._mental_states, mental_states_threshold)

    def _init_object_num_on_map(self) -> np.array:
        # e.g., self.few_many_objects : ['few', 'many']
        few_range = np.array([1]) #np.array([1, 2])
        many_range = np.array([3]) #np.array([3, 4, 5])
        ranges = {'few': few_range,
                  'many': many_range}
        each_type_object_num = np.zeros((self.object_type_num,), dtype=int)
        for i, item in enumerate(self.few_many_objects):
            at_type_obj_num = np.random.choice(ranges[item])
            each_type_object_num[i] = at_type_obj_num

        return each_type_object_num

    def _init_random_map(self, object_num_on_map=None):  # add agent location
        if self._env_map[0, :, :].sum() == 0:  # no agent on map
            self._agent_location = np.random.randint(low=0, high=self.height, size=(2,))
            self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

        # object_num_already_on_map = self._env_map[1:, :, :].sum(axis=(1, 2))
        if object_num_on_map is not None:
            self.each_type_object_num = object_num_on_map

        object_count = 0
        for obj_type in range(self.object_type_num):
            for at in range(self.each_type_object_num[obj_type]):
                at_object = self._object_pool[obj_type][at]
                if not at_object.visible and at_object.to_appear:
                    while True:
                        sample_location = np.random.randint(low=0, high=[self.height, self.width],
                                                            size=(self.object_type_num,))
                        if self._env_map[:, sample_location[0], sample_location[1]].sum() == 0:
                            self._env_map[1 + obj_type, sample_location[0], sample_location[1]] = 1
                            self._env_map_dict[(obj_type, sample_location[0], sample_location[1])] = at_object
                            at_object.visible = True
                            at_object.visible_dur = 0
                            break

                object_count += 1

    def _init_random_mental_states(self):
        self._mental_states[:] = self._get_random_vector(size=self.object_type_num,
                                                         attr_range=self.params.INITIAL_MENTAL_STATES_RANGE,
                                                         prob_equal=self.params.PROB_EQUAL_PARAMETERS)

    def _init_random_parameters(self):
        for i in range(len(self._environment_states_parameters_range)):
            self._environment_states_parameters[i][:] = self._get_random_vector(
                size=self.object_type_num,
                attr_range=self._environment_states_parameters_range[i],
                prob_equal=self.params.PROB_EQUAL_PARAMETERS,
                only_positive=True,
                dtype=self.params.MENTAL_STATES_SLOPE_DTYPE)
        # self._init_object_rewards()

        diag = np.eye(self._object_coefficients.shape[0], dtype=bool)
        p = random.uniform(0, 1)
        if p >= self._object_coefficients_min:  # fixed pref and time-vary
            non_diag_ele = 0
            diag_ele = self._object_coefficients[diag]
            for i in range(diag_ele.shape[0]):
                if diag_ele[i] < self.params.MIN_MAX_OBJECT_REWARD[
                    0]:  # we put zero if diagonal of object_coef is less than 5
                    diag_ele[i] = 0
            consistent = (self._mental_states_slope > 0) * (diag_ele != 0)
            self._mental_states_slope *= consistent
            diag_ele *= consistent
        else:  # pref for variety
            diag_ele = self._object_coefficients[diag].copy()
            for i in range(diag_ele.shape[0]):
                diag_ele[i] = 1 if diag_ele[i] < self.params.MIN_MAX_OBJECT_REWARD[0] else diag_ele[i]
            non_diag_ele = diag_ele.copy()

        diag_ele *= -1
        self._object_coefficients[~diag] = non_diag_ele
        self._object_coefficients[diag] = diag_ele

    def _get_random_vector(self, size, attr_range, prob_equal=0, only_positive=False, dtype='float'):
        p = random.uniform(0, 1)
        if p <= prob_equal:
            size = 1

        if dtype == 'float':
            random_vec = np.random.uniform(low=attr_range[0],
                                           high=attr_range[1],
                                           size=(size,))
        elif dtype == 'int':
            random_vec = np.random.randint(low=attr_range[0],
                                           high=attr_range[1] + 1,
                                           size=(size,),
                                           dtype=dtype)
        else:
            print('unrecognized dtype for parameters vector')
            exit(1)
        if only_positive:
            random_vec = np.maximum(random_vec, 0)
        return random_vec

    def set_environment(self,
                        agent_location=None,
                        mental_states=None,
                        mental_states_slope=None,
                        object_locations=None,
                        object_rewards=None,
                        object_coefficients=None):
        assert not ((mental_states_slope is None) ^ (object_coefficients is None)), \
            'Both or None of states_params must be None'
        super().reset(seed=None)
        self._env_map[0, :, :] = 0
        self.each_type_object_num = [len(object_rewards[i]) for i in range(len(object_rewards))]
        if mental_states is not None:
            self._mental_states = np.array(mental_states, dtype=float)
        else:
            self._init_random_mental_states()

        if mental_states_slope is not None and object_coefficients is not None:
            self._mental_states_slope = mental_states_slope
            self._object_coefficients = object_coefficients
            self._environment_states_parameters[0][:] = self._mental_states_slope
            self._environment_states_parameters[1][:] = self._object_coefficients
        else:
            self._init_random_parameters()

        if agent_location is not None:
            self._agent_location = np.array(agent_location)
            self._env_map[0, agent_location[0], agent_location[1]] = 1
        self._init_objects(rewards=object_rewards)

        for obj_type in range(self.object_type_num):
            for at in range(self.each_type_object_num[obj_type]):
                at_object = self._object_pool[obj_type][at]
                if not at_object.visible:
                    self._env_map[
                        1 + obj_type, object_locations[obj_type][at][0], object_locations[obj_type][at][1]] = 1
                    self._env_map_dict[(obj_type,
                                        object_locations[obj_type][at][0],
                                        object_locations[obj_type][at][1])] = at_object
                    at_object.visible = True

    def get_env_dict(self):
        return deepcopy(self._env_map_dict)

    def get_possible_goal_locations(self):
        object_locations = np.argwhere(self._env_map[1:, :, :])[:, 1:]
        agent_locations = np.argwhere(self._env_map[0, :, :])
        return object_locations, agent_locations

    def set_mental_state(self, mental_state):
        assert mental_state.shape == self._mental_states.shape, "invalid mental_state size"
        self._mental_states = mental_state

    def _get_mental_states_reward(self, before_object, after_object, reached_object):
        is_time_varying = self._mental_states_slope > 0
        time_reward = np.maximum(0,
                                 before_object - after_object).sum()
        fixed_reward = (~is_time_varying * after_object * reached_object).sum()
        return time_reward, fixed_reward

    def _init_objects(self, rewards=None):
        if rewards is None:
            rewards = [[] for _ in range(self.object_type_num)]
        index = 0
        for i in range(self.object_type_num):
            self._object_pool.append([])
            for j in range(self.each_type_object_num[i]):
                if len(rewards[i]) > 0:
                    reward = int(rewards[i][j])
                else:
                    reward = self._get_random_vector(size=1,
                                                     attr_range=[r / 2 for r in self.params.OBJECT_REWARD_RANGE],
                                                     prob_equal=0,
                                                     only_positive=True).item()  # We only want even rewards in the range
                    reward = int(reward) * 2
                self._object_pool[i].append(Object(obj_type=i,
                                                   reward=reward,
                                                   visible=False,
                                                   index=index))
                index += 1

    def print_object_pool(self, print_visibility=False):
        for i in range(len(self._object_pool)):
            for j in range(len(self._object_pool[i])):
                print('index:', self._object_pool[i][j].index,
                      'reward:', self._object_pool[i][j].reward, end=' ')
                if print_visibility:
                    print('visible:', self._object_pool[i][j].visible,
                          'visible dur:', self._object_pool[i][j].visible_dur,
                          'invisible_dur:', self._object_pool[i][j].invisible_dur
                          )
                else:
                    print()

    def print_env_dict(self, print_visibility=False):
        for loc in self._env_map_dict.keys():
            print(loc, self._env_map_dict[loc].type, self._env_map_dict[loc].reward, end=' ')
            if print_visibility:
                print('visible:', self._env_map_dict[loc].visible,
                      'visible dur:', self._env_map_dict[loc].visible_dur,
                      'invisible_dur:', self._env_map_dict[loc].invisible_dur)
            else:
                print()

    def check_dict(self):
        all_obj = np.argwhere(self._env_map[1:, :, :])
        for obj in all_obj:
            if tuple(obj) not in self._env_map_dict.keys():
                print(obj + 'not in dict')
                return True
        return False
