import os.path
from copy import deepcopy
import operator
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from Environment import Environment, get_distance_between_locations, get_pairwise_distance
import numpy as np
import math
from math import inf
from torch import nn
from View import plot_env
import pickle


def init_empty_tensor(size: tuple):
    return torch.empty(size, dtype=torch.float64)


class DecisionMaking:
    def __init__(self, params):
        self.step_action_dict = None
        self.dt_tensor = None
        self.output_str = None
        self.action_step = None
        self.few_many_dict = None
        self.episode_step_num = None
        self.env_steps_tensor = None
        self.states_params_steps_tensor = None
        self.mental_state_steps_tensor = None
        self.states_params_tensor = None
        self.mental_state_tensor = None
        self.env_tensor = None
        self.params = params
        self.horizon = self.params.TIME_HORIZON  # self.params.OBJECT_HORIZON
        if self.params.DEVICE == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.params.DEVICE
        # self.sigmoid = nn.Sigmoid()
        self.mean_goal_returns = dict()
        self.possible_trajectories = []
        self.possible_trajectories_horizon = []
        self.mean_goal_returns_2 = dict()
        self.gamma = self.params.GAMMA
        self.init_data_tensors()
        self.use_estimated_transition = self.params.USE_ESTIMATED_TRANSITION
        self.min_object_time_init = 13
        self.min_object_time = self.min_object_time_init
        self.temp_plots_dir = './TempPlots'
        if not os.path.exists(self.temp_plots_dir):
            os.mkdir(self.temp_plots_dir)

    def reset_estimates(self, goal_locations):
        self.possible_trajectories = []
        self.possible_trajectories_horizon = []
        self.mean_goal_returns_2 = dict()
        for goal in goal_locations:
            self.mean_goal_returns_2[tuple(goal)] = -inf

    def estimate_reward_of_possible_trajectories(self, environment: Environment):
        for i, trajectory in enumerate(self.possible_trajectories):
            cum_reward = 0
            imagined_environment = deepcopy(environment)
            state = environment.get_observation()
            env_map = torch.Tensor(state[0]).unsqueeze(0)
            for obj in trajectory:
                goal_map = torch.zeros_like(env_map[:, 0, :, :])
                goal_map[0, obj[0], obj[1]] = 1
                next_obs, pred_reward, _, _, _ = imagined_environment.step(goal_map=goal_map.squeeze().numpy())
                next_mental_state = next_obs[1]
                imagined_environment.set_mental_state(next_mental_state)
                cum_reward += pred_reward * self.params.GAMMA
            self.mean_goal_returns_2[tuple(trajectory[0])] = max(self.mean_goal_returns_2[tuple(trajectory[0])],
                                                                 cum_reward / self.possible_trajectories_horizon[i])

    def generate_possible_trajectories(self,
                                       agent_location,
                                       object_locations,
                                       horizon,
                                       grabbed_goals):
        if horizon >= self.horizon:
            self.possible_trajectories.append(grabbed_goals)
            self.possible_trajectories_horizon.append(horizon)
            return

        # if not np.all(object_locations == agent_location, axis=1).any():  # agent location not in objects
        goal_locations = np.concatenate([object_locations, agent_location], axis=0)
        # else:
        #     goal_locations = deepcopy(object_locations)
        diagonal, straight = get_distance_between_locations(agent_location[0, 0], agent_location[0, 1],
                                                            goal_locations[:, 0], goal_locations[:, 1])

        agent_to_objects_distances = math.sqrt(2) * diagonal + straight
        for obj_id, obj in enumerate(goal_locations):
            dt = np.array(1) if agent_to_objects_distances[obj_id] < 1.4 else agent_to_objects_distances[obj_id]
            if agent_to_objects_distances[obj_id] == 0:  # stayed
                new_object_locations = deepcopy(object_locations)
            else:
                new_object_locations = object_locations[~np.all(object_locations == obj, axis=1)]
            new_grabbed = deepcopy(grabbed_goals)
            new_grabbed.append(obj.tolist())
            self.generate_possible_trajectories(agent_location=np.expand_dims(obj, axis=0),
                                                object_locations=new_object_locations,
                                                horizon=horizon + dt,
                                                grabbed_goals=new_grabbed)

    # def imagine(self, environment: Environment,
    #             horizon,
    #             grabbed_goals,
    #             cum_reward,
    #             n_object_grabbed,
    #             n_total_objects,
    #             time,
    #             n_rewarding_object,
    #             only_agent_location=False,
    #             cum_reward_list=None) -> Tuple[dict, dict]:
    #     # We should count the number of staying steps to avoid infinite recursions
    #     if cum_reward_list is None:
    #         cum_reward_list = []
    #     goal_returns = dict()
    #     goal_returns_time_took = dict()

    # if horizon >= self.horizon and n_rewarding_object > 0:  # at least one rewarding object for updating min time
    #     self.min_object_time = min(self.min_object_time, time)
    # if horizon >= self.horizon:  # or n_object_grabbed == n_total_objects:
    #     # and n_rewarding_object > 0:  # at least one rewarding object
    #     # print(time, grabbed_goals, end=' ')
    #     # print(cum_reward, cum_reward/time)
    #     first_step = tuple(grabbed_goals[0])
    #     if first_step in self.mean_goal_returns:
    #         self.mean_goal_returns[first_step] = max(self.mean_goal_returns[first_step],
    #                                                  cum_reward / horizon)
    #     else:
    #         self.mean_goal_returns[first_step] = cum_reward / horizon
    #     self.output_str += '{0} {1} {2} {3}\n'.format(horizon, grabbed_goals, cum_reward, cum_reward / time)
    #     self.output_str += '{0}\n\n'.format(cum_reward_list)
    #     # if horizon >= 23: #self.horizon+4:  # give it another 2 steps chance to see if it reaches a better goal
    #     return goal_returns, goal_returns_time_took
    # # elif horizon >= self.horizon and time >= self.min_object_time:  # no objects, only staying
    # #     print(time, grabbed_goals, end=' ')
    # #     print(cum_reward)
    # #     return goal_returns
    # object_locations, agent_location = environment.get_possible_goal_locations()
    # state = environment.get_observation()
    # env_map = torch.Tensor(state[0]).unsqueeze(0)
    #
    # staying_goal = []
    # if only_agent_location:
    #     all_goal_locations = deepcopy(agent_location)
    # elif not np.all(object_locations == agent_location, axis=1).any():
    #     all_goal_locations = np.concatenate([object_locations, agent_location], axis=0)
    #     staying_goal.append(agent_location[0])
    # else:
    #     all_goal_locations = deepcopy(object_locations)
    # for obj in all_goal_locations:
    #     imagined_environment = deepcopy(environment)
    #     goal_map = torch.zeros_like(env_map[:, 0, :, :])
    #     goal_map[0, obj[0], obj[1]] = 1
    #
    #     next_obs, pred_reward, _, _, info = imagined_environment.step(goal_map=goal_map.squeeze().numpy())
    #     next_mental_state = next_obs[1]
    #
    #     imagined_environment.set_mental_state(next_mental_state)
    #     new_grabbed = deepcopy(grabbed_goals)
    #     new_grabbed.append(obj.tolist())
    #     future_goal_returns, future_goal_returns_time_takes = self.imagine(environment=imagined_environment,
    #                                                                        horizon=horizon + info['dt'],  # 1,
    #                                                                        grabbed_goals=new_grabbed,
    #                                                                        cum_reward=self.gamma * pred_reward + cum_reward,
    #                                                                        n_object_grabbed=n_object_grabbed + int(
    #                                                                            info['object']),
    #                                                                        n_total_objects=n_total_objects,
    #                                                                        time=time + info['dt'],
    #                                                                        n_rewarding_object=n_rewarding_object + int(
    #                                                                            info['rewarding']),
    #                                                                        only_agent_location=only_agent_location,
    #                                                                        cum_reward_list=cum_reward_list + [
    #                                                                            pred_reward])
    #
    #     # max_future_goal_return = tuple([0, 0]) if not future_goal_returns else max(future_goal_returns.items(),
    #     #                                                                            key=operator.itemgetter(1))
    #     # future_returns = max_future_goal_return[1]
    #     # time_takes = 0 if not future_goal_returns_time_takes else future_goal_returns_time_takes[max_future_goal_return[0]]
    #     # goal_returns[tuple(obj)] = self.gamma ** info['dt'] * (pred_reward + future_returns)
    #     # goal_returns_time_took[tuple(obj)] = info['dt'] + time_takes
    #
    # return goal_returns, goal_returns_time_took

    def get_goal_return(self, environment: Environment):
        # print('all objects: ', environment.get_possible_goal_locations())
        object_locations, agent_location = environment.get_possible_goal_locations()
        self.reset_estimates(goal_locations=np.concatenate([object_locations,
                                                            agent_location], axis=0))
        self.generate_possible_trajectories(agent_location=agent_location,
                                            object_locations=object_locations,
                                            horizon=0,
                                            grabbed_goals=[])
        self.estimate_reward_of_possible_trajectories(environment=deepcopy(environment))
        # print(datetime.now() - st1)
        # st2 = datetime.now()
        # self.mean_goal_returns.clear()
        # goal_returns, goal_returns_time = self.imagine(environment=deepcopy(environment),
        #                                                horizon=0,
        #                                                grabbed_goals=[],
        #                                                cum_reward=0,
        #                                                n_object_grabbed=0,
        #                                                n_total_objects=sum(environment.each_type_object_num),
        #                                                time=0,
        #                                                n_rewarding_object=0)
        # print(datetime.now() - st2)
        # return goal_returns, goal_returns_time

    def take_action(self, environment: Environment):
        self.output_str = ''
        self.min_object_time = self.min_object_time_init
        object_locations, agent_location = environment.get_possible_goal_locations()
        self.horizon = self.get_horizon(agent_location, object_locations)
        # goal_returns, goal_returns_time = self.get_goal_return(environment)
        self.get_goal_return(environment)
        # mean_goal_returns = self.mean_goal_returns.copy()
        # mean_goal_returns = {
        #     goal: (goal_returns[goal] / goal_returns_time[goal]) for goal in goal_returns.keys()
        # }
        # with open('./output.txt', 'w') as f:
        #     f.write(self.output_str)
        best_goal_location = max(self.mean_goal_returns_2.items(), key=operator.itemgetter(1))[0]
        goal_map = np.zeros((self.params.HEIGHT, self.params.WIDTH))
        goal_map[best_goal_location[0], best_goal_location[1]] = 1
        return goal_map

    def generate_behavior(self, few_many=None, episodes_initial_environment=None):
        plot_during_data_generation = False
        if episodes_initial_environment is None:
            episodes_initial_environment = []
        if few_many is None:
            few_many = [None] * int(self.params.EPISODE_NUM)
        for episode in range(int(self.params.EPISODE_NUM)):
            print(episode)
            self.action_step = 0
            self.step_action_dict[episode] = dict()
            if episode < len(episodes_initial_environment):  # each element is an Environment object
                environment = episodes_initial_environment[episode]
            else:
                if episode > len(few_many) or few_many[episode] is None:
                    episode_few_many = [np.random.choice(['few', 'many']) for _ in range(self.params.OBJECT_TYPE_NUM)]
                else:
                    episode_few_many = few_many[episode]
                # self.few_many_dict[episode] = ' '.join(episode_few_many)
                self.few_many_dict[' '.join(episode_few_many)].append(episode)
                environment = Environment(params=self.params, few_many_objects=episode_few_many)
            state, _ = environment.reset()
            env_dict = environment.get_env_dict()
            self.step_action_dict[episode][0] = []
            self.add_data_point(state, env_dict, episode, step=0, dt=0,
                                plot_data_point=plot_during_data_generation)  # initial env
            # for step in range(int(self.params.EPISODE_STEPS)):
            step = 0
            while step < int(self.params.EPISODE_STEPS):
                step_completed = True
                if step not in self.step_action_dict[episode].keys():
                    self.step_action_dict[episode][step] = []
                print(step, end=' ')
                environment.object_reappears = False
                final_goal_map = self.take_action(environment)
                environment.object_reappears = True

                final_goal_location = np.argwhere(final_goal_map).flatten()
                _, agent_location = environment.get_possible_goal_locations()
                n_diagonal, n_straight, directions = get_distance_between_locations(agent_location[0, 0],
                                                                                    agent_location[0, 1],
                                                                                    final_goal_location[0],
                                                                                    final_goal_location[1],
                                                                                    with_direction=True)
                intermediate_goal_location = agent_location.flatten()
                if n_diagonal + n_straight > 0:  # agent moves
                    for intermediate_step in next_step(n_diagonal, n_straight, directions):
                        intermediate_goal_location += intermediate_step
                        intermediate_goal_map = np.zeros_like(final_goal_map)
                        intermediate_goal_map[intermediate_goal_location[0],
                        intermediate_goal_location[1]] = 1
                        next_state, reward, _, _, info = environment.step(goal_map=intermediate_goal_map,
                                                                          update_objects_status=True)
                        state = deepcopy(next_state)
                        env_dict = environment.get_env_dict()
                        # There is bug in adding point to the tensor. check the plots!
                        self.add_data_point(state, env_dict, episode,
                                            step=step, dt=info['dt'].item(),
                                            plot_data_point=plot_during_data_generation)
                        if info['environment_changed']:  # appearing or disappearing of objects, ends the current episode_step
                            step_completed = False
                            break
                else:  # agent stays
                    next_state, reward, _, _, info = environment.step(goal_map=final_goal_map,
                                                                      update_objects_status=True)
                    state = deepcopy(next_state)
                    env_dict = environment.get_env_dict()
                    self.add_data_point(state, env_dict, episode,
                                        step=step, dt=info['dt'].item(),
                                        plot_data_point=plot_during_data_generation)
                if step_completed:
                    step += 1
                # self.add_data_point(state, env_dict, episode, step)
                # state = deepcopy(next_state)
                # env_dict = environment.get_env_dict()
            print()
        self.save_tensors()

    # def load_transition_model(self) -> TransitionNet:
    #     transition_weights = torch.load(os.path.join(self.params.TRANSITION_MODEL, 'model.pt'),
    #                                     map_location=self.device)
    #     transition = TransitionNet(self.params, device=self.device)
    #     transition.load_state_dict(transition_weights)
    #     return transition

    def get_horizon(self, agent_location, object_locations):
        diagonal, straight = get_distance_between_locations(agent_location[0, 0], agent_location[0, 1],
                                                            object_locations[:, 0], object_locations[:, 1])
        if len(diagonal) + len(straight) == 0:  # no object no map
            return math.sqrt(2) * self.params.WIDTH

        pairwise_object_distances = get_pairwise_distance(object_locations, object_locations)
        agent_to_objects_distances = math.sqrt(2) * diagonal + straight
        farthest_object = np.argmax(agent_to_objects_distances)
        horizon = agent_to_objects_distances[farthest_object]
        at_object = farthest_object
        for i in range(1, self.params.OBJECT_HORIZON):
            farthest_from_at = np.argmax(pairwise_object_distances[at_object, :])
            horizon += max(pairwise_object_distances[at_object, farthest_from_at], 1)  # stay if no objects
            pairwise_object_distances[at_object, :] = 0  # remove the farthest
            pairwise_object_distances[:, at_object] = 0
            at_object = farthest_from_at

        return horizon
        # if len(diagonal) > 0 or len(straight) > 0:
        #     self.horizon = max(math.sqrt(2) * diagonal + straight) + 1  # distance to the farthest
        #     # object plus one step of staying
        #     # self.horizon = math.ceil(max(math.sqrt(2) * diagonal + straight))
        # else:  # No objects on the map
        #     self.horizon = math.sqrt(2) * self.params.WIDTH

    def add_data_point(self, state: list, env_dict: dict, episode, step, dt, plot_data_point=False):
        env_map = torch.Tensor(state[0])
        mental_states = torch.Tensor(state[1])
        states_params = torch.Tensor(np.concatenate([state[2], state[3].flatten()]))
        for loc in env_dict.keys():
            env_map[loc[0] + 1, loc[1], loc[2]] = deepcopy(env_dict[loc].reward)

        self.env_tensor[episode, self.action_step, :, :, :] = env_map
        self.mental_state_tensor[episode, self.action_step, :] = mental_states
        self.states_params_tensor[episode, self.action_step, :] = states_params
        self.dt_tensor[episode, self.action_step] = dt

        self.step_action_dict[episode][step].append(self.action_step)
        if plot_data_point:
            fig, ax = plot_env(env_map, show_reward=True)
            fig.savefig(os.path.join(self.temp_plots_dir,
                                     '{0}_{1}_ms({2:.2f}_{3:.2f})_sl({4:.2f}_{5:.2f}).png'.format(episode,
                                                                                                  self.action_step,
                                                                                                  mental_states[0],
                                                                                                  mental_states[1],
                                                                                                  states_params[0],
                                                                                                  states_params[1])))
            plt.close()
        self.action_step += 1
        self.episode_step_num[episode] = self.action_step

    def init_data_tensors(self):
        self.action_step = 0
        self.env_tensor = init_empty_tensor(size=(self.params.EPISODE_NUM,
                                                  # self.params.EPISODE_STEPS,
                                                  self.params.EPISODE_ACTION_STEPS,
                                                  self.params.OBJECT_TYPE_NUM + 1,
                                                  self.params.HEIGHT,
                                                  self.params.WIDTH))
        self.mental_state_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                      # self.params.EPISODE_STEPS,
                                                      self.params.EPISODE_ACTION_STEPS,
                                                      self.params.OBJECT_TYPE_NUM))
        self.states_params_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                                       # self.params.EPISODE_STEPS,
                                                       self.params.EPISODE_ACTION_STEPS,
                                                       self.params.OBJECT_TYPE_NUM * 2 + self.params.OBJECT_TYPE_NUM))

        self.dt_tensor = init_empty_tensor((self.params.EPISODE_NUM,
                                            self.params.EPISODE_ACTION_STEPS))
        self.episode_step_num = torch.zeros((self.params.EPISODE_NUM, 1))
        self.step_action_dict = np.empty((self.params.EPISODE_NUM, ), dtype=object)
        self.few_many_dict = {'few few': [],
                              'few many': [],
                              'many few': [],
                              'many many': []}  # np.zeros((self.params.EPISODE_NUM,), dtype='<U9')

    def save_tensors(self):
        data_dir = './Data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        torch.save(self.env_tensor, os.path.join(data_dir, 'environments.pt'))
        torch.save(self.mental_state_tensor, os.path.join(data_dir, 'mental_states.pt'))
        torch.save(self.states_params_tensor, os.path.join(data_dir, 'states_params.pt'))
        torch.save(self.dt_tensor, os.path.join(data_dir, 'dt.pt'))
        with open(os.path.join(data_dir, 'few_many_dict.pkl'), 'wb') as file_handler:
            pickle.dump(self.few_many_dict, file_handler)

        np.save(os.path.join(data_dir, 'step_actions_dict.npy'), self.step_action_dict)
        # torch.save(self.env_steps_tensor, os.path.join(data_dir, 'environments_steps.pt'))
        # torch.save(self.mental_state_steps_tensor, os.path.join(data_dir, 'mental_states_steps.pt'))
        # torch.save(self.states_params_steps_tensor,
        #            os.path.join(data_dir, 'states_params_steps.pt'))  # [slope1, slope2, object coeffs (4)]

    # def next_step_state(self, episode, state_0_ind, state_1_ind):
    #     env_map_0 = self.env_tensor[episode, state_0_ind, :, :, :]
    #     env_map_1 = self.env_tensor[episode, state_1_ind, :, :, :]
    #     mental_state = self.mental_state_tensor[episode, state_0_ind, :].clone()
    #     location_0 = torch.argwhere(env_map_0[0, :, :])
    #     location_1 = torch.argwhere(env_map_1[0, :, :])
    #     diagonal_steps, straight_steps = get_distance_between_locations(location_0[0, 0],
    #                                                                     location_0[0, 1],
    #                                                                     location_1[0, 0],
    #                                                                     location_1[0, 1])
    #     d = location_1 - location_0
    #     diagonal_path = torch.tensor(np.tile(np.sign([d[0, 0], d[0, 1]]), reps=[diagonal_steps, 1]))
    #     for diag_step in diagonal_path:
    #         d -= diag_step
    #     straight_path = torch.tensor(np.tile(np.sign([d[0, 0], d[0, 1]]), reps=[straight_steps, 1]))
    #     all_steps = torch.cat([diagonal_path, straight_path], dim=0)
    #     for step in range(all_steps.shape[0] - 1):
    #         location_0 = location_0 + all_steps[step, :]
    #         next_env_map = torch.zeros_like(env_map_0)
    #         next_env_map[0, location_0[0, 0], location_0[0, 1]] = 1
    #         next_env_map[1:, :, :] = env_map_0[1:, :, :]
    #         mental_state += torch.linalg.vector_norm(all_steps[step, :].float()) * self.states_params_tensor[episode,
    #                                                                                state_0_ind, :2]
    #         yield next_env_map, mental_state


def next_step(n_diagonal, n_straight, directions):
    for i in range(n_diagonal):
        step = directions[0]
        yield step
    for i in range(n_straight):
        step = directions[1]
        yield step


def check_dict(state, env_dict):
    env_map = torch.Tensor(state[0])
    all_obj = torch.argwhere(env_map[1:, :, :])
    for obj in all_obj:
        if tuple(obj.numpy()) not in env_dict.keys():
            print(obj, 'not in dict')
            return True
    return False
