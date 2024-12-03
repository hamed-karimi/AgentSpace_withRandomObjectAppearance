from Environment import Environment
from Utils import Utils
from DecisionMaking import DecisionMaking
import numpy as np
from Test import Test
from View import plot_tensors, create_video_from_plots
from Object import Object

# def get_random_action(state: list):
#     env_map = np.array(state[0])
#     goal_map = np.zeros_like(env_map[0, :, :])
#
#     all_object_locations = np.stack(np.where(env_map), axis=1)
#     goal_index = np.random.randint(low=0, high=all_object_locations.shape[0], size=())
#     goal_location = all_object_locations[goal_index, 1:]
#
#     # goal_location = all_object_locations[0, 1:] # ERASE THIS!!!!!
#     goal_map[goal_location[0], goal_location[1]] = 1
#     return goal_map


if __name__ == '__main__':
    utils = Utils()
    # agent = DecisionMaking(params=utils.params)
    # few_many = [
    #     ['few', 'few'],
    #     ['few', 'many'],
    #     ['many', 'few'],
    #     ['many', 'many']
    # ]
    # agent.generate_behavior(few_many=few_many)
    # print('Plotting...')
    # plot_tensors(utils.params.EPISODE_NUM,
    #              agent.few_many_dict,
    #              agent.env_tensor,
    #              agent.mental_state_tensor,
    #              agent.states_params_tensor,
    #              agent.episode_step_num
    #              )
    # print('Making video...')
    create_video_from_plots(utils.params)

    # environment = Environment(params=utils.params, few_many_objects=['few', 'many'], object_reappears=False)
    # index = 0
    # each_type_object_num = [2, 2]
    # object_type_num = 2
    # rewards = [[18., 10], [6., 18.]]
    # object_locations = [[(2, 7), (7, 4)], [(5, 7), (0, 1)]]
    #
    # environment.set_environment(agent_location=np.array([2, 3]), # work on this function and make it useful for debugging
    #                             mental_states=np.array([63.18, -0.12]),
    #                             mental_states_slope=np.array([3., 3.]),
    #                             object_locations=object_locations,
    #                             object_rewards=rewards,
    #                             object_coefficients=np.array([[-1., 0.], [0., -1.]]))
    # episodes = 12
    # while True:
    #     environment.object_reappears = False
    #     goal_map = agent.take_action(environment=environment)
    #     print(goal_map)
    #     environment.object_reappears = True
    #     next_state, reward, _, _, _ = environment.step(goal_map=goal_map)
    #     print(environment._mental_states)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
