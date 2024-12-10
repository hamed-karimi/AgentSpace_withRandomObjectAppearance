import math
import os.path
import pickle

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox, TextArea)
import numpy as np
import torch
import cv2
import re


class Face():
    def __init__(self):
        self.large_face = Image.open('./face_ill.png')
        self.face_scale = 0.168  # 1.59
        self.face_copy = self.large_face.resize((int(self.face_scale * self.large_face.size[0]),
                                                 int(self.face_scale * self.large_face.size[1])))
        self.grid_start = (1645, 1580)
        self.box = [532, 532]
        self.gap = ((self.box[0] - self.face_copy.size[0]) // 2, (self.box[1] - self.face_copy.size[1]) // 2 + 5)
        big_side = max(self.face_copy.size)
        self.background = Image.new('RGBA', (big_side, big_side), (255, 255, 255, 255))


def get_intermediate_face_loc(face: Face, agent_loc1, time_point1, agent_loc2, time_point2, jump, smooth=1):
    # d = np.linalg.norm(agent_loc2 - agent_loc1)
    d = time_point2 - time_point1
    n_steps = max(int(d / jump), 1)
    if smooth:
        xs = np.linspace(agent_loc1[0], agent_loc2[0], num=n_steps, endpoint=False)
        ys = np.linspace(agent_loc1[1], agent_loc2[1], num=n_steps, endpoint=False)
    else:
        xs = np.linspace(agent_loc1[0], agent_loc1[0], num=n_steps, endpoint=False)
        ys = np.linspace(agent_loc1[1], agent_loc1[1], num=n_steps, endpoint=False)

    face_locations = []
    for i in range(n_steps):
        face_xy = (int(face.grid_start[0] + face.box[0] * ys[i] + face.gap[0]),
                   int(face.grid_start[1] + face.box[1] * xs[i] + face.gap[1]))

        face_locations.append(face_xy)

    return face_locations


def make_frame_square(image):
    # image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    big_side = width if width > height else height
    background = Image.new('RGBA', (big_side, big_side), (255, 255, 255, 255))
    offset = (int(round(((big_side - width) / 2), 0)), int(round(((big_side - height) / 2), 0)))
    background.paste(image, offset)
    # background.save('out.png')
    # print("Image has been resized !")
    return background

def get_marker(layer: int, reward: int):
    markers = {
        # 1: '■',
        # 2: '▲',
        # 0: 'face',
        1: 'beverage',
        2: 'food'
    }
    if layer == 0:
        return './face_225.png'
    else:
        return './{0}/{1}.png'.format(markers[layer], reward)


def plot_env(env_map, show_reward=False):
    zoom = {
        0: .1,
        1: .01,
        2: .01
    }

    # Create a figure and axis
    fig, ax = plt.subplots(dpi=1152)

    # Iterate through the grid and plot the shapes
    for i in range(env_map.shape[1]):
        for j in range(env_map.shape[2]):
            layer = torch.where(env_map[:, i, j])[0]
            # print(value)
            if len(layer) > 0:  # and value[0].item() > 0:
                reward = str(env_map[layer[0].item(), i, j].int().item())
                reward_plot = int(reward)  # // 5
                marker = get_marker(layer[0].item(),
                                    reward_plot)
                arr_img = plt.imread(marker, format='png')

                imagebox = OffsetImage(arr_img, zoom=zoom[layer[0].item()])
                imagebox.image.axes = ax

                ab = AnnotationBbox(imagebox, (j + .5, i + .5),
                                    frameon=False
                                    )

                ax.add_artist(ab)

                if layer[0].item() > 0 and show_reward:
                    text_box = TextArea(reward, textprops={'fontsize': 5})
                    ab = AnnotationBbox(text_box, (j + .2, i + .9),
                                        frameon=False)
                    ax.add_artist(ab)

    # Set axis limits and labels
    ax.set_xlim(0, env_map.shape[1])
    ax.set_ylim(0, env_map.shape[2])
    ax.set_xticks(np.arange(env_map.shape[1]))
    ax.set_yticks(np.arange(env_map.shape[2]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_box_aspect(aspect=1)

    # Show the plot
    plt.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    # plt.show()
    return fig, ax


def plot_tensors(episode_num, few_many_array, env_tensor, mental_state_tensor,
                 states_params_tensor, step_num_tensor):
    base_dir = './Plots'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # episode_dict = {
    #     'few few': 0,
    #     'few many': 0,
    #     'many few': 0,
    #     'many many': 0,
    # }
    for it1 in ['few', 'many']:
        for it2 in ['few', 'many']:
            path = os.path.join(base_dir, it1 + ' ' + it2)
            if not os.path.exists(path):
                os.mkdir(path)

    for few_many in few_many_array.keys():
        episodes = few_many_array[few_many]
        for episode in episodes:
            for step in range(int(step_num_tensor[episode])):
                save_dir = os.path.join(base_dir, few_many)

                fig, ax = plot_env(env_tensor[episode, step, :, :, :], show_reward=False)
                fig.savefig('{0}/episode_{1}_step{2}_ms({3:.2f}_{4:.2f})_sl({5:.2f}_{6:.2f}).png'.format(save_dir,
                                                                                                         # episode_dict[few_many_dict[episode]],
                                                                                                         episode,
                                                                                                         step,
                                                                                                         mental_state_tensor[
                                                                                                             episode, step, 0],
                                                                                                         mental_state_tensor[
                                                                                                             episode, step, 1],
                                                                                                         states_params_tensor[
                                                                                                             episode, step, 0],
                                                                                                         states_params_tensor[
                                                                                                             episode, step, 1]))
                plt.close()
        # episode_dict[few_many_dict[episode]] += 1


def get_plots_file_info(it1: str, it2: str, few_many_dict):
    plots_dir = './Plots'
    all_filenames = os.listdir(os.path.join(plots_dir, it1 + ' ' + it2))
    if len(all_filenames) == 0:
        return []
    episode_step = np.array([[int(f.split('_')[1]), int(f.split('_')[2][4:])] for f in all_filenames])
    plots_filenames = []
    condition_episodes = []
    if len(episode_step) > 0:
        condition_episodes = few_many_dict[it1 + ' ' + it2]  # np.unique(episode_step[:, 0])
        # episode_num = condition_episodes.shape[0]
        for episode in condition_episodes:
            step_num = episode_step[episode_step[:, 0] == episode, :].shape[0]
            episode_plots = []
            for step in range(step_num):
                for name in all_filenames:
                    if name.startswith('episode_{0}_step{1}_'.format(episode, step)):
                        episode_plots.append(name)
            plots_filenames.append(episode_plots)
    return plots_filenames, condition_episodes


def get_next_metronome_filename(at_index, metronome_jump, direction):
    if at_index + direction * metronome_jump >= 120 or at_index + direction * metronome_jump < 0:
        direction *= -1
    return at_index + direction * metronome_jump, direction


def get_one_step_face_locations(face, agent_loc1=None, time_point1=None, agent_loc2=None, time_point2=None, smooth=True,
                                intermediate_jump=0.):
    face_locations = get_intermediate_face_loc(face,
                                               agent_loc1,
                                               time_point1,
                                               agent_loc2,
                                               time_point2,
                                               intermediate_jump,
                                               smooth)  # from the second frame onwar

    return face_locations  # frames

def get_next_frame_with_metronome(grid: Image,
                                  face: Face = None,
                                  background_location=None,
                                  face_location=None,
                                  metronome_filename=None):
    xy = [int(7372 // 2), int(600)]
    metronome = Image.open(metronome_filename).copy()
    metronome.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    grid.paste(metronome, (xy[0] - 500, xy[1]),
               metronome.convert('RGBA'))
    if face is not None:
        grid.paste(face.background, background_location,
                   face.background.convert('RGBA'))

        grid.paste(face.face_copy, face_location,
                   face.face_copy.convert('RGBA'))
    frame_np = np.array(grid, dtype=np.uint8)  # [:, 922:6451, :]
    frame = cv2.cvtColor(np.array(frame_np), cv2.COLOR_RGB2BGR)
    return frame


def get_step_actions(episode_step_actions_dict, step):
    if step == 0:  # first step
        step_actions = episode_step_actions_dict[step][:-1]
    elif step == max(episode_step_actions_dict.keys()):  # last step
        step_actions = [episode_step_actions_dict[step - 1][-1]]
        step_actions.extend(episode_step_actions_dict[step])
    else:  # all others
        step_actions = [episode_step_actions_dict[step - 1][-1]]
        step_actions.extend(episode_step_actions_dict[step][:-1])
    return step_actions


def create_video_from_plots(params):
    videos_dir = './Videos'
    if not os.path.exists('./Videos'):
        os.mkdir('./Videos')

    plots_dir = './Plots'
    data_dir = './Data'
    dt_tensor = torch.load(os.path.join(data_dir, 'dt.pt'))
    env_tensor = torch.load(os.path.join(data_dir, 'environments.pt'))
    with open(os.path.join(data_dir, 'few_many_dict.pkl'), 'rb') as f:
        few_many_dict = pickle.load(f)
    steps_action_dict = np.load(os.path.join(data_dir, 'step_actions_dict.npy'), allow_pickle=True)
    face = Face()

    for it1 in ['few', 'many']:
        for it2 in ['few', 'many']:
            if len(few_many_dict[it1 + ' ' + it2]) == 0:
                continue
            print(it1 + ' ' + it2)
            few_many_path = os.path.join(videos_dir, it1 + ' ' + it2)
            if not os.path.exists(few_many_path):
                os.mkdir(few_many_path)

            plots_filename, condition_episodes = get_plots_file_info(it1, it2, few_many_dict)
            frame_size = 7372  # 5529
            face_time_step = 4
            # frame_size = 1000
            # scale = 11.52
            for i, episode in enumerate(condition_episodes):
                at_filenames = plots_filename[i]
                step_action_num = len(at_filenames)
                episode_dt = dt_tensor[episode, :step_action_num]
                episode_env = env_tensor[episode, :step_action_num]
                fps = 20
                metronome_jump = 5
                frame_timepoints = (episode_dt * fps).cumsum(dim=0)  # params.SECONDS_PER_FRAME * step_num * 100

                # video_writer = cv2.VideoWriter(filename=os.path.join(few_many_path, 'episode_{0}.avi'.format(episode)),
                #                                # fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                #                                fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                #                                # MJPG codec with .MJPG output file works but the video is 142MB
                #                                # fourcc=-1,
                #                                fps=fps,
                #                                frameSize=(frame_size, frame_size))
                t1 = 0
                m_ind = 0
                direction = 1
                # step_num = 3
                steps = steps_action_dict[episode].keys()
                for step in range(len(steps)-1):
                    video_writer = cv2.VideoWriter(
                        filename=os.path.join(few_many_path, 'episode_{0}_step_{1}.avi'.format(episode, step)),
                        fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                        fps=fps,
                        frameSize=(frame_size, frame_size)
                    )
                    # step_actions = steps_action_dict[episode][step]
                    step_actions = get_step_actions(steps_action_dict[episode], step=step)
                    for step_action in step_actions: #range(step_num - 1):
                        # step_action = step_actions[action_i]
                        frame_path = os.path.join(plots_dir, it1 + ' ' + it2, at_filenames[step_action])
                        agent_loc1 = torch.argwhere(episode_env[step_action, 0, :, :]).squeeze()
                        agent_loc2 = torch.argwhere(episode_env[step_action + 1, 0, :, :]).squeeze()
                        # frame = cv2.imread(frame_path, cv2.IMREAD_COLOR) #[:, 922:6451, :]
                        show_agents = (agent_loc1 != agent_loc2)
                        grid = make_frame_square(Image.open(frame_path))
                        t2 = int(frame_timepoints[step_action + 1].round().item())
                        one_step_face_locations = get_one_step_face_locations(face=face,
                                                                              agent_loc1=agent_loc1,
                                                                              time_point1=t1,
                                                                              agent_loc2=agent_loc2,
                                                                              time_point2=t2,
                                                                              smooth=True,
                                                                              intermediate_jump=1)
                        div = int((t2 - t1) // len(one_step_face_locations))
                        for f_i, f in enumerate(range(t1, t2)):
                            m_name = './metronome/{0}.png'.format(m_ind)
                            face_location = one_step_face_locations[int(f_i // div)]
                            frame = get_next_frame_with_metronome(grid.copy(),
                                                                  face=face,
                                                                  background_location=one_step_face_locations[0],
                                                                  face_location=face_location,
                                                                  metronome_filename=m_name)
                            video_writer.write(frame)
                            m_ind, direction = get_next_metronome_filename(m_ind, metronome_jump, direction)

                        t1 = t2
                    video_writer.release()

                # video_writer = cv2.VideoWriter(
                #     filename=os.path.join(few_many_path, 'episode_{0}_step_{1}.avi'.format(episode, step_action_num - 1)),
                #     fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                #     fps=fps,
                #     frameSize=(frame_size, frame_size)
                # )
                # frame_path = os.path.join(plots_dir, it1 + ' ' + it2, at_filenames[step_action_num - 1])  # last step
                # grid = Image.open(frame_path)
                # last_m_name = './metronome/{0}.png'.format(m_ind)
                # frame = get_next_frame_with_metronome(grid.copy(), metronome_filename=last_m_name)
                #
                # video_writer.write(frame)
                #
                # video_writer.release()
