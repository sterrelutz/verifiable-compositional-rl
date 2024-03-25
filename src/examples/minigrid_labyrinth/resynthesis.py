# Imports
import os
import sys

sys.path.append('../..')
from environments.minigrid_labyrinth import Maze, MazeVariant
import numpy as np
from controllers.minigrid_controller import MiniGridController
from controllers.meta_controller import MetaController
from datetime import datetime
from MDP.high_level_mdp import HLMDP
from utils.results_saver import Results
import torch
import random


# TODO: fix that changing the environment of the

def evaluate_variants(env_list, meta_ctrl, ctrl_list, n_episodes, n_steps_per_meta_episode, n_steps_per_sub_episode,
                      log=True, save=False):
    if log:
        print(f'Evaluating performance of controllers with the following settings:\n'
              f'- n_episodes: {n_episodes}\n'
              f'- n_steps_meta: {n_steps_per_meta_episode}\n'
              f'- n_steps_sub: {n_steps_per_sub_episode}')

    if log:
        print(
            f'---------------------------\nMeta-controller')
    for e in range(len(env_list)):
        meta_success_rate = meta_ctrl.eval_performance(env_list[e], n_episodes=n_episodes,
                                                       n_steps=n_steps_per_meta_episode)
        if log:
            print(f'Environment variant {e}: {meta_success_rate}')

    for ctrl in ctrl_list:
        if log:
            print(f'---------------------------\nSub-controller {ctrl.controller_ind}')
        for e in range(len(env_list)):
            # ctrl.set_environment(env_list[e])
            sub_success_rate = ctrl.eval_performance_variant(n_episodes=n_episodes, n_steps=n_steps_per_sub_episode,
                                                             variant=e, save=save)
            if log:
                print(f'Environment variant {e}: {sub_success_rate[0]}')


def get_load_dir(load_folder_name):
    base_path = os.path.abspath(os.path.curdir)
    string_ind = base_path.find('src')
    assert string_ind >= 0
    base_path = os.path.join(base_path[0:string_ind + 4], 'data', 'saved_controllers')
    return os.path.join(base_path, load_folder_name)


def load_controllers(load_dir):
    # Load sub-controllers
    controller_list = []

    for controller_dir in os.listdir(load_dir):
        controller_load_path = os.path.join(load_dir, controller_dir)
        if os.path.isdir(controller_load_path):
            controller = MiniGridController(0, load_dir=controller_load_path)
            controller_list.append(controller)

    # re-order the controllers by index
    reordered_list = []
    for i in range(len(controller_list)):
        for controller in controller_list:
            if controller.controller_ind == i:
                reordered_list.append(controller)
    controller_list = reordered_list

    return controller_list


# Load results from previous training
load_dir = get_load_dir('2021-05-26_22-35-16_minigrid_labyrinth_copy')  # (final trained controllers by Neary et al.)
results = Results(load_dir=load_dir)
print('is this taking long?')
rseed = results.data['random_seed']
torch.manual_seed(rseed)
random.seed(rseed)
np.random.seed(rseed)
print(f'Random seed: {rseed}.')
# Setup environments
# env0 = MazeVariant(agent_start_states=[(1, 1, 0)], slip_p=0.1, variant=5)
env1 = MazeVariant(agent_start_states=[(1, 1, 0)], slip_p=0.1, variant=0)

# Load controllers
controller_list = load_controllers(load_dir)
print('or this?')
hlmdp = HLMDP([(1, 1, 0)], env1.goal_states, controller_list)
last_compositional_policy = results.data['composition_policy'][list(results.data['composition_policy'])[-1]]
last_reach_prob = results.data['composition_predicted_success_prob'][
    list(results.data['composition_predicted_success_prob'])[-1]]
meta_controller = MetaController(last_compositional_policy, hlmdp.controller_list, hlmdp.state_list)

unflagged = controller_list[:4] + controller_list[5:]

print(f'state_list: {hlmdp.state_list}')
print(f'len(controller_list): {len(controller_list)}')
print(f'len(unflagged): {len(unflagged)}')
print(f'last_compositional_policy: {last_compositional_policy}')
# Flag controller 4 as unverified
# controller_flags = [False] * len(controller_list)
# controller_flags[4] = True

hlmdp.controller_list = unflagged


# (Re)synthesize meta-controller without using flagged controllers
policy, reach_prob, feasible_flag = hlmdp.solve_max_reach_prob_policy()
print(f'policy: {policy}, reach_prob: {reach_prob}, feasible_flag: {feasible_flag}')

# def resynthesize(env, meta_ctrl, ctrl_list):
# policy, reach_prob, feasible_flag =
