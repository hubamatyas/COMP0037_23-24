#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import numpy as np
import matplotlib.pyplot as plt

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    
    full_time_list = []
    full_steps_list = []
    iteration_time_list = []
    iteration_steps_list = []
    for i in range(40):
        time_list, updates_list = policy_learner.find_policy()

        full_time_list.extend(time_list)
        full_steps_list.extend(updates_list)

        iteration_time_list.append(time_list[-1])
        iteration_steps_list.append(updates_list[-1])
        
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()

        epsilon = 1 / math.sqrt(1 + 0.25 * i)
        pi.set_epsilon(epsilon)

        print(f"Iteration {i}: epsilon={epsilon}, alpha={policy_learner.alpha()}")
        print(f"Total time taken for iteration {i}: {sum(time_list)}")
        print(f"Total number of steps in iteration {i}: {sum(updates_list)}\n")

    # plot number of updates as episodes progress
    plt.plot(full_steps_list)
    plt.xlabel('Episode')
    plt.ylabel('Number of steps in newly generated episode')

    # plot time taken for each episode
    plt.figure()
    plt.plot(full_time_list, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Time taken')

    plt.show()

        