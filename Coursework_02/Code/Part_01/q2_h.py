#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

from collections import defaultdict
import math

from common.scenarios import corridor_scenario
from common.airport_map_drawer import AirportMapDrawer


from td.sarsa import SARSA
from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np
import matplotlib.pyplot as plt

# Calculate delta change in the value function
def calculate_value_function_delta(previous_v, current_v):
    delta = np.abs(previous_v - current_v)
    return np.nanmean(delta)  # Assuming that 'nan' values are present and should be ignored

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Create learners
    learners = [SARSA(env), QLearner(env)]
    policies = [env.initial_policy() for _ in learners]

    num_runs = 5
    num_iter = 10000
    smoothing_window = 50

    cumulative_value_function_deltas = defaultdict(list)
    cumulative_sum_of_rewards = defaultdict(list)
    
    avg_value_function_deltas = defaultdict(list)
    avg_sum_of_rewards = defaultdict(list)
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Initialise a dictionary to store the results for this run
        value_function_deltas = {learner.__class__.__name__: [] for learner in learners}
        sum_of_rewards = {learner.__class__.__name__: [] for learner in learners}

        # Reset the environment and the policy learner
        for i, learner in enumerate(learners):
            policies[i].set_epsilon(1)
            learner.set_initial_policy(policies[i])
            learner.set_alpha(0.1)
            learner.set_experience_replay_buffer_size(64)
            learner.set_number_of_episodes(32)

        # Run the learning loop for the specified number of iterations
        for i in range(num_iter):
            for learner, pi in zip(learners, policies):
                previous_value_function = np.copy(learner.value_function()._values)
                _, _, episode_rewards = learner.find_policy()
                current_value_function = learner.value_function()._values

                # Calculate the delta change and sum of rewards
                delta = calculate_value_function_delta(previous_value_function, current_value_function)
                total_rewards = np.sum(episode_rewards)

                # Store the results
                learner_name = learner.__class__.__name__
                value_function_deltas[learner_name].append(delta)
                sum_of_rewards[learner_name].append(total_rewards)

                epsilon = 1 / math.sqrt(1 + 0.25 * i)
                pi.set_epsilon(epsilon)

        # Store the results for this run
        for learner in learners:
            learner_name = learner.__class__.__name__
            cumulative_value_function_deltas[learner_name].append(value_function_deltas[learner_name])
            cumulative_sum_of_rewards[learner_name].append(sum_of_rewards[learner_name])
            
    # Calculate the average results
    for learner in learners:
        print(f"Calculating average results for {learner.__class__.__name__}")
        learner_name = learner.__class__.__name__
        avg_value_function_deltas[learner_name] = np.mean(cumulative_value_function_deltas[learner_name], axis=0)
        avg_sum_of_rewards[learner_name] = np.mean(cumulative_sum_of_rewards[learner_name], axis=0)

    # Smooth the results by averaging over 50 iterations
    for learner_name in avg_value_function_deltas:
        avg_value_function_deltas[learner_name] = [np.mean(avg_value_function_deltas[learner_name][i:i+smoothing_window]) for i in range(0, len(avg_value_function_deltas[learner_name]), smoothing_window)]
        avg_sum_of_rewards[learner_name] = [np.mean(avg_sum_of_rewards[learner_name][i:i+smoothing_window]) for i in range(0, len(avg_sum_of_rewards[learner_name]), smoothing_window)]

    # Create tables for the mean and variance of the delta and sum of rewards of SARSA and Q-Learning from the 50th iteration onwards
    print("Mean and variance of the delta change in the value function and sum of rewards from the 50th iteration onwards")
    for learner_name in avg_value_function_deltas:
        print(f"{learner_name}:")
        print(f"Mean delta: {np.mean(avg_value_function_deltas[learner_name][50:])}")
        print(f"Variance delta: {np.var(avg_value_function_deltas[learner_name][50:])}")
        print(f"Mean sum of rewards: {np.mean(avg_sum_of_rewards[learner_name][50:])}")
        print(f"Variance sum of rewards: {np.var(avg_sum_of_rewards[learner_name][50:])}")
        print()

    # Plot for the smoothed results
    plt.figure(figsize=(12, 6))
    for learner_name in avg_value_function_deltas:
        plt.plot(avg_value_function_deltas[learner_name], label=f"{learner_name}")

    plt.xlabel("Iteration")
    plt.ylabel("Average Delta Change in Value Function")
    plt.yscale('log')  # Apply log scale to see differences clearly
    plt.legend()
    plt.title(f'Comparison of SARSA and Q-Learning: Value Function Delta (Averaged over {num_runs} runs, smoothed over {smoothing_window} iterations)')
    plt.show()

    plt.figure(figsize=(12, 6))
    for learner_name in avg_sum_of_rewards:
        plt.plot(avg_sum_of_rewards[learner_name], label=f"{learner_name}")

    plt.xlabel("Iteration")
    plt.ylabel("Average Sum of Rewards")
    plt.legend()
    plt.title(f'Comparison of SARSA and Q-Learning: Sum of Rewards (Averaged over {num_runs} runs, smoothed over {smoothing_window} iterations)')
    plt.show()
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    for learner_name in value_function_deltas:
        plt.plot(value_function_deltas[learner_name], label=f"{learner_name}")

    plt.xlabel("Iteration")
    plt.ylabel("Average Delta Change in Value Function")
    plt.yscale('log')  # Apply log scale to see differences clearly
    plt.legend()
    plt.title(f'Comparison of SARSA and Q-Learning: Value Function Delta')
    plt.show()

    plt.figure(figsize=(12, 6))
    for learner_name in sum_of_rewards:
        plt.plot(sum_of_rewards[learner_name], label=f"{learner_name}")

    plt.xlabel("Iteration")
    plt.ylabel("Sum of Rewards")
    plt.legend()
    plt.title(f'Comparison of SARSA and Q-Learning: Sum of Rewards')
    plt.show()
