#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

# Calculate delta change in the value function
def calculate_value_function_delta(previous_v, current_v):
    delta = np.abs(previous_v - current_v)
    return np.nanmean(delta)  # Assuming that 'nan' values are present and should be ignored

if __name__ == '__main__':

    do_test = False

    if do_test:
        num_iters = [10, 20, 100, 200]
        max_iter = num_iters[-1]
        q_runs = 2
    else:
        num_iters = [100, 500, 1000, 5000, 10000, 50000, 100000]
        max_iter = num_iters[-1]
        q_runs = 5

    overall_results = {'avg_value_function_deltas': [], 'avg_sum_of_rewards': []}

    for run in range(q_runs):
        results_per_run = {'value_function_deltas': [], 'sum_of_rewards': []}
        print(f"Run {run + 1}/{q_runs}")
        # Reset the environment and the policy learner
        airport_map, drawer_height = corridor_scenario()

        # Show the scenario        
        airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
        airport_map_drawer.update()

        # Create the environment
        env = LowLevelEnvironment(airport_map)

        pi = env.initial_policy()
        pi.set_epsilon(1) # Reset epsilon to 1 for e-greedy policy
        
        policy_learner = QLearner(env)   
        policy_learner.set_initial_policy(pi)

        # These values worked okay for me.
        policy_learner.set_alpha(0.1)
        policy_learner.set_experience_replay_buffer_size(64)
        policy_learner.set_number_of_episodes(32)

        # The drawers for the state value and the policy
        value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
        greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)

        # Run the learning loop for the specified number of iterations
        for i in range(max_iter):
            print(f"Iteration {i+1} for {max_iter} iterations")

            previous_value_function = np.copy(policy_learner.value_function()._values)
            
            # Run an episode of Q-learning
            _, _, episode_rewards = policy_learner.find_policy()
            current_value_function = policy_learner.value_function()._values
        
            # Calculate the delta change and sum of rewards
            delta_change = calculate_value_function_delta(previous_value_function, current_value_function)
            sum_of_rewards = np.sum(episode_rewards)
        
            # Store the results
            results_per_run['value_function_deltas'].append(delta_change)
            results_per_run['sum_of_rewards'].append(sum_of_rewards)
        
            # Update exploration rate
            epsilon = 1 / math.sqrt(1 + 0.25 * i)
            pi.set_epsilon(epsilon)
        
            # Update the drawers
            value_function_drawer.update()
            greedy_optimal_policy_drawer.update()

        # Add the results to the overall results
        overall_results['avg_value_function_deltas'].append(results_per_run['value_function_deltas'])
        overall_results['avg_sum_of_rewards'].append(results_per_run['sum_of_rewards'])

    # Save the screenshots for the value function and policy
    if not do_test:
        value_function_drawer.fancy_save_screenshot(f"q2_d2_value_function_run_{run + 1}_iterations_{max_iter}.pdf")
        greedy_optimal_policy_drawer.save_screenshot(f"q2_d2_policy_run_{run + 1}_iterations_{max_iter}.pdf")

    # Calculate the average results
    overall_results['avg_value_function_deltas'] = np.mean(overall_results['avg_value_function_deltas'], axis=0)
    overall_results['avg_sum_of_rewards'] = np.mean(overall_results['avg_sum_of_rewards'], axis=0)

    # Create a table for n number of iterations, of the mean and variance of the delta change and sum of rewards up to the nth iteration
    print(f"{'Num Iterations':<15}{'Mean Delta Change':<20}{'Variance Delta Change':<25}{'Mean Sum of Rewards':<25}{'Variance Sum of Rewards':<25}")
    for num_iter in num_iters:
        mean_delta = np.mean(overall_results['avg_value_function_deltas'][:num_iter])
        var_delta = np.var(overall_results['avg_value_function_deltas'][:num_iter])
        mean_rewards = np.mean(overall_results['avg_sum_of_rewards'][:num_iter])
        var_rewards = np.var(overall_results['avg_sum_of_rewards'][:num_iter])
        print(f"{num_iter:<15}{mean_delta:<20}{var_delta:<25}{mean_rewards:<25}{var_rewards:<25}")

    # Create a table for n number of iterations, of the mean and variance of the delta change and sum of rewards up to the nth iteration excluding the first 50 iterations
    print(f"{'Num Iterations':<15}{'Mean Delta Change':<20}{'Variance Delta Change':<25}{'Mean Sum of Rewards':<25}{'Variance Sum of Rewards':<25}")
    for num_iter in num_iters:
        mean_delta = np.mean(overall_results['avg_value_function_deltas'][50:num_iter])
        var_delta = np.var(overall_results['avg_value_function_deltas'][50:num_iter])
        mean_rewards = np.mean(overall_results['avg_sum_of_rewards'][50:num_iter])
        var_rewards = np.var(overall_results['avg_sum_of_rewards'][50:num_iter])
        print(f"{num_iter:<15}{mean_delta:<20}{var_delta:<25}{mean_rewards:<25}{var_rewards:<25}")


    deltas = overall_results['avg_value_function_deltas']
    sum_of_rewards = overall_results['avg_sum_of_rewards']
    

    # Smooth the results by averaging over a window of iterations
    smoothing_window = 2 if do_test else 200
    deltas = [np.mean(deltas[i:i+smoothing_window]) for i in range(0, len(deltas), smoothing_window)]
    sum_of_rewards = [np.mean(sum_of_rewards[i:i+smoothing_window]) for i in range(0, len(sum_of_rewards), smoothing_window)]

    # Set the x-ticks positions and labels to reflect the original iteration count
    tick_labels = [0, int(max_iter/5), int(max_iter/5*2), int(max_iter/5*3), int(max_iter/5*4), max_iter]
    tick_positions = [i // smoothing_window for i in tick_labels]

    # Plot the delta change in the value function
    plt.figure(figsize=(12, 6))
    plt.plot(deltas)
    for num_iter in num_iters[:-1]:
        plt.axvline(x=num_iter//smoothing_window, color='red', linestyle='--', linewidth=1)

    # Set the x-ticks positions and labels to reflect the original iteration count
    plt.xticks(tick_positions, tick_labels) 
    plt.xlabel('Iteration')
    plt.ylabel('Log Delta in the Value Function')
    plt.yscale('log')
    plt.title(f'Smoothed log delta in the value function for {max_iter} iterations averaged over {q_runs} runs with key checkpoints')

    # Plot a second y-axis to show the epsilon
    ax2 = plt.twinx()
    ax2.plot([1 / math.sqrt(1 + 0.25 * i) for i in range(max_iter//smoothing_window)], color='purple', alpha=0.8)
    ax2.plot
    ax2.set_ylabel('Epsilon')

    # Create custom legend handles
    handle1 = mlines.Line2D([], [], color='#1f77b4', label='Log Delta Change in Value Function')
    handle2 = mlines.Line2D([], [], color='purple', label='Epsilon')
    handle3 = mlines.Line2D([], [], color='red', linestyle='--', label=f'Checkpoints = {num_iters[:-1]}')

    # Create a single legend box containing all the handles
    plt.legend(handles=[handle1, handle2, handle3])

    plt.show()

    # Plot the sum of rewards
    plt.figure(figsize=(12, 6))
    plt.plot(sum_of_rewards)
    for num_iter in num_iters[:-1]:
        plt.axvline(x=num_iter//smoothing_window, color='red', linestyle='--', linewidth=1)
    
    # Rescale the x-axis to show the number of iterations
    plt.xticks(tick_positions, tick_labels) 
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Rewards')
    plt.title(f'Smoothed sum of rewards for {max_iter} iterations averaged over {q_runs} runs with key checkpoints')

    # Plot a second y-axis to show the epsilon
    ax2 = plt.twinx()
    ax2.plot([1 / math.sqrt(1 + 0.25 * i) for i in range(max_iter//smoothing_window)], color='purple', alpha=0.8)
    ax2.set_ylabel('Epsilon')

    # Edit custom legend handles
    handle1 = mlines.Line2D([], [], color='#1f77b4', label='Sum of Rewards')

    # Create a single legend box containing all the handles
    plt.legend(handles=[handle1, handle2, handle3], loc='center right')

    plt.show()
