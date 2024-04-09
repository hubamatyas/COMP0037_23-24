#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario, full_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np
from matplotlib import pyplot as plt

# Function to calculate rmse
def calculate_rmse(value_function_td, value_function_benchmark):
    value_function_td = np.nan_to_num(value_function_td, nan=100)
    value_function_benchmark = np.nan_to_num(value_function_benchmark, nan=100)
    return np.sqrt(np.nanmean((value_function_td - value_function_benchmark) ** 2))

if __name__ == '__main__':
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)  

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    pe.evaluate()

    # Benchmark/ground truth value function from policy evaluation
    benchmark_value_function = pe.value_function()._values

    # Define initial conditions
    iterations = 500 # Number of iterations to run in a TD learning scenario
    td_iterations = 10  # Number of TD learning iterations for averaging and smoothing

    initial_alpha = 0.1 # Initial alpha for the decaying alpha scenario
    decay_rate = 0.05  # Decay rate for the decaying alpha scenario

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Scenarios to compare
    scenarios = {
        'Constant Alpha = 0.0001': 0.0001,
        'Constant Alpha = 0.0005': 0.0005,
        'Constant Alpha = 0.05': 0.05,
        'Constant Alpha = 0.5': 0.5,
        'Decaying Alpha': 'decaying'
    }

    # Negative scenario
    # scenarios = {
    #     'Negative Alpha = -0.0001': -0.001,
    #     'Constant Alpha = 0.0001': 0.005,
    # }

    for scenario, alpha in scenarios.items():

        td_predictor = TDPolicyPredictor(env)
        td_predictor.set_experience_replay_buffer_size(64)
        td_predictor.set_target_policy(pi)

        aggregate_rmse_values = []
        for iteration in range(iterations):
            iteration_errors = []
            if alpha == 'decaying':
                # Adjust alpha for the decaying scenario
                current_alpha = initial_alpha * np.exp(-decay_rate * iteration)
            else:
                # Use the provided constant alpha value
                current_alpha = alpha
            td_predictor.set_alpha(current_alpha)
            
            for _ in range(td_iterations):
                # TD evaluation with current_alpha
                td_predictor.evaluate()
                value_function_td = td_predictor.value_function()._values
                rmse = calculate_rmse(value_function_td, benchmark_value_function)
                iteration_errors.append(rmse)
                
            # Average errors across runs for the current iteration
            aggregate_rmse_values.append(np.mean(iteration_errors))

        # Print the final RMSE for the current scenario, for a table of results
        print(f'Final RMSE for {scenario}: {aggregate_rmse_values[-1]}')

        # Plot the error trajectory for the current scenario
        if alpha == 'decaying':
            plt.plot(range(iterations), aggregate_rmse_values, label=scenario + f' (Initial Alpha = {initial_alpha}, Decay Rate = {decay_rate})', alpha=1.0)
        else:
            plt.plot(range(iterations), aggregate_rmse_values, label=scenario, alpha=0.3)

    # Finalize the plot
    plt.xlabel('Iteration')
    plt.ylabel('Log(RMSE)')
    plt.title('TD Learning Comparison of Log(RMSE) Iteration-by-Iteration for Constant vs Negative Alpha')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Optional: Apply log scale to y-axis if there are large variances in rmse values
    plt.show()
