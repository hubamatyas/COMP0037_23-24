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

# Function to calculate MSE
def calculate_mse(value_function_td, value_function_benchmark):
    value_function_td = np.nan_to_num(value_function_td, nan=100)
    value_function_benchmark = np.nan_to_num(value_function_benchmark, nan=100)
    return np.mean((value_function_td - value_function_benchmark) ** 2)

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
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    pe.evaluate()
    v_pe.update()  
    v_pe.update()  

    # Benchmark value function from policy evaluation
    benchmark_value_function = pe.value_function()._values

    # Define your alphas and iterations
alphas = [0.001, 0.1, 0.2, 0.5, 1.0]  # Add more alpha values as needed
iterations = 400
td_iterations = 20  # Number of TD learning iterations for averaging

# Initialize the plot
plt.figure(figsize=(12, 6))

# Loop through the alphas and plot the error trajectory for each one
for alpha in alphas:
    # Initialize TD Policy Predictor for the current alpha
    td_predictor = TDPolicyPredictor(env)
    td_predictor.set_experience_replay_buffer_size(64)
    td_predictor.set_alpha(alpha)
    td_predictor.set_target_policy(pi)

    # Aggregate errors for each iteration across multiple runs
    aggregate_mse_values = []

    # Run TD learning for the specified number of iterations and average the MSE
    for iteration in range(iterations):
        iteration_errors = []
        for _ in range(td_iterations):
            td_predictor.evaluate()
            value_function_td = td_predictor.value_function()._values
            mse = calculate_mse(value_function_td, benchmark_value_function)
            iteration_errors.append(mse)
        # Average errors across runs for current iteration
        aggregate_mse_values.append(np.mean(iteration_errors))

    # Plot the error trajectory for the current alpha
    plt.plot(range(iterations), aggregate_mse_values, label=f'Alpha = {alpha}')

# Finalize the plot
plt.xlabel('Iteration Number')
plt.ylabel('MSE')
plt.title('Error Trajectory of TD Learning Over Iterations for Multiple Alphas')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Optional: Apply log scale to y-axis if there are large variances in MSE values
plt.show()