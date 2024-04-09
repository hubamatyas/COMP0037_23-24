#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
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
import matplotlib.pyplot as plt


# Function to calculate RMSE
def calculate_rmse(value_function_td, value_function_benchmark):
    value_function_td = np.nan_to_num(value_function_td, nan=100)
    value_function_benchmark = np.nan_to_num(value_function_benchmark, nan=100)
    return np.sqrt(np.nanmean((value_function_td - value_function_benchmark) ** 2))

if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)

    # Initial conditions
    initial_broad = True
    narrowed_down = False

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

    # Define alpha and iteration values for grid search
    if initial_broad:
        alpha_values = np.linspace(0.01, 0.5, 20)
        iteration_values = np.linspace(10, 100, 10, dtype=int)
    elif narrowed_down:
        alpha_values = np.linspace(0.1, 1.0, 10)
        iteration_values = np.linspace(100, 600, 6, dtype=int)

    # Fixed number of TD learning iterations for averaging
    td_iterations = 20

    errors = np.zeros((len(alpha_values), len(iteration_values)))
    
    num_values = len(alpha_values)
    
    td_predictors: list[TDPolicyPredictor] = [None] * num_values
    td_drawers: list[ValueFunctionDrawer] = [None] * num_values

    # Grid search over alpha values and iteration counts
    for i, alpha in enumerate(alpha_values):
        for j, iteration in enumerate(iteration_values):
            # Temporary list to store rmse values for each repetition
            rmse_values = []

            for _ in range(td_iterations):
                # Initialize TD Policy Predictor
                td_predictor = TDPolicyPredictor(env)
                td_predictor.set_experience_replay_buffer_size(64)
                td_predictor.set_alpha(alpha)
                td_predictor.set_target_policy(pi)
                
                # Run TD learning for the specified number of iterations
                for _ in range(iteration):
                    td_predictor.evaluate()

                value_function_td = td_predictor.value_function()._values

                # Calculate RMSE between TD predictor and benchmark for the current repetition
                rmse = calculate_rmse(value_function_td, benchmark_value_function)
                rmse_values.append(rmse)

            errors[i][j] = np.mean(rmse_values)

            print(f"Alpha: {alpha}, Iterations: {iteration}, RMSE: {rmse}")

    # Plot a meshgrid of RMSE values

    Alpha, Iterations = np.meshgrid(alpha_values, iteration_values)
    Alpha, Iterations_log = np.meshgrid(alpha_values, np.log10(iteration_values))
    errors = np.array(errors)
    log_errors = np.log(errors + 1)

    fig_linear = plt.figure(figsize=(10, 8))
    ax_linear = fig_linear.add_subplot(111, projection='3d')

    # Plot the meshgrid
    surf_linear = ax_linear.plot_surface(Alpha, Iterations, log_errors.T, cmap='viridis', edgecolor='black', alpha=0.75)

    # Add labels and titles
    ax_linear.set_xlabel('Alpha')
    ax_linear.set_ylabel('Number of Iterations')
    ax_linear.set_zlabel('RMSE')
    ax_linear.set_title('RMSE of TD Learning Against Alpha and Number of Iterations')

    fig_linear.colorbar(surf_linear, shrink=0.5, aspect=5, pad=0.1)

    plt.show()

    fig_log = plt.figure(figsize=(10, 8))
    ax_log = fig_log.add_subplot(111, projection='3d')

    # Plot the meshgrid
    surf_log = ax_log.plot_surface(Alpha, Iterations_log, log_errors.T, cmap='viridis', edgecolor='black', alpha=0.75)

    # Add labels and titles
    ax_log.set_xlabel('Alpha')
    ax_log.set_ylabel('Log10(Iterations)')
    ax_log.set_zlabel('RMSE')
    ax_log.set_title('RMSE of TD Learning')

    # Set the y-axis ticks to represent the original iterations values
    ax_log.set_yticks(np.log10(iteration_values))  # Set ticks at the positions of the log-scaled iterations
    ax_log.set_yticklabels(iteration_values)  # But label them with the original iterations values

    fig_log.colorbar(surf_log, shrink=0.5, aspect=5, pad=0.1)

    plt.show()
    