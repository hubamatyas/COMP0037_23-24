#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Q3e: 

    # Grid-based search to investigate effect of changing parameters
    # Values you can change:
    theta_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    max_steps_values =  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # List to store the dictionaries of results
    results = []
    # Initialize a matrix to store the results
    results_matrix = np.zeros((len(theta_values), len(max_steps_values)))

    for i, theta in enumerate(theta_values):
        for j, max_steps in enumerate(max_steps_values):
            # Create the policy iterator
            policy_solver = PolicyIterator(airport_environment)
            policy_solver.set_theta(theta)
            policy_solver.set_max_policy_evaluation_steps_per_iteration(max_steps)

            # Set up initial state
            policy_solver.initialize()
            
            # Compute the solution
            start_time = time.time()
            v, pi, _, _ = policy_solver.solve_policy()
            end_time = time.time()

            computation_time = end_time - start_time
            
            # Store the results
            results.append({
                'theta': theta,
                'max_steps': max_steps,
                'computation time': computation_time})
            
            # Store the results in the matrix
            results_matrix[i, j] = computation_time
    
    # Use log scale for theta
    theta_scale = np.log10(np.array(theta_values))
    max_steps_scale = np.array(max_steps_values)

    # Create mesh grids for theta and max_steps
    Theta, MaxSteps = np.meshgrid(theta_scale, max_steps_values)

    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(Theta, MaxSteps, results_matrix.T, cmap='plasma', edgecolor='black', alpha=0.75)

    # Find the minimum computation time and its corresponding theta and max_steps for optimal point plotting
    min_time_index = np.unravel_index(np.argmin(results_matrix, axis=None), results_matrix.shape)
    optimal_theta = theta_values[min_time_index[0]]  # Original theta value
    optimal_max_steps = max_steps_values[min_time_index[1]]
    optimal_time = results_matrix[min_time_index]

    # Plot the optimal point in red with log10(theta) adjustment
    optimal_theta_log_scale = np.log10(optimal_theta)
    ax.scatter(optimal_theta_log_scale, optimal_max_steps, optimal_time, color='r', s=50, label='Optimal Time')

    # Labels, legends and titles
    ax.set_xlabel('Log10(Theta)')
    ax.set_ylabel('Max Steps per Iteration')
    ax.set_zlabel('Computation Time (seconds)')
    ax.set_title('3D Plot of Policy Performance')
    ax.legend()

    # Color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # After grid-based search, we find the lowest max_steps that is optimal
    # Fixed values
    theta = 1e-5
    threshold = 1e-4

    # Values you can change:
    max_steps_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # Re-initialise the list to store the dictionaries of results
    results = []

    # Initial slow run to calculate optimal v and pi
    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_theta(1e-10)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(500)

    # Set up initial state
    policy_solver.initialize()

    optimal_v, optimal_pi, _, _ = policy_solver.solve_policy()

    # for i, theta in enumerate(theta_values):
    for j, max_steps in enumerate(max_steps_values):
        # Create the policy iterator
        policy_solver = PolicyIterator(airport_environment)
        policy_solver.set_theta(theta)
        policy_solver.set_max_policy_evaluation_steps_per_iteration(max_steps)

        # Set up initial state
        policy_solver.initialize()
        
        # Compute the solution
        start_time = time.time()
        v, pi, _, _ = policy_solver.solve_policy()
        end_time = time.time()

        computation_time = end_time - start_time
        
        # Store the results
        results.append({
            'theta': theta,
            'max_steps': max_steps,
            'computation time': computation_time,
            'delta': np.nanmax(np.abs(optimal_v._values - v._values))})

for result in results:
    print(f"Theta: {result['theta']}")
    print(f"Max Steps: {result['max_steps']}")
    print(f"Computation Time (seconds): {result['computation time']}")
    print(f"Delta: {result['delta']}")
    print(f"Optimal?: {result['delta'] < threshold}")
    print()
