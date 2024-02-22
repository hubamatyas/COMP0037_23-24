#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import time
import numpy as np

if __name__ == '__main__':

    # Parameters
    theta = 10e-5
    max_steps = 16
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()

    # Add high traversability costs
    # airport_map.set_use_cell_type_traversability_costs(True)
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)

    # Set Theta and Max_Steps
    policy_solver.set_theta(theta)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(max_steps)
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    # Compute the solution
    start_time = time.time()
    policy_v, policy_pi, policy_iteration_step, total_policy_evaluation_iterations = policy_solver.solve_policy()
    end_time = time.time()

    policy_computation_time = end_time - start_time
    
    # Save screen shot; this is in the current directory
    # policy_drawer.save_screenshot("policy_iteration_results.pdf")

    # Number of iteration and computation time results
    print(f"Number of policy iteration steps: {policy_iteration_step}, where each step is 1 policy evaluation + 1 policy improvement")
    print(f"Total number of policy evaluation iterations: {total_policy_evaluation_iterations}")
    print(f"Average number of policy evaluation iterations per policy iteration step: {total_policy_evaluation_iterations/policy_iteration_step}")
    print(f"Computational time (seconds): {policy_computation_time}")

    # Wait for a key press
    value_function_drawer.wait_for_key_press()

    # Q3i: Add code to evaluate value iteration down here.
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Add high traversability costs
    # airport_map.set_use_cell_type_traversability_costs(True)

    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Create the policy iterator
    policy_solver = ValueIterator(airport_environment)
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    # Compute the solution
    start_time = time.time()
    value_v, value_pi, value_iteration_step = policy_solver.solve_policy()
    end_time = time.time()

    value_computation_time = end_time - start_time
    
    # Save screen shot; this is in the current directory
    # policy_drawer.save_screenshot("value_iterator_results.pdf")
    
    # Number of iteration and computation time results
    print(f"Number of value iteration steps: {value_iteration_step}")
    print(f"Computational time (seconds): {value_computation_time}")

    # Is there a significant difference between the values?
    delta = np.nanmax(np.abs(policy_v._values - value_v._values))
    print(f"Max difference between policy and value iteration results: {delta}")

    # Wait for a key press
    value_function_drawer.wait_for_key_press()

