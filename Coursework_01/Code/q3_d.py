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

if __name__ == '__main__':
    p_values = [0.3, 0.6, 0.9, 1]

    for p in p_values:
        # Get the map for the scenario
        airport_map, drawer_height = full_scenario()

        # Add high traversability costs
        # airport_map.set_use_cell_type_traversability_costs(True)
        
        # Set up the environment for the robot driving around
        airport_environment = LowLevelEnvironment(airport_map)
        
        # Q3d:
        # Configure the process model using different probabilities
        airport_environment.set_nominal_direction_probability(p)

        # Note that you can create multiple instances of the same object, with different
        # settings, and run them in the same programme. Therefore, you do not need to
        # create lots of separate scripts to run the code.

        # Create the policy iterator
        policy_solver = PolicyIterator(airport_environment)

        # Set up initial state
        policy_solver.initialize()
            
        # Bind the drawer with the solver
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        
        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)
            
        # Compute the solution
        v, pi = policy_solver.solve_policy()
        
        # Save screen shot; this is in the current directory
        p_str = str(p).replace(".", "_")
        policy_drawer.save_screenshot(f"policy_p_{p_str}.pdf")
        value_function_drawer.save_screenshot(f"value_p_{p_str}.pdf")
        
        # Wait for a key press
        value_function_drawer.wait_for_key_press()
