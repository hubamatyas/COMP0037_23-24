#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np
import matplotlib.pyplot as plt
import copy

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
    
    # Record the errors for plotting the performance of first and every visit later
    errors = []

    # Run the experiment for first-visit and every-visit
    for is_first_visit in [True, False]:
        name = "First" if is_first_visit else "Every"

        # On policy MC predictor
        mcpp = OnPolicyMCPredictor(env)
        mcpp.set_target_policy(pi)
        mcpp.set_experience_replay_buffer_size(64)
        
        # Q1b: Experiment with this value
        mcpp.set_use_first_visit(is_first_visit)
        
        v_mcpp = ValueFunctionDrawer(mcpp.value_function(), drawer_height)
        
        # Off policy MC predictor
        mcop = OffPolicyMCPredictor(env)
        mcop.set_target_policy(pi)
        mcop.set_experience_replay_buffer_size(64)
        b = env.initial_policy()
        b.set_epsilon(0.2)
        mcop.set_behaviour_policy(b)
        
        # Q1b: Experiment with this value
        mcop.set_use_first_visit(is_first_visit)
        v_mcop = ValueFunctionDrawer(mcop.value_function(), drawer_height)
        
        # Record the intermediate value functions for plotting the error later
        mcpp_intermediate_value_functions = []
        mcop_intermediate_value_functions = []

        episodes = 1000
        for e in range(episodes):
            mcpp.evaluate()
            v_mcpp.update()
            mcpp_intermediate_value_functions.append(copy.deepcopy(mcpp.value_function()))

            mcop.evaluate()
            v_mcop.update()
            mcop_intermediate_value_functions.append(copy.deepcopy(mcop.value_function()))

        # v_pe.fancy_save_screenshot(f"q1_b_{name}_pe.pdf", title="GT Policy Evaluator")
        # v_mcpp.fancy_save_screenshot(f"q1_b_{name}_mc-on_pe.pdf", title=f"{name}-visit: On Policy MC Predictor")
        # v_mcop.fancy_save_screenshot(f"q1_b_{name}_mc-off_pe.pdf", title=f"{name}-visit: Off Policy MC Predictor")

        # Necessary to avoid nan result when comparing with np.abs
        pe.value_function()._values = np.nan_to_num(pe.value_function()._values, nan=100)
        mcpp.value_function()._values = np.nan_to_num(mcpp.value_function()._values, nan=100)
        mcop.value_function()._values = np.nan_to_num(mcop.value_function()._values, nan=100)

        # Calculate and store MSE (change to np.mean for average, etc.)
        mcpp_errors = []
        mcop_errors = []
        for i in range(episodes):
            mcpp_value_function = np.nan_to_num(mcpp_intermediate_value_functions[i]._values, nan=100)
            mcop_value_function = np.nan_to_num(mcop_intermediate_value_functions[i]._values, nan=100)
            mcpp_errors.append(np.mean((pe.value_function()._values - mcpp_value_function)**2))
            mcop_errors.append(np.mean((pe.value_function()._values - mcop_value_function)**2))
            
        # plt.plot(mcpp_errors, label=f"{name}-visit: On Policy MC Predictor")
        # plt.plot(mcop_errors, label=f"{name}-viist: Off Policy MC Predictor")
        errors.append(mcpp_errors)
        errors.append(mcop_errors)

        # Calculate the sum of absolute errors, max absolute error, and average absolute error for off-policy
        sum_off_policy_error = np.sum(np.abs(pe.value_function()._values - mcop.value_function()._values))
        max_off_policy_error = np.max(np.abs(pe.value_function()._values - mcop.value_function()._values))
        avg_off_policy_error = np.mean(np.abs(pe.value_function()._values - mcop.value_function()._values))
        mean_squared_error = np.mean((pe.value_function()._values - mcop.value_function()._values)**2)

        print(f"{name}-visit: Off-policy error after {episodes} episodes:")
        print("MAX:", max_off_policy_error)
        print("MSE:", mean_squared_error)
        print()

        # Calculate the sum of absolute errors, max absolute error, and average absolute error for on-policy
        sum_on_policy_error = np.sum(np.abs(pe.value_function()._values - mcpp.value_function()._values))
        max_on_policy_error = np.max(np.abs(pe.value_function()._values - mcpp.value_function()._values))
        avg_on_policy_error = np.mean(np.abs(pe.value_function()._values - mcpp.value_function()._values))
        mean_squared_error = np.mean((pe.value_function()._values - mcpp.value_function()._values)**2)

        print(f"{name}-vist: On-policy error after {episodes} episdoes:")
        print("MAX:", max_on_policy_error)
        print("MSE:", mean_squared_error)
        print()

        # Sample way to generate outputs    
        # v_pe.save_screenshot("q1_b_every_truth_pe.pdf")
        # v_mcop.save_screenshot("q1_b_every_mc-off_pe.pdf")
        # v_mcpp.save_screenshot("q1_b_every_mc-on_pe.pdf")

    
    # Plot the errors for both first-visit and every-visit and on-policy and off-policy
    # overtime, as episodes increase
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(errors[0], label="First-visit: On Policy MC Predictor")
    ax.plot(errors[1], label="First-visit: Off Policy MC Predictor")
    ax.plot(errors[2], label="Every-visit: On Policy MC Predictor")
    ax.plot(errors[3], label="Every-visit: Off Policy MC Predictor")
    ax.set_xlabel("Episode")
    ax.set_xlim(right=500)
    ax.set_ylabel("Mean Squared Error")
    ax.legend()
    # plt.savefig("q1_b_errors.pdf")

    new_fig = plt.figure()
    ax = new_fig.add_subplot(111)
    ax.plot(errors[0], label="First-visit: On Policy MC Predictor")
    ax.plot(errors[1], label="First-visit: Off Policy MC Predictor")
    ax.plot(errors[2], label="Every-visit: On Policy MC Predictor")
    ax.plot(errors[3], label="Every-visit: Off Policy MC Predictor")
    ax.set_xlim(left=500)
    ax.set_ylim(top=2, bottom=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()
    # plt.savefig("q1_b_errors_500.pdf")
    plt.show()