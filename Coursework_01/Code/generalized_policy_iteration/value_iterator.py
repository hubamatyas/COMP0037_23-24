'''
Created on 29 Jan 2022

@author: ucacsjj
'''
import copy
from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):

        # This method returns no value.
        # The method updates self._pi

        environment = self._environment
        map = environment.map()

        # Execute the loop at least once
        iteration = 0
        while True:
            delta = 0
            for x in range(map.width()):
                for y in range(map.height()):
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue
                    cell = (x, y)
                    old_v = self._v.value(x, y)
                    new_best_v = float('-inf')
                    for a in range(8):
                        s_prime, r, p = environment.next_state_and_reward_distribution(cell, a)
                        new_v = 0
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                        if new_v > new_best_v:
                            new_best_v = new_v
                    self._v.set_value(x, y, new_best_v)
                    delta = max(delta, abs(old_v - new_best_v))
            iteration += 1
            print(f'Finished value iteration iteration {iteration}')
            if delta < self._theta:
                break
            if iteration >= self._max_optimal_value_function_iterations:
                print('Maximum number of iterations exceeded')
                break

        

    def _extract_policy(self):

        # This method returns no value.
        # The policy is in self._pi

        environment = self._environment
        map = environment.map()
        actions = [0, 1, 2, 3, 4, 5, 6, 7]

        for x in range(map.width()):
            for y in range(map.height()):
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue
                cell = (x, y)
                new_best_a = None
                new_best_v = float('-inf')
                for new_a in actions:
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, new_a)
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                    if new_v > new_best_v:
                        new_best_a = new_a
                        new_best_v = new_v
                self._pi.set_action(x, y, new_best_a)


    # def _compute_optimal_value_function(self):
    #     # Initialize iteration counter
    #     iteration = 0
        
    #     # Iterate until convergence or max iterations
    #     while iteration < self._max_optimal_value_function_iterations:
    #         # Initialize delta for convergence check
    #         delta = 0
            
    #         # Copy current value function for updates
    #         new_v = copy.deepcopy(self._v)
            
    #         # Iterate over all states
    #         for x in range(self._environment.map().width()):
    #             for y in range(self._environment.map().height()):
    #                 # Skip terminal and obstruction states
    #                 if self._environment.map().cell(x, y).is_obstruction() or self._environment.map().cell(x, y).is_terminal():
    #                     continue

    #                 # Initialize variables for finding the maximum value
    #                 max_v = float('-inf')

    #                 # Iterate over all possible actions
    #                 for action in range(8):
    #                     # Compute the value for the current action in the current state
    #                     s_prime, rewards, transition_probs = self._environment.next_state_and_reward_distribution((x, y), action)
    #                     value = sum(transition_probs[i] * (rewards[i] + self._gamma * self._v.value(s_prime[i].coords()[0], s_prime[i].coords()[1])) for i in range(len(s_prime)))

    #                     # Update maximum value if found
    #                     if value > max_v:
    #                         max_v = value

    #                 # Update the new value function with the maximum value
    #                 new_v.set_value(x, y, max_v)

    #                 # Update delta for convergence check
    #                 delta = max(delta, abs(self._v.value(x, y) - max_v))

    #         # Update value function
    #         self._v = new_v

    #         # Check for convergence
    #         if delta < self._theta:
    #             break

    #         iteration += 1

    # def _extract_policy(self):
    #     # Iterate over all states
    #     for x in range(self._environment.map().width()):
    #         for y in range(self._environment.map().height()):
    #             # Skip terminal and obstruction states
    #             if self._environment.map().cell(x, y).is_obstruction() or self._environment.map().cell(x, y).is_terminal():
    #                 continue

    #             # Initialize variables for finding the best action
    #             best_action = None
    #             max_v = float('-inf')

    #             # Iterate over all possible actions
    #             for action in range(8):
    #                 # Compute the value for the current action in the current state
    #                 s_prime, rewards, transition_probs = self._environment.next_state_and_reward_distribution((x, y), action)
    #                 value = sum(transition_probs[i] * (rewards[i] + self._gamma * self._v.value(s_prime[i].coords()[0], s_prime[i].coords()[1])) for i in range(len(s_prime)))

    #                 # Update best action if found
    #                 if value > max_v:
    #                     best_action = action
    #                     max_v = value

    #             # Set the best action for the current state in the policy
    #             self._pi.set_action(x, y, best_action)
