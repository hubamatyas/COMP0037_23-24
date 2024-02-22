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
        
        value_iteration_step = self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        return self._v, self._pi, value_iteration_step

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):
        environment = self._environment
        map = environment.map()

        # Actions the robot might take in each cell are the octagonal actions
        # Ints representing the actions in LowLevelActionType
        actions = [0, 1, 2, 3, 4, 5, 6, 7]

        iteration = 0

        while True:

            delta = 0

            # Iterate over all the states
            for x in range(map.width()):
                for y in range(map.height()):

                    # Skip terminal and obstruction states
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue

                    cell = (x, y)

                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    # Initialize variables for finding the maximum value
                    new_best_v = float('-inf')

                    for a in actions:

                        # Compute p(s',r|s,a)
                        s_prime, r, p = environment.next_state_and_reward_distribution(cell, a)
                        
                        # Sum over the rewards
                        new_v = 0
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                        
                        # Update new best value if this action leads to a higher value
                        if new_v > new_best_v:
                            new_best_v = new_v

                    # Update the value function with the new maximum value found
                    self._v.set_value(x, y, new_best_v)

                    # Update delta for convergence check
                    delta = max(delta, abs(old_v - new_best_v))

            # Increment the iteration counter
            iteration += 1

            print(f'Finished value iteration step {iteration}')
            
            # Terminate the loop if the change was very small (ie. convergence is reached)
            if delta < self._theta:
                break

            if iteration >= self._max_optimal_value_function_iterations:
                print('Maximum number of iterations exceeded')
                break

        
    def _extract_policy(self):
        environment = self._environment
        map = environment.map()

        # Actions the robot might take in each cell are the octagonal actions
        # Integers represent the actions in LowLevelActionType
        actions = [0, 1, 2, 3, 4, 5, 6, 7]

        # Iterate over all states
        for x in range(map.width()):
            for y in range(map.height()):

                # Skip terminal and obstruction states
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue

                cell = (x, y)

                # Initialize variables for finding the best action
                new_best_a = None
                new_best_v = float('-inf')

                for new_a in actions:
                    # Compute p(s',r|s,a)
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, new_a)
                    
                    # Sum over the rewards
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                    
                    # Update new best action and value if this action leads to a higher value
                    if new_v > new_best_v:
                        new_best_a = new_a
                        new_best_v = new_v

                # Set the best action for the current state in the policy
                self._pi.set_action(x, y, new_best_a)