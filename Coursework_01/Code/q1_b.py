#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType
from grid_search.planned_path import PlannedPath
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    # Create the scenario
    airport_map, drawer_height = full_scenario()

    print(airport_map)
    
    # Draw what the map looks like. This is optional and you
    # can comment it out
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()    
    airport_map_drawer.wait_for_key_press()
        
    # Create the gym environment
    # Q1b:
    # Evaluate breadth and depth first algorithms.
    # Check the implementation of the environment
    # to see how the planner type is used.
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.DEPTH_FIRST)
    
    # Set to this to True to generate the search grid and
    # show graphics. If you set this to false, the
    # screenshot will not work.
    airport_environment.show_graphics(True)

    # Set to this to True to show step-by-step graphics.
    # This is potentially useful for debugging, but can be very slow.    
    airport_environment.show_verbose_graphics(False)
    
    # First specify the start location of the robot
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    if reward is -float('inf'):
        print('Unable to teleport to (1, 1)')
        
    # Get all the rubbish bins and toilets; these are places which need cleaning
    all_rubbish_bins = airport_map.all_rubbish_bins()
        
    # Q1b:
    # Modify this code to collect the data needed to assess the different algorithms
    # This code also shows how to dump the search grid. For your submitted coursework,
    # please DO NOT include all the graphs - just the important ones
    
    # Now go through them and plan a path sequentially

    bin_number = 1
    path_performance = []
    
    for rubbish_bin in all_rubbish_bins:
            action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
            observation, reward, done, info = airport_environment.step(action)
            print("Observation: ", observation)
            print("Reward: ", reward)
            print("Done: ", done)
            print("Info: ", info)

            print("\nNumber of cells visited", info.number_of_cells_visited)
            print("Number of waypoints", info.number_of_waypoints)
            print("Path travel cost", info.path_travel_cost)

            # print("\nNumber of cells visited", info.number_of_cells_visited)
            # print("Number of waypoints", info.number_of_waypoints)
            # print("Path travel cost", info.path_travel_cost)
            path_performance.append((info.number_of_cells_visited, info.number_of_waypoints, info.path_travel_cost))

            screen_shot_name = f'binDFS_{bin_number:02}.pdf'
            airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
            bin_number += 1
    
            try:
                input("Press enter in the command window to continue.....")
            except SyntaxError:
                pass


    path_performance = np.array(path_performance)

    print(f"\nSum of visited cells: {np.sum(path_performance[:, 0])}")
    print(f"Mean of visited cells: {np.mean(path_performance[:, 0])}")
    print(f"Median of visited cells: {np.median(path_performance[:, 0])}")
    print(f"Standard deviation of visited cells: {np.std(path_performance[:, 0])}\n")

    print(f"\nSum of number of path cost: {np.sum(path_performance[:, 2])}")
    print(f"Mean of number of path cost: {np.mean(path_performance[:, 2])}")
    print(f"Median of number of path cost: {np.median(path_performance[:, 2])}")
    print(f"Standard deviation of number of path cost: {np.std(path_performance[:, 2])}\n")

    # BFS
    # Sum of visited cells: 13050.0
    # Mean of visited cells: 501.9230769230769
    # Median of visited cells: 359.0
    # Standard deviation of visited cells: 402.9020613076051

    # Sum of number of path cost: 580.4701294725886
    # Mean of number of path cost: 22.325774210484177
    # Median of number of path cost: 16.44974746830583
    # Standard deviation of number of path cost: 13.85645367592506

    # DFS
    # Sum of visited cells: 32296.0
    # Mean of visited cells: 1242.1538461538462
    # Median of visited cells: 1565.0
    # Standard deviation of visited cells: 683.2783356902725


    # Sum of number of path cost: 7538.815218587393
    # Mean of number of path cost: 289.9544314841305
    # Median of number of path cost: 408.93964620054044
    # Standard deviation of number of path cost: 180.16071209597828


    # Dijkstra
    # Sum of visited cells: 13069.0
    # Mean of visited cells: 502.65384615384613
    # Median of visited cells: 349.0
    # Standard deviation of visited cells: 426.0825075668302


    # Sum of number of path cost: 550.6467529817257
    # Mean of number of path cost: 21.178721268527912
    # Median of number of path cost: 15.828427124746192
    # Standard deviation of number of path cost: 13.099306237566916


    # Dijkstra with penalty factor
    # Sum of visited cells: 12189.0
    # Mean of visited cells: 468.8076923076923
    # Median of visited cells: 341.0
    # Standard deviation of visited cells: 412.70598765770376


    # Sum of number of path cost: 635.2325394193526
    # Mean of number of path cost: 24.43202074689818
    # Median of number of path cost: 15.828427124746192
    # Standard deviation of number of path cost: 25.098906340104612

    # A* with penalty factor
    # Sum of visited cells: 5201.0
    # Mean of visited cells: 200.03846153846155
    # Median of visited cells: 64.0
    # Standard deviation of visited cells: 392.637088334883


    # Sum of number of path cost: 635.2325394193526
    # Mean of number of path cost: 24.43202074689818
    # Median of number of path cost: 15.828427124746192
    # Standard deviation of number of path cost: 25.098906340104612