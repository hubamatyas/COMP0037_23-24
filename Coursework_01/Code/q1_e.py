#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType
import numpy as np

if __name__ == '__main__':
    
    # Create the scenario
    airport_map, drawer_height = full_scenario()
    
    # Just use Euclidean distance for traversability costs for now
    airport_map.set_use_cell_type_traversability_costs(False)
    
    # Draw what the map looks like. This is optional and you
    # can comment it out
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()    
    airport_map_drawer.wait_for_key_press()
        
    # Create the gym environment
    # Q1e:
    # You will need to enable your implementation of Dijkstra
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.DIJKSTRA)
    
    # Show the graphics
    airport_environment.show_graphics(True)
    
    # First specify the start location of the robot
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    if reward is -float('inf'):
        print('Unable to teleport to (1, 1)')
        
    # Get all the rubbish bins and toilets; these are places which need cleaning
    all_rubbish_bins = airport_map.all_rubbish_bins()
        
    # Q1e:
    # Modify to collect statistics for assessing algorithms
    # Now go through them and plan a path sequentially
    bin_number = 1
    path_performance = []
    for rubbish_bin in all_rubbish_bins:
            action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
            observation, reward, done, info = airport_environment.step(action)

            path_performance.append((info.number_of_cells_visited, info.number_of_waypoints, info.path_travel_cost))
            screen_shot_name = f'binDIJ_{bin_number:02}.pdf'
            airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
            bin_number += 1
            # try:
            #     input("Press enter in the command window to continue.....")
            # except SyntaxError:
            #     pass  

    path_performance = np.array(path_performance)

    print(f"\nSum of visited cells: {np.sum(path_performance[:, 0])}")
    print(f"Mean of visited cells: {np.mean(path_performance[:, 0])}")
    print(f"Median of visited cells: {np.median(path_performance[:, 0])}")
    print(f"Standard deviation of visited cells: {np.std(path_performance[:, 0])}\n")

    print(f"\nSum of number of path cost: {np.sum(path_performance[:, 2])}")
    print(f"Mean of number of path cost: {np.mean(path_performance[:, 2])}")
    print(f"Median of number of path cost: {np.median(path_performance[:, 2])}")
    print(f"Standard deviation of number of path cost: {np.std(path_performance[:, 2])}\n")
     
    
