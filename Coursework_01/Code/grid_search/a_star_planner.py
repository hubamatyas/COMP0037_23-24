'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math

from .dijkstra_planner import DijkstraPlanner
from .occupancy_grid import OccupancyGrid
from .search_grid import SearchGridCell

class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        DijkstraPlanner.__init__(self, occupancy_grid)

    # Q2d:
    # Complete implementation of A*.
    
    def push_cell_onto_queue(self, cell: SearchGridCell):
        parent: SearchGridCell = cell.parent
        if parent:
            cell.path_cost = parent.path_cost + self.compute_l_stage_additive_cost(parent, cell)

        priority = cell.path_cost + self.compute_heuristic(cell)
        self.priority_queue.put((priority, cell))

    def compute_heuristic(self, cell: SearchGridCell):
        # Current implementation is Euclidean distance
        cell_coords = cell.coords()
        goal_coords = self.goal.coords()

        dX = cell_coords[0] - goal_coords[0]
        dY = cell_coords[1] - goal_coords[1]
        return math.sqrt(dX * dX + dY * dY)
