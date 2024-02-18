'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from collections import deque
from math import sqrt
from queue import PriorityQueue

from .occupancy_grid import OccupancyGrid
from .search_grid import SearchGridCell
from .planner_base import PlannerBase

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore

    # Q1d:
    # Modify this class to finish implementing Dijkstra

    def push_cell_onto_queue(self, cell: SearchGridCell):
        parent: SearchGridCell = cell.parent
        if parent:
            cell.path_cost = parent.path_cost + self.compute_l_stage_additive_cost(parent, cell)

        # this is wrong, coz it would just take the euclidean distance from the start, that's not actually the cost to come
        # cell.path_cost = self.compute_l_stage_additive_cost(self.start, cell)
        
        self.priority_queue.put((cell.path_cost, cell))

    # Check the queue size is zero
    def is_queue_empty(self) -> bool:
        return self.priority_queue.empty()

    def pop_cell_from_queue(self) -> SearchGridCell:
        t = self.priority_queue.get()
        return t[1]

    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        predicted_cost = parent_cell.path_cost + self.compute_l_stage_additive_cost(parent_cell, cell)
        if predicted_cost < cell.path_cost:
            cell.set_parent(parent_cell)
            cell.path_cost = predicted_cost
            self.priority_queue.put((cell.path_cost, cell))