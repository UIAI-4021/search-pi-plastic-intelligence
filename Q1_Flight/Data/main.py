import heapq
import math
infinity = float('inf')
class Graph:
    def __init__(self, graph_dict = None, directed = True) :
        self.graph_dict = graph_dict or {}
        self.directed = directed
    def get(self, a, b=None) :
        links = self.graph_dict.setdefault(a,{})
        if b is None:
            return links
        else:
            return links.get(b)
        


class Problem(object):
    def __init__(self, initial, goal = None):
        self.initial = initial
        self.goal = goal
    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_in(element, my_list):
        return element in my_list
        
    def goal_test(self, state):
        if isinstance(self.goal, list):
            return self.goal in list
        else:
            return state == self.goal
            
    def path_cost(self, c, state1, action, state2):
        return c + 1 
    


class Node:
    def __init__(self, state, parent = None, action = None, path_cost = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    def __repr__(self) :
            return "<Node {}>".format(self.state)
    def expand(self, problem):
            return [self.child_node(problem, action)
                    for action in problem.actions(self.state)]
    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        new_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
        next_node = Node(next_state, self, action, new_cost)
        return next_node 
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))
    def solution(self):
        return [node.state for node in self.path()]
            
def haversine_distance(coord1, coord2):
    # Coordinates are given as (latitude, longitude) pairs in degrees
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Compute differences in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate the distance
    distance = R * c
    
    return distance

class GraphProblem(Problem):
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph
    def actions(self, A):
        return self.graph.get(A)
    def result(self, state, action):
        return action
    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B)or infinity)
    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return haversine_distance(locs[node.state], locs[self.goal])
        else:
            return float('inf')
        

        
        
def closest_node_entry_num(nodelist):
    min_index = 0
    min_dist = list(nodelist[0].keys())[0]
    for n in range(1, len(nodelist)):
        dist = list(nodelist[n].keys())[0]
        if dist < min_dist:
            min_index = n
            min_dist = dist
    return min_index


def astar_search(problem):
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    gval = node.path_cost #start to node n
    hval = problem.h(node) #calculate cheapest from n to goal
    #each entry -> a dictionary item with key as f-val and value as node
    nodelist = [{gval + hval: node}]
    while nodelist:
        entry_num = closest_node_entry_num(nodelist)
        min_dist = list(nodelist[entry_num].keys())[0]
        closest_node = nodelist[entry_num][min_dist]
        print("Current Nodes : ", nodelist)
        print("min dist", min_dist, ", closest node = ", closest_node)
        input('Press enter to continue...')
        if problem.goal_test(closest_node.state):
            return closest_node
        nodelist.pop(entry_num)
        for child in closest_node.expand(problem):
            gval = child.path_cost
            hval = problem.h(child)
            nodelist.append({gval + hval : child})













