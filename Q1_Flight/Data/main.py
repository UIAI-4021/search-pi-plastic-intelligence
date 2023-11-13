import math
import csvloct_to_dictionary as ctd
import json
import result_to_text as rtt
from timeit import default_timer as timer
import sys


a_star_start = timer()

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
    # Coordinates are given as (latitude, longitude) pairs in 
    lat1 = coord1['latitude']
    lon1 = coord1['longitude']
    lat2 = coord2['latitude']
    lon2 = coord2['longitude']
    
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
        
        if problem.goal_test(closest_node.state):
            return closest_node
        nodelist.pop(entry_num)
        for child in closest_node.expand(problem):
            gval = child.path_cost
            hval = problem.h(child)
            nodelist.append({gval + hval : child})
def UndirectedGraph(graph_dict):
    return Graph(graph_dict = graph_dict, directed= False)


with open('Nodes.json', 'r') as json_file:
    # Load JSON data into a Python dictionary
    data = json.load(json_file)
    

def UndirectedGraph(graph_dict= None):
    return Graph(graph_dict = graph_dict, directed = False)

source_airport, destination_airport = input('Enter Airports names : ').split(' - ')

earth_map = UndirectedGraph(data)
earth_map.locations = ctd.source_locations
earth_problem = GraphProblem(source_airport , destination_airport, earth_map )
resultnode = astar_search(earth_problem)

rn = resultnode.path()

a_star_result = [ item.state for item in rn ]

a_star_end = timer()
a_star_exe = a_star_end - a_star_start



rtt.text_file_generator('A Star', a_star_exe , a_star_result)


# -------------------------------------------------- dijkestra -------------------------------------------

dij_start = timer()

class Graph2(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes

    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph
    shortest_path = {}

    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}

    # We'll use max_value to initialize the "infinity" value of the unvisited nodes
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0
    shortest_path[start_node] = 0

    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def dij_path_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Add the start node manually
    path.append(start_node)

    return list(reversed(path))




with open('Nodes.json', 'r') as file :
    graph_data = json.load(file)

airports = list(graph_data.keys())

nodes = airports

init_graph = graph_data
graph = Graph2(nodes, init_graph)
previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=source_airport)

dijkestra_result = dij_path_result(previous_nodes, shortest_path,
                   start_node=source_airport,
                   target_node=destination_airport)

dij_end = timer()
dij_exe = dij_end - dij_start

rtt.text_file_generator('Dijkestra', dij_exe, dijkestra_result )


