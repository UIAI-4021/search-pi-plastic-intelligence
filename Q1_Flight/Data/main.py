import heapq

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
    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
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
            return [self.child_node(Problem, action)
                    for action in problem.actions(self.state)]
        def child_node(self, problem, action):
            next_state = problem.result(self.state, action)
            new_cost = problem.path_cost(self.path_cost, self.state. action, next_state)
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
            

        
           
                     








def a_star(graph, start, goal):
    

    

def euclidean_distance(city1, city2):
    # Replace this with your actual distance calculation based on city coordinates or other data
    # For this example, return a constant value of 1 for simplicity
    return 1

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.insert(0, current)
        current = came_from[current]
    path.insert(0, current)  # Add the starting city
    return path

# Define your dataset of cities
cities = {
    'city1': [('city2', 20), ('city4', 22)],
    'city4' : [('city1', 22), ('city99', 1000)],
    'city2': [('city1', 20), ('city5', 25), ('city7', 30)],
    'city5' : [('city2', 25), ('city99', 50)],
    'city7' : [('city2', 30)],
    'city99' : [('city4', 1), ('city5', 50)]
    # Add more cities and connections here
}



start_city = 'city1'
goal_city = 'city99'

path = a_star(cities, start_city, goal_city)

if path:
    print(f"Shortest path from {start_city} to {goal_city}: {path}")
else:
    print(f"No path found from {start_city} to {goal_city}")
