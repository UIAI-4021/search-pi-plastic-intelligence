import heapq

def a_star(graph, start, goal):
    open_list = [(0, start)]  # Priority queue of nodes to explore (f-value, node)
    came_from = {}  # To keep track of the parent node for each node
    g_values = {city: float('inf') for city in graph}  # g(n) values for nodes, initially set to infinity
    g_values[start] = 0
    f_values = {city: float('inf') for city in graph}  # f(n) values for nodes, initially set to infinity
    f_values[start] = h(start, goal)  # Calculate h(n) for the starting node

    while open_list:
        f, current_city = heapq.heappop(open_list)

        if current_city == goal:
            return reconstruct_path(came_from, current_city)

        for neighbor, distance in graph[current_city]:
            tentative_g = g_values[current_city] + distance

            if tentative_g < g_values[neighbor]:
                came_from[neighbor] = current_city
                g_values[neighbor] = tentative_g
                f_values[neighbor] = tentative_g + h(neighbor, goal)
                heapq.heappush(open_list, (f_values[neighbor], neighbor))

    return None  # No path found

def h(city, goal):
    # Heuristic function: You can use the straight-line distance (Euclidean distance) as an example
    # Calculate the Euclidean distance between the coordinates of 'city' and 'goal'
    # Replace the following line with your specific distance calculation
    return euclidean_distance(city, goal)

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
