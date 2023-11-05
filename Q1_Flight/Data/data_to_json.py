import pandas as pd
import json

graph = {}
data = pd.read_csv('train_data.csv')
print(data[data['DestinationAirport'] == 'Abraham Lincoln Capital Airport'])

for i in range(len(data)) :
    row = data.iloc[i]
    source = row.SourceAirport
    target = row.DestinationAirport
    distance = row.Distance

    if source not in graph.keys() :
        graph[source] = {}

    if source in graph.keys() :

        if target not in graph[source].keys():
            graph[source][target] = distance

    if target not in graph.keys():
        graph[target] = {}

    if target in graph.keys():

        if source not in graph[target].keys():
            graph[target][source] = distance




print(graph['Abraham Lincoln Capital Airport'])

with open('Nodes.json', 'w') as file :
    json.dump(graph, file)

