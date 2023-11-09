import pandas as pd
import json

graph = {}
data = pd.read_csv('train_data.csv')

for i in range(len(data)) :
    row = data.iloc[i]
    if row.SourceAirport not in graph.keys() :
        graph[row.SourceAirport] = {}

    if row.SourceAirport in graph.keys() :

        if row.DestinationAirport not in graph[row.SourceAirport].keys():
            graph[row.SourceAirport][row.DestinationAirport] = row.Distance


with open('Nodes.json', 'w') as file :
    json.dump(graph, file)

