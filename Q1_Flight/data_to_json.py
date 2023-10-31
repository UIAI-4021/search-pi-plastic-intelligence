import pandas as pd
import json

data = pd.read_csv('train_data.csv')
print('number of examples : ', len(data))

graph = {}
for i in range(len(data)) :
    airport = data.iloc[i].SourceAirport
    if airport not in graph.keys() :
        graph[airport] = {'Neighbors': [], 'Distance' : [], 'Price' : [], 'FlyTime' : []}

    if airport in graph.keys() :
        if data.iloc[i].DestinationAirport not in graph[airport]['Neighbors'] :
            graph[airport]['Neighbors'].append(data.iloc[i].DestinationAirport)
            graph[airport]['Distance'].append(data.iloc[i].Distance)
            graph[airport]['FlyTime'].append(data.iloc[i].FlyTime)
            graph[airport]['Price'].append(data.iloc[i].Price)

i = 0
for key, value in graph.items() :
    print(key, value , end = '\n')
    i += 1
    if i == 5 :
        break



print(len(graph.keys()))
print(len(graph['Imam Khomeini International Airport']['Neighbors']))
graph = { 'Nodes' : graph}


with open('Nodes.json', 'w') as file :
    json.dump(graph, file)
