import pandas

def text_file_generator(algorithm_name, execution_time , path_list) :
    flight_data = pandas.read_csv('Q1_Flight/Data/Flight_Data.csv')
    pair_airports = [[path_list[i], path_list[i+1]] for i in range(len(path_list) - 1)]

    print(pair_airports)

    for item in pair_airports :
        for i in range(len(flight_data)) :
            source_airport = flight_data.iloc[i].SourceAirport
            destination_airport = flight_data.iloc[i].DestinationAirport

            if item == [source_airport, destination_airport] :
                item.append(flight_data.iloc[i].SourceAirport_City)
                item.append(flight_data.iloc[i].SourceAirport_Country)
                item.append(flight_data.iloc[i].DestinationAirport_City)
                item.append(flight_data.iloc[i].DestinationAirport_Country)
                item.append(flight_data.iloc[i].Airline)
                item.append(flight_data.iloc[i].Price)
                item.append(flight_data.iloc[i].FlyTime)
                item.append(flight_data.iloc[i].Distance)
                break

            elif item == [destination_airport, source_airport] :
                item.append(flight_data.iloc[i].DestinationAirport_City)
                item.append(flight_data.iloc[i].DestinationAirport_Country)
                item.append(flight_data.iloc[i].SourceAirport_City)
                item.append(flight_data.iloc[i].SourceAirport_Country)
                item.append(flight_data.iloc[i].Airline)
                item.append(flight_data.iloc[i].Price)
                item.append(flight_data.iloc[i].FlyTime)
                item.append(flight_data.iloc[i].Distance)
                break
    # pair_airport items : [ source airport , target airport , source airport city, source airport country, target airport city, target airport country,  Airline, price , fly time, distance ]
    text = []
    text.append(f'{algorithm_name}\n')
    text.append(f'Execution Time : {execution_time}s\n')
    text.append('.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n')

    i = 1
    total_time = 0
    total_price = 0
    total_duration = 0
    for flight in pair_airports :
        text.append(f'Flight #{i}  ({flight[6]})\n')
        text.append(f'From :  {flight[0]} - {flight[2]}, {flight[3]}\n')
        text.append(f'To :  {flight[1]} - {flight[4]}, {flight[5]}\n')
        duration = str(flight[-1])[:4]
        text.append(f'Duration : {duration}km\n')
        total_duration += flight[-1]
        total_time += flight[-2]
        time = str(flight[-2])[:4]
        text.append(f'Time : {time} h\n')
        total_price += flight[-3]
        price = str(flight[-3])[:6]
        text.append(f'Price : {price} $\n')
        text.append('--------------------------------------------\n')

    text.append(f'Total Time : {str(total_time)[:4]} h\n')
    text.append(f'Total Price : {str(total_price)[:6]} $\n')
    text.append(f'Total Duration : {str(total_duration)[:7]} km\n')

    with open('Q1_Flight/Data/test.txt', 'w') as f :
        f.writelines(text)




text_file_generator('A Star', 312,  ['Imam Khomeini International Airport', 'Atatürk International Airport', 'Washington Dulles International Airport', 'Hartsfield Jackson Atlanta International Airport'])
