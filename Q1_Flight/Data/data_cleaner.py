import pandas as pd

original_data = pd.read_csv('Q1_Flight/Data/Flight_Data.csv')
data = original_data[['SourceAirport', 'DestinationAirport', 'Distance', 'FlyTime', 'Price']]
print('Real data examples count : ' , len(original_data))
unrepeated_data = data.drop_duplicates(['SourceAirport', 'DestinationAirport'])



unrepeated_data.to_csv('train_data.csv')
print('Done')
