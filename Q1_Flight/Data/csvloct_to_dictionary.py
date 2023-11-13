import pandas as pd
# Define an empty dictionary to store latitude and longitude values
source_locations = {}

# Read data from the CSV file
data = pd.read_csv('Flight_Data.csv')
    
    # Skip the header row if it exists

    
    # Iterate over each row in the CSV file 
for index, row in data.iloc[1:].iterrows():
    if row[1] in source_locations:
        continue
    # Extract relevant columns (latitude and longitude)
    airline_name = row[1]
    source_lat = float(row[5])
    source_lon = float(row[6])
    
    # Store latitude and longitude in the dictionary using airline name as the key
    source_locations[airline_name] = {'latitude': source_lat, 'longitude': source_lon}
    
    