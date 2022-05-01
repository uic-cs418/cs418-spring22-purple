#!/usr/bin/env python
# coding: utf-8

#to reflect changes made in modules
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn

def read_data(gun_data, gunlaws):
    # Reading in data for gun violence counts and firearm laws
    missing_row = ['sban_1', '2017-10-01', 'Nevada', 'Las Vegas', 'Mandalay Bay 3950 Blvd S', 59, 489, 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', '-', '-', '-', '-', '-', '36.095', 'Hotel', 
                   '-115.171667', 47, 'Route 91 Harvest Festiva; concert, open fire from 32nd floor. 47 guns seized; TOTAL:59 kill, 489 inj, number shot TBD,girlfriend Marilou Danley POI', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    gun_data.loc[len(gun_data)] = missing_row

    gun_laws = pd.DataFrame(data = gunlaws,columns=['state', 'year', 'lawtotal'])
    return gun_data, missing_row, gun_laws


def clean_data(gun_data):
    missing_row = ['sban_1', '2017-10-01', 'Nevada', 'Las Vegas', 'Mandalay Bay 3950 Blvd S', 59, 489, 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', '-', '-', '-', '-', '-', '36.095', 'Hotel', 
               '-115.171667', 47, 'Route 91 Harvest Festiva; concert, open fire from 32nd floor. 47 guns seized; TOTAL:59 kill, 489 inj, number shot TBD,girlfriend Marilou Danley POI', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    gun_data.loc[len(gun_data)] = missing_row
    # Created some additional features
    gun_data['date'] = pd.to_datetime(gun_data['date'])
    gun_data['year'] = gun_data['date'].dt.year
    gun_data['month'] = gun_data['date'].dt.month
    gun_data['monthday'] = gun_data['date'].dt.day
    gun_data['weekday'] = gun_data['date'].dt.weekday
    gun_data['loss'] = gun_data['n_killed'] + gun_data['n_injured']
    # Sorted data by state since gun reform data is sorted by state and year in ascending order
    gun_data = gun_data.sort_values(['state', 'year'], ascending = [True, True])
    # Filtered only cases where the year is between 2014 and 2017
    gun_data = gun_data[(gun_data['year'] >= 2014) & (gun_data['year'] <= 2017)]
    # Removed some unecessary columns
    gun_data = gun_data.drop(columns=['city_or_county','address','source_url','participant_relationship',
                                        'incident_url_fields_missing', 'latitude', 'longitude', 'location_description',
                                        'participant_name', 'sources', 'month', 'monthday', 'weekday', 'loss',
                                       'gun_stolen', 'gun_type', 'n_guns_involved', 'notes'])
    # Missing values dropped rows with NaN values
    gun_data = gun_data.dropna()
    return gun_data

# Wrapper function to incorporate both read and cleaning
def processAll(gun_data, gun_laws):
    gundata, missing_row, gun_laws = read_data(gun_data, gun_laws)
    return clean_data(gun_data), missing_row, gun_laws