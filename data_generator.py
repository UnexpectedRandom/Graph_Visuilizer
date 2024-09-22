# IMPORTANT SIDE NOTE: this file is for generating flu data not good for practical use

import pandas as pd
import numpy as np
from datetime import timedelta, date

def generate_dates(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date)

def generate_flu_data(start_date, end_date, num_samples):
    dates = generate_dates(start_date, end_date)
    
    num_samples = min(num_samples, len(dates))
    
    data = {
        'Date': dates[:num_samples],
        'Temperature': np.random.uniform(0, 30, num_samples),
        'Humidity': np.random.uniform(20, 100, num_samples),
        'Precipitation': np.random.uniform(0, 50, num_samples),
        'Vaccination Rate': np.random.uniform(0.5, 1.0, num_samples),
        'Population Density': np.random.uniform(100, 1000, num_samples),
        'Social Distancing Measures': np.random.choice([0, 1], num_samples),
        'Previous Flu Cases': np.random.poisson(10, num_samples),
        'Age Distribution 0-14': np.random.uniform(0, 1, num_samples),
        'Age Distribution 15-64': np.random.uniform(0, 1, num_samples),
        'Age Distribution 65+': np.random.uniform(0, 1, num_samples),
    }
    
    data['Flu Cases'] = (data['Temperature'] * 0.3 + data['Humidity'] * 0.2 - 
                         data['Vaccination Rate'] * 0.5 + 
                         data['Previous Flu Cases'] * 0.7 + 
                         np.random.normal(0, 5, num_samples)).clip(0)
    
    return pd.DataFrame(data)

# Example usage
flu_data = generate_flu_data(date(2020, 1, 1), date(2020, 12, 31), 365)
flu_data.to_csv('flu_data.csv', index=False)
