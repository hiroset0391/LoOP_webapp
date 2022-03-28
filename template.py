"""
This is a sample script to calculate Local Outlier Probabilities (LoOPs).
The input CSV file and Python script that includes packages for LoOP calculation are also included in the same Zip file.
The input CSV file and tuning parameters (extent and neighborhood size) you used on the web app are set. 
"""
import datetime
import pandas as pd
import numpy as np
import loop_functions

if __name__ == '__main__':

    """
    Input 
    """
    csv_name = 'test.csv'
    extent = 3
    n_neighbors = 10

    """
    Calculate LoOP [%]
    """
    df = pd.read_csv(csv_name)
    date = df.values[:,0]
    Date = []
    for _ in range(len(date)):
        tdatetime = datetime.datetime.strptime( date[_], '%Y-%m-%d') 
        Date.append(datetime.date(tdatetime.year, tdatetime.month, tdatetime.day))
    
    X = np.array(df.values[:,1:]).astype(np.float32)
    
    scores = loop_functions.LocalOutlierProbability(X, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
    scores *= 100

    """
    Output 
    """
    df_out = pd.DataFrame({'date': Date, 'LoOP': scores})
    df_out.to_csv('loop_out.csv')