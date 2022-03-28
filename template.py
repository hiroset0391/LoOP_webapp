"""
This is a sample script to calculate Local Outlier Probabilities (LoOPs).
The input CSV file and Python script that includes packages for LoOP calculation are also included in the same Zip file.
The input CSV file and tuning parameters (extent and neighborhood size) you used on the web app are set. 
"""
import datetime
import pandas as pd
import numpy as np
import loop_functions

def normalize(X):
    if X.shape[1]==1:
        X = np.stack([X[:,0], np.zeros(X.shape[0])], 1) 
        
    else:
        for col in range(X.shape[1]):
            Q1 = np.nanpercentile(X[:,col], 25)
            Q2 = np.nanpercentile(X[:,col], 50)
            Q3 = np.nanpercentile(X[:,col], 75)
            X[:,col] = (X[:,col]-Q2)/(Q3-Q1)
    
    return X

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

    try:
        import dateutil.parser
        date = df.values[:,0]
        Date = []
        for _ in range(len(date)):
            tdatetime = dateutil.parser.parse(date[_]) 
            Date.append(datetime.datetime(tdatetime.year, tdatetime.month, tdatetime.day))

        X = np.array(df.values[:,1:]).astype(np.float32)
        
    except:
        X = np.array(df.values[:,:]).astype(np.float32)
        Date = list( range(X.shape[0]) )
        
    
    X = np.array(df.values[:,1:]).astype(np.float32)
    X = normalize(X) 
    
    scores = loop_functions.LocalOutlierProbability(X, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
    scores *= 100

    """
    Output 
    """
    df_out = pd.DataFrame({'date': Date, 'LoOP': scores})
    df_out.to_csv('loop_out.csv')