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
            X[:,col] = (X[:,col]-np.nanmean(X[:,col])) / np.nanstd(X[:,col])
    
    return X

def remove_nan(X, Date):
    nnan_idx = ~np.isnan(X).any(axis=1)
    X = X[nnan_idx, :]
    Date = Date[nnan_idx]
    return X, Date

def score_fillnan(scores, Date, Date_scores):
    
    scores_new = np.zeros(len(Date))*np.nan
    for dd, fday in enumerate(Date):
        idx = np.where(Date_scores==fday)[0]
        if len(idx)>0:
            scores_new[dd] = scores[idx]  
        else:
            pass
    return scores_new

if __name__ == '__main__':

    """
    Input 
    """
    csv_name = 'test.csv'
    extent = 3
    n_neighbors = 10
    time_series = True

    """
    Calculate LoOP [%]
    """
    df = pd.read_csv(csv_name)
    date = df.values[:,0]
    Date = []
    for _ in range(len(date)):
        tdatetime = datetime.datetime.strptime( date[_], '%Y-%m-%d') 
        Date.append(datetime.date(tdatetime.year, tdatetime.month, tdatetime.day))

    if time_series:
        import dateutil.parser
        date = df.values[:,0]
        Date = []
        for _ in range(len(date)):
            tdatetime = dateutil.parser.parse(date[_]) 
            Date.append(datetime.datetime(tdatetime.year, tdatetime.month, tdatetime.day))

        X = np.array(df.values[:,1:]).astype(np.float32)
        
    else:
        X = np.array(df.values[:,:]).astype(np.float32)
        Date = list( range(X.shape[0]) )
        
    
    X = np.array(df.values[:,1:]).astype(np.float32)

    X_scores, Date_scores = remove_nan(X, Date)
    X_scores = normalize(X_scores) 

    scores = loop_functions.LocalOutlierProbability(X_scores, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
    scores *= 100

    scores = score_fillnan(scores, Date, Date_scores)

    """
    Output 
    """
    df_out = pd.DataFrame({'date': Date, 'LoOP': scores})
    df_out.to_csv('loop_result.csv')