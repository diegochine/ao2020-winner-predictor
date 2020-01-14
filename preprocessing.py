import pandas as pd
import numpy as np

DATA_TYPES = {
    'ATP': int,
    'Location': object,
    'Tournament': object,
    'Series': object,
    'Court': object,
    'Surface': object,
    'Round': object,
    'Best of': int,
    'Winner': object,
    'Loser': object,
    'WRank': float,
    'LRank': float,
    'WPts': float,
    'LPts': float,
    'W1': float,
    'L1': float,
    'W2': float,
    'L2': float,
    'W3': float,
    'L3': float,
    'W4': float,
    'L4': float,
    'W5': float,
    'L5': float,
    'Wsets': float,
    'Lsets': float,
    'Comment': object,
    'B365W': float,
    'B365L': float,
    'EXW': object,
    'EXL': float,
    'LBW': float,
    'LBL': float,
    'PSW': float,
    'PSL': float,
    'SJW': float,
    'SJL': float,
    'MaxW': float,
    'MaxL': float,
    'AvgW': float,
    'AvgL': float,
    'WElo': float,
    'WSurfElo': float,
    'WHand': object,
    'WBHand': float,
    'LElo': float,
    'LSurfElo': float,
    'LHand': object,
    'LBHand': float
}

def unify_data(df,
               features_to_drop=[], 
               missing_values="drop", 
               drop_first=False):
     # Sort by date to calculate ELO
    X = df.sort_values(by='Date')
    
    # Drop unuseful columns
    features_to_drop += ['ATP', 'Location', 'Tournament', 'Date', 'Comment', 
                         'Winner', 'Loser', 'Wsets', 'Lsets', 
                         'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 
                         'B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW', 'PSL', 'SJW', 'SJL',
                         'WBD', 'LBD']
    X = X.drop(columns=features_to_drop)
    
    # Deal with missing values
    X['WRank'] = X['WRank'].fillna(value=X['WRank'].max()+100).astype(int)
    X['LRank'] = X['LRank'].fillna(value=X['LRank'].max()+100).astype(int)

    if missing_values == 'drop':
        X = X.dropna()
    elif missing_values == 'custom':
        pass
    else:
        raise ValueError('Wrong parameter: missing_values')

    # Convert ordinal features to int (higher value means more important)
    conversion_dict = {}
    if 'Series' not in features_to_drop:
        series = ['ATP250', 'ATP500', 'Masters 1000', 'Masters Cup', 'Grand Slam']
        series2int = {s: i for i, s in enumerate(series)}
        conversion_dict['Series'] = series2int
        
    if 'Round' not in features_to_drop:
        rounds2int = {'1st Round': 0,
                      '2nd Round': 1,
                      '3rd Round': 2,
                      '4th Round': 3,
                      'Round Robin': 4,
                      'Quarterfinals': 5,
                      'Semifinals': 6,
                      'The Final': 7,
                     }
        conversion_dict['Round'] = rounds2int
    
    # Convert categorical (binary) fields to int
    if 'Court' not in features_to_drop:
        conversion_dict['Court'] = {'Outdoor': 0, 'Indoor': 1}
        
    if 'WHand' not in features_to_drop:
        conversion_dict['WHand'] = {'R': 0, 'L': 1}
    
    if 'LHand' not in features_to_drop:
        conversion_dict['LHand'] = {'R': 0, 'L': 1}
    
    X = X.replace(conversion_dict)
    
    if 'WBHand' not in features_to_drop:
        X['WBHand'] = X['WBHand'].astype(int)
       
    if 'LBHand' not in features_to_drop:
        X['LBHand'] = X['LBHand'].astype(int)
    
    # One hot encode categorical features into binary features
    if 'Surface' not in features_to_drop:
        X = pd.get_dummies(X, prefix=['Surface_'], columns=['Surface'], drop_first=drop_first)
    
    # Generate new columns
    #X['GreaterRank'] = (X['WRank'] < X['LRank']).astype(int)
    
    # Rename columns
    X = X.rename(columns={'WRank':'P1Rank', 'LRank':'P2Rank', 
                          'MaxW':'MaxP1', 'MaxL':'MaxP2', 
                          'AvgW':'AvgP1', 'AvgL':'AvgP2',
                          'WPts':'P1Pts', 'LPts':'P2Pts',
                          'WElo':'P1Elo', 'LElo':'P2Elo',
                          'WSurfElo':'P1SurfElo', 'LSurfElo':'P2SurfElo',
                          'WHand':'P1Hand', 'LHand':'P2Hand',
                          'WBHand':'P1BHand', 'LBHand':'P2BHand'})
    
    return X
    

def preprocess_data(min_date=2011,
                    max_date=2019,
                    features_to_drop=[], 
                    missing_values="drop", 
                    drop_first=False):
    """
    Processes raw data and returns a tuple (X, Y) where X is the cleaned dataset and Y is the array of labels.
    """
    # Loads data for the given years
    if max_date > 2019 or min_date < 2011:
        raise ValueError("Wrong date parameter")
    df = pd.read_csv("data/" + str(min_date) + ".csv", encoding='utf-8-sig', dtype=DATA_TYPES)
    for year in range (min_date + 1, max_date + 1):
        filename = "data/" + str(year) + ".csv"
        df = pd.concat((df, pd.read_csv(filename, encoding='utf-8-sig', dtype=DATA_TYPES)))
        
    X = unify_data(df, features_to_drop, missing_values, drop_first)
   
    # Generate labels
    Y = np.concatenate([np.ones(X.shape[0], dtype=int), np.zeros(X.shape[0], dtype=int)])
    
    # Duplicate data with swapped columns
    tmp = X.copy()
    cols_to_swap = ['P1Rank', 'P2Rank', 'MaxP1', 'MaxP2',  'AvgP1',  'AvgP2', 'P1Pts', 'P2Pts',
                    'P1Elo', 'P2Elo', 'P1SurfElo', 'P2SurfElo', 'P1Hand', 'P2Hand', 'P1BHand', 'P2BHand']
    cols_to_swap = [f for f in cols_to_swap if f not in features_to_drop]
    cols_swapped = ['P2Rank', 'P1Rank', 'MaxP2', 'MaxP1',  'AvgP2',  'AvgP1', 'P2Pts', 'P1Pts',
                    'P2Elo', 'P1Elo', 'P2SurfElo', 'P1SurfElo', 'P2Hand', 'P1Hand', 'P2BHand', 'P1BHand']
    cols_swapped = [f for f in cols_swapped if f not in features_to_drop]
    
    tmp[cols_to_swap] = tmp[cols_swapped]
    tmp.index = np.array(range(X.shape[0] + 1, X.shape[0] * 2 + 1))
    X = pd.concat((X, tmp))
    
    return X, Y

