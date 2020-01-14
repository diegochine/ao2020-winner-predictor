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

def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes 
    the elo ranking of the 2 players at the beginning of the match
    """
    print("Elo rankings computing...")
    players=list(pd.Series(list(data.Winner)+list(data.Loser)).value_counts().index)
    elo=pd.Series(np.ones(len(players))*1500,index=players)
    ranking_elo=[(1500,1500)]
    for i in range(1,len(data)):
        w=data.iloc[i-1,:].Winner
        l=data.iloc[i-1,:].Loser
        elow=elo[w]
        elol=elo[l]
        pwin=1 / (1 + 10 ** ((elol - elow) / 400))    
        K_win=32
        K_los=32
        new_elow=elow+K_win*(1-pwin)
        new_elol=elol-K_los*(1-pwin)
        elo[w]=new_elow
        elo[l]=new_elol
        ranking_elo.append((elo[data.iloc[i,:].Winner],elo[data.iloc[i,:].Loser])) 
    ranking_elo=pd.DataFrame(ranking_elo,columns=["elo_winner","elo_loser"])    
    ranking_elo["proba_elo"]=1 / (1 + 10 ** ((ranking_elo["elo_loser"] - ranking_elo["elo_winner"]) / 400))   
    return ranking_elo

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
    r = compute_elo_rankings(full_dataset)
    X['WEloCalc'] = r['elo_winner']
    X['LEloCalc'] = r['elo_loser']
    X['PElo'] = r['proba_elo']
    
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
    cols_to_swap = ['WRank', 'LRank', 'MaxW', 'MaxL',  'AvgW',  'AvgL', 'WPts', 'LPts',
                    'WElo', 'LElo', 'WSurfElo', 'LSurfElo', 'WHand', 'LHand', 'WBHand', 'LBHand']
    cols_to_swap = [f for f in cols_to_swap if f not in features_to_drop]
    cols_swapped = ['LRank', 'WRank', 'MaxL', 'MaxW',  'AvgL',  'AvgW', 'LPts', 'WPts',
                    'LElo', 'WElo', 'LSurfElo', 'WSurfElo', 'LHand', 'WHand', 'LBHand', 'WBHand']
    cols_swapped = [f for f in cols_swapped if f not in features_to_drop]
    
    tmp[cols_to_swap] = tmp[cols_swapped]
    tmp.index = np.array(range(X.shape[0] + 1, X.shape[0] * 2 + 1))
    X = pd.concat((X, tmp))
    
    return X, Y

