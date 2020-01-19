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
    'WHand': object,
    'WBHand': float,
    'LHand': object,
    'LBHand': float
}

def uniform_name(w):
    surname = w.replace('-', ' ').split()
    newName = surname[-1] + ' '
    for n in range(len(surname)-2, -1, -1):
        newName += surname[n][0] + '.'
    return newName

def compute_probability_elo(p1Elo, p2Elo):
    return 1 / (1 + 10 ** ((p2Elo - p1Elo) / 400)) 

def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes 
    the elo ranking of the 2 players at the beginning of the match.
    Source: https://www.betfair.com.au/hub/tennis-elo-modelling/
    """
    def k_factor(m, c=250, o=5, s=0.4):
        return c / ((m + o) ** s)
    
    players = list(pd.Series(list(data.Winner) + list(data.Loser)).value_counts().index)
    elo = pd.Series(np.ones(len(players))*1500, index=players)
    ranking_elo = [(1500,1500)]
    for i in range(1,len(data)):
        winner = data.iloc[i-1,:].Winner
        loser = data.iloc[i-1,:].Loser
        elow = elo[winner]
        elol = elo[loser]
        pwin = 1 / (1 + 10 ** ((elol - elow) / 400))
        mw = (data.iloc[:i-1,:].Winner == winner).sum()
        ml = (data.iloc[:i-1,:].Loser == loser).sum()
        K_win = k_factor(mw)
        K_los = k_factor(ml)
        new_elow = elow + K_win * (1-pwin)
        new_elol = elol - K_los * (1-pwin)
        elo[winner] = new_elow
        elo[loser] = new_elol
        ranking_elo.append((elo[data.iloc[i,:].Winner],elo[data.iloc[i,:].Loser])) 
    ranking_elo = pd.DataFrame(ranking_elo, columns=["EloWinner","EloLoser"])    
    ranking_elo["ProbaElo"] = compute_probability_elo(ranking_elo["EloWinner"], ranking_elo["EloLoser"])
    return ranking_elo, elo

def unify_data(X,
               features_to_drop=[],
               features_to_add=['elo', 'diff', 'top10']):
    
    # Drop unuseful columns
    if any(f not in X.columns for f in features_to_drop):
        raise ValueError('{} column doesn\'t exist'.format(f))
    features_to_drop += ['ATP', 'Location', 'Tournament', 'Date', 'Comment', 'Winner', 'Loser', 
                         'Wsets', 'Lsets', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 
                         'B365W', 'B365L', 'EXW', 'EXL', 'LBW', 'LBL', 'PSW', 'PSL', 'SJW', 'SJL', 'MaxW', 'MaxL', 
                         'AvgW', 'AvgL', 'WBD', 'LBD']
    X = df.drop(columns=features_to_drop)
    
    # fill missing values for ranks and points
    X['WRank'] = X['WRank'].fillna(value=X['WRank'].max()+1).astype(int)
    X['LRank'] = X['LRank'].fillna(value=X['LRank'].max()+1).astype(int)
    X['WPts'] = X['WPts'].fillna(value=X['WPts'].min()-1).astype(int)
    X['LPts'] = X['LPts'].fillna(value=X['LPts'].min()-1).astype(int)
    # drop rows that still have missing values
    X = X.dropna()
    
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
        X = pd.get_dummies(X, prefix=['Surface_'], columns=['Surface'], drop_first=False)
    
    # Generate new columns
    if 'diff' in features_to_add:
        if all(x not in features_to_drop for x in ('WRank', 'LRank')):
            X['RankDiff'] = (X['WRank'] - X['LRank']).astype(int)
        if all(x not in features_to_drop for x in ('WPts', 'LPts')):
            X['PtsDiff'] = (X['WPts'] - X['LPts']).astype(int)
    
    if 'top10' in features_to_add:
        X['Top10W'] = (X['WRank'] <= 10).astype(int)
        X['Top10L'] = (X['LRank'] <= 10).astype(int)
    
    return X
    

def preprocess_data(min_date=2011,
                    max_date=2020,
                    features_to_drop=[], 
                    features_to_add=['elo', 'diff', 'top10'],
                    labels="duplicate",
                    returnElo=False):
    """
    Processes raw data and returns a tuple (X, Y) where X is the cleaned dataset and Y is the array of labels.
    """
    # Loads data for the given years
    if max_date > 2020 or min_date < 2011:
        raise ValueError("Wrong date parameter")
    X = pd.read_csv("data/" + str(min_date) + ".csv", encoding='utf-8-sig', dtype=DATA_TYPES, parse_dates=['Date', 'WBD', 'LBD'])
    for year in range (min_date + 1, max_date + 1):
        filename = "data/" + str(year) + ".csv"
        X = pd.concat((X, pd.read_csv(filename, encoding='utf-8-sig', dtype=DATA_TYPES, parse_dates=['Date', 'WBD', 'LBD'])))
       
    if 'elo' in features_to_add:
        # Sort by date to calculate ELO
        X = df.sort_values(by='Date')
        # Calculating Elo
        r, playersElo = compute_elo_rankings(X)
        X['WEloCalc'] = r['EloWinner']
        X['LEloCalc'] = r['EloLoser']
        X['ProbaElo'] = r['ProbaElo']
    
    X = unify_data(X, features_to_drop, features_to_add)
    X.index = np.array(range(0, X.shape[0]))
    
    # Duplicate data with swapped columns or random swap
    cols_to_swap = ['WRank', 'LRank', 'WPts', 'LPts', 'WHand', 'LHand', 'WBHand', 'LBHand']
    cols_to_swap = [f for f in cols_to_swap if f not in features_to_drop]
    cols_swapped = ['LRank', 'WRank', 'LPts', 'WPts','LHand', 'WHand', 'LBHand', 'WBHand']
    cols_swapped = [f for f in cols_swapped if f not in features_to_drop]
    
    # Generate labels
    if labels == 'duplicate':
        # swaps winner data with loser data on the whole dataset (duplicates the dataset)
        Y = np.concatenate([np.ones(X.shape[0], dtype=int), np.zeros(X.shape[0], dtype=int)])
        tmp = X.copy()
        tmp[cols_to_swap] = tmp[cols_swapped]
        if 'diff' in features_to_add:
            tmp['RankDiff'] = -1 * tmp['RankDiff']
            tmp['PtsDiff'] = -1 * tmp['PtsDiff']
        if 'elo' in features_to_add:
            tmp[['WEloCalc', 'LEloCalc']] = tmp[['LEloCalc', 'WEloCalc']]
            tmp['ProbaElo'] = 1 - tmp['ProbaElo']
        if 'top10' in features_to_add:
            tmp[['Top10W', 'Top10L']] = tmp[['Top10L', 'Top10W']]
        tmp.index = np.array(range(X.shape[0] + 1, X.shape[0] * 2 + 1))
        X = pd.concat((X, tmp))
    elif labels == 'random':
        # swaps winner data with loser data for a random number of entries (dataset size doesn't change)
        from random import randint
        Y = np.ones(X.shape[0], dtype=int)
        random_rows = X.sample(randint(X.shape[0]//3, X.shape[0]//3*2))
        random_rows[cols_to_swap] = random_rows[cols_swapped]
        if 'diff' in features_to_add:
            random_rows['RankDiff'] = -1 * random_rows['RankDiff']
            random_rows['PtsDiff'] = -1 * random_rows['PtsDiff']
        if 'elo' in features_to_add:
            random_rows[['WEloCalc', 'LEloCalc']] = random_rows[['LEloCalc', 'WEloCalc']]
            random_rows['ProbaElo'] = 1 - random_rows['ProbaElo']
        if 'top10' in features_to_add:
            random_rows[['Top10W', 'Top10L']] = random_rows[['Top10L', 'Top10W']]
        X.update(random_rows)
        for i in random_rows.index:
            Y[i] = 1 - Y[i]
            
    X = X.rename(columns={'WRank':'P1Rank', 'LRank':'P2Rank', 
                          'WPts':'P1Pts', 'LPts':'P2Pts', 
                          'WEloCalc':'P1Elo', 'LEloCalc':'P2Elo', 
                          'WHand':'P1Hand', 'LHand':'P2Hand', 
                          'WBHand':'P1BHand', 'LBHand':'P2LBHand', 
                          'Top10W':'Top10P1', 'Top10L':'Top10P2'})
    if returnElo:
        return X, Y, playersElo
    else:
        return X, Y

