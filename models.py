import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from time import time
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

def timeit(fun):
    # This is a decorator function used to log the model construction execution time
    def timed(*args, **kwargs):
        start = time()
        result = fun(*args, **kwargs)
        end = time()
        print('Execution took {:.2f} min'.format((end-start)/60))
        return result
    return timed


def baseline_model(df):
    # "stupid" models, the lower bounds on accuracy that we want to improve upon
    y_true = np.ones(df.shape[0])
    y_pred = (df['WRank'] > df['LRank']).astype(int)
    print('Player with higher rank wins. Accuracy:', round((y_pred == y_true).sum()/len(y_true), 2))
    y_pred = (df['AvgW'] > df['AvgL']).astype(int)
    print('Player with higher avg bet wins. Accuracy:', round((y_pred == y_true).sum()/len(y_true), 2))


@timeit
def build_decision_tree(X_train, Y_train, X_valid, Y_valid, draw_graphs=True, draw_tree=False):
    # Builds a decision tree and performs hyper-parameters tuning
    scores = {}
    criterions = ('gini', 'entropy')
    splitters = ("best", "random")
    depths = list(range(5, 101, 5)) + [None]
    leaves = list(range(10, 201, 10)) + [None]
    for criterion in criterions:
        for splitter in splitters:
            for depth in depths:
                for max_leaves in leaves:
                    dt = DecisionTreeClassifier(max_leaf_nodes=max_leaves,
                                                splitter=splitter,
                                                criterion=criterion,
                                                max_depth=depth)
                    dt.fit(X_train, Y_train)
                    train_acc = round(accuracy_score(y_true=Y_train, y_pred=dt.predict(X_train)), 3)
                    valid_acc = round(accuracy_score(y_true=Y_valid, y_pred=dt.predict(X_valid)), 3)
                    scores[(depth, max_leaves, criterion, splitter)] = (valid_acc, train_acc)
                    
    best_acc = max(scores.values())
    best_params = [params for params, acc in scores.items() if acc == best_acc][0]
    depth, max_leaves, criterion, splitter = best_params
    print('Max accuracy (validation, training):', best_acc)
    print('Criterion:', criterion)
    print('Splitter:', splitter)
    print('Max depth:', depth)
    print('Max leaves:', max_leaves)
    
    dt = DecisionTreeClassifier(max_leaf_nodes=max_leaves,
                                splitter=splitter,
                                criterion=criterion,
                                max_depth=depth)
    dt.fit(pd.concat([X_train, X_valid]), np.concatenate([Y_train, Y_valid]))
    
    if draw_graphs:
        fig, (leaves_ax, depth_ax) = plt.subplots(1, 2, figsize=(12, 6))
        # plotting acc graph for max_leaves hyperparameter
        errors = []
        for tmp_leaves in leaves:
            valid_acc, train_acc = scores[(depth, tmp_leaves, criterion, splitter)]
            errors += [ [tmp_leaves, valid_acc, train_acc] ]
        errors = np.array(errors)
        leaves_ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        leaves_ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        leaves_ax.set_ylabel("Accuracy")
        leaves_ax.set_xlabel("Number of Leaves")
        leaves_ax.grid()
        leaves_ax.legend()
        
        # plotting acc graph for max_depth hyperparameter
        errors = []
        for tmp_depth in depths:
            valid_acc, train_acc = scores[(tmp_depth, max_leaves, criterion, splitter)]
            errors += [ [tmp_depth, valid_acc, train_acc] ]
        errors = np.array(errors)
        depth_ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        depth_ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        depth_ax.set_ylabel("Accuracy")
        depth_ax.set_xlabel("Max tree depth")
        depth_ax.grid()
        depth_ax.legend()
        
    if draw_tree:
        dot_data = export_graphviz(dt, out_file=None, 
                                   feature_names=X_train.columns, class_names=True, rotate=True,
                                   filled=True, rounded=True, special_characters=True)  
        graph = graphviz.Source(dot_data)
        display(graph)
    return dt


@timeit
def build_bagging_classifier(X_train, Y_train, X_valid, Y_valid, draw_graphs=True):
    scores = {}
    n_estimators = list(range(10, 211, 20))
    for bootstrap in (True, False):
        for n_est in n_estimators:
            for max_samples in (0.25, 0.50, 0.75, 1.0):
                for criterion in ('gini', 'entropy'):
                    dt = DecisionTreeClassifier(criterion=criterion)
                    bagged_dt = BaggingClassifier(dt, bootstrap=bootstrap,
                                                  n_estimators=n_est,
                                                  max_samples=max_samples,
                                                  n_jobs=-1)
                    bagged_dt.fit(X_train, Y_train)
                    train_acc = round(accuracy_score(y_true=Y_train, y_pred=bagged_dt.predict(X_train)), 3)
                    valid_acc = round(accuracy_score(y_true=Y_valid, y_pred=bagged_dt.predict(X_valid)), 3)
                    scores[(bootstrap, n_est, max_samples, criterion)] = (valid_acc, train_acc)
                    
    best_acc = max(scores.values())
    best_params = [params for params, acc in scores.items() if acc == best_acc][0]
    bootstrap, n_est, max_samples, criterion = best_params
    print('Max accuracy (validation, training):', best_acc)
    print('Boostrap:', bootstrap)
    print('N. estimators:', n_est)
    print('Max samples:', max_samples)
    print('Tree criterion:', criterion)
    
    bagged_dt = BaggingClassifier(DecisionTreeClassifier(criterion=criterion), 
                                  bootstrap=bootstrap,
                                  n_estimators=n_est, 
                                  max_samples=max_samples)
    bagged_dt.fit(pd.concat([X_train, X_valid]), np.concatenate([Y_train, Y_valid]))
    
    if draw_graphs:
        fig, ax = plt.subplots(figsize=(6, 6))
        # plotting acc graph for n_estimators hyperparameter
        errors = []
        for tmp_n_est in n_estimators:
            valid_acc, train_acc = scores[(bootstrap, tmp_n_est, max_samples, criterion)]
            errors += [ [tmp_n_est, valid_acc, train_acc] ]
        errors = np.array(errors)
        ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Number of estimators")
        ax.grid()
        ax.legend()
    
    return bagged_dt

@timeit
def build_adaboost(X_train, Y_train, X_valid, Y_valid, draw_graphs=True):
    scores = {}
    n_estimators = list(range(50, 501, 50))
    learning_rates = (0.01, 0.05, 0.1, 0.25, 0.50)
    for n_est in n_estimators:
        for learning_rate in learning_rates:
            for criterion in ('gini', 'entropy'):
                boosted_dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, criterion=criterion),
                                                n_estimators=n_est,
                                                learning_rate=learning_rate)
                boosted_dt.fit(X_train, Y_train)
                train_acc = round(accuracy_score(y_true=Y_train, y_pred=boosted_dt.predict(X_train)), 3)
                valid_acc = round(accuracy_score(y_true=Y_valid, y_pred=boosted_dt.predict(X_valid)), 3)
                scores[(n_est, learning_rate, criterion)] = (valid_acc, train_acc)
                    
    best_acc = max(scores.values())
    best_params = [params for params, acc in scores.items() if acc == best_acc][0]
    n_est, learning_rate, criterion = best_params
    print('Max accuracy (validation, training):', best_acc)
    print('N. estimators:', n_est)
    print('Learning rate:', learning_rate)
    print('Tree criterion:', criterion)
    
    boosted_dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, criterion=criterion),
                                    n_estimators=n_est,
                                    learning_rate=learning_rate)
    boosted_dt.fit(pd.concat([X_train, X_valid]), np.concatenate([Y_train, Y_valid]))
    
    if draw_graphs:
        fig, (est_ax, rate_ax) = plt.subplots(1, 2, figsize=(12, 6))
        # plotting acc graph for n_estimators hyperparameter
        errors = []
        for tmp_n_est in n_estimators:
            valid_acc, train_acc = scores[(tmp_n_est, learning_rate, criterion)]
            errors += [ [tmp_n_est, valid_acc, train_acc] ]
        errors = np.array(errors)
        est_ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        est_ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        est_ax.set_ylabel("Accuracy")
        est_ax.set_xlabel("Number of estimators")
        est_ax.grid()
        est_ax.legend()
        
        # plotting acc graph for learning_rate hyperparameter
        errors = []
        for tmp_learning_rate in learning_rates:
            valid_acc, train_acc = scores[(n_est, tmp_learning_rate, criterion)]
            errors += [ [tmp_learning_rate, valid_acc, train_acc] ]
        errors = np.array(errors)
        rate_ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        rate_ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        rate_ax.set_ylabel("Accuracy")
        rate_ax.set_xlabel("Learning rate")
        rate_ax.grid()
        rate_ax.legend()
    return boosted_dt


@timeit
def build_random_forest(X_train, Y_train, X_valid, Y_valid, draw_graphs=True):
    scores = {}
    n_estimators = list(range(100, 1000, 100))
    MAX_DEPTH = 20
    for n_est in n_estimators:
        for criterion in ('gini', 'entropy'):
            for bootstrap in (True, False):
                for n_features in (None, 'sqrt', 'log2'):
                    rf = RandomForestClassifier(n_estimators=n_est,
                                                bootstrap=bootstrap,
                                                criterion=criterion,
                                                max_features=n_features,
                                                max_depth=MAX_DEPTH,
                                                n_jobs=-1)
                    rf.fit(X_train, Y_train)
                    train_acc = round(accuracy_score(y_true=Y_train, y_pred=rf.predict(X_train)), 3)
                    valid_acc = round(accuracy_score(y_true=Y_valid, y_pred=rf.predict(X_valid)), 3)
                    scores[(n_est, criterion, bootstrap, n_features)] = (valid_acc, train_acc)
                    
    best_acc = max(scores.values())
    best_params = [params for params, acc in scores.items() if acc == best_acc][0]
    n_est, criterion, bootstrap, n_features = best_params
    print('Max accuracy (validation, training):', best_acc)
    print('N. estimators:', n_est)
    print('Criterion:', criterion)
    print('Bootstrap:', bootstrap)
    print('Features criterion (None means all features):', n_features)
    print('Max tree depth:', MAX_DEPTH)
    
    rf = RandomForestClassifier(n_estimators=n_est,
                                bootstrap=bootstrap,
                                criterion=criterion,
                                max_features=n_features,
                                max_depth=MAX_DEPTH,
                                n_jobs=-1)
    rf.fit(pd.concat([X_train, X_valid]), np.concatenate([Y_train, Y_valid]))
    
    if draw_graphs:
        fig, est_ax = plt.subplots(figsize=(6, 6))
        # plotting acc graph for n_estimators hyperparameter
        errors = []
        for tmp_n_est in n_estimators:
            valid_acc, train_acc = scores[(tmp_n_est, criterion, bootstrap, n_features)]
            errors += [ [tmp_n_est, valid_acc, train_acc] ]
        errors = np.array(errors)
        est_ax.plot(errors[:,0], errors[:,1], "x:", label="Validation")
        est_ax.plot(errors[:,0], errors[:,2], "o-", label="Train")
        est_ax.set_ylabel("Accuracy")
        est_ax.set_xlabel("Number of estimators")
        est_ax.grid()
        est_ax.legend()
        
    return rf


