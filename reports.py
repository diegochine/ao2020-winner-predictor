import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

def feature_importance(X, model):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(range(0,X.shape[1]), model.feature_importances_)
    ax.set_title("Feature Importances")
    for i, f in enumerate(model.feature_importances_):
        print('{:2}'.format(i), ' -> ', X.columns[i])
        
        
def model_decision_boundary(models, X, y):
    print('Approximate decision boundaries for the different models')
    # We need PCA to make the data 2D (approximately)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_proj = pca.transform(X)
    
    x_min, x_max = X_proj[:, 0].min() - .1, X_proj[:, 0].max() + .1
    y_min, y_max = X_proj[:, 1].min() - .1, X_proj[:, 1].max() + .1
    zz = [ [xx,yy] for xx in np.linspace(x_min, x_max, 50) 
                   for yy in np.linspace(y_min, y_max, 50) ]
    zz = np.array(zz)
    fig, ax_lst = plt.subplots(2, 2, figsize=(12, 12))
    fig.tight_layout(pad=3.0)
    
    for model, ax in zip(models, ax_lst.flatten()):
        algo, model = model
        z_labels = model.predict(pca.inverse_transform(zz))
        ax.scatter(zz[:,0], zz[:,1], c=z_labels, marker='+', alpha=0.3)
        ax.scatter(X_proj[:,0], X_proj[:,1], c=y, alpha=0.5)
        ax.set_title(algo)
        
def report(X, Y, models):
    for algo, model in models:
        print('Algorithm:', algo)
        rep = classification_report(y_true=Y, y_pred=model.predict(X))
        print(rep)
        print()