import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_curve

def feature_importance(X, model, name):
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title('Features importance for ' + name)

        
def model_decision_boundary(models, X, y):
    print('Approximate decision boundaries for the different models')
    sns.set(style="white")
    # We need PCA to make the data 2D (approximately)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_proj = pca.transform(X)
    
    x_min, x_max = X_proj[:, 0].min() - .1, X_proj[:, 0].max() + .1
    y_min, y_max = X_proj[:, 1].min() - .1, X_proj[:, 1].max() + .1
    zz = [ [xx,yy] for xx in np.linspace(x_min, x_max, 50) 
                   for yy in np.linspace(y_min, y_max, 50) ]
    '''
    #from here
    model = models[0][1]
    xx, yy = np.mgrid[x_min:x_max:1, y_min:y_max:1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(pca.inverse_transform(grid))[:, 1].reshape(xx.shape)
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.scatter(X_proj[:,0], X_proj[:, 1], c=y[:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    ax.set(aspect="equal",
           xlim=(x_min, x_max), ylim=(y_min, y_max),
           xlabel="$X_1$", ylabel="$X_2$")
    
    '''
    fig, ax_lst = plt.subplots(2, 2, figsize=(12, 12))
    fig.tight_layout(pad=3.0)
    
    for model, ax in zip(models, ax_lst.flatten()):
        algo, model = model
        z_labels = model.predict(pca.inverse_transform(zz))
        ax.scatter(zz[:,0], zz[:,1], c=z_labels, marker='+', alpha=0.3)
        ax.scatter(X_proj[:,0], X_proj[:,1], c=y, alpha=0.5)
        ax.set_title(algo)
    
        
def report(X, Y, models):
    scores = []
    for algo, model in models:
        print('Algorithm:', algo)
        rep = classification_report(y_true=Y, y_pred=model.predict(X))
        print(rep)
        print()
        rep_dict = classification_report(y_true=Y, y_pred=model.predict(X), output_dict=True)
        scores += [(rep_dict['accuracy'], algo)]
    best_model = max(scores)
    print('Best model is {} with accuracy of {}'.format(best_model[1], best_model[0]))