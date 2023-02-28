import numpy as np
import pandas as pd
from sklearn.datasets import load_iris # Reference: https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['Species']).astype({'Species': 'int32'})

df = df[(df["Species"] == 1) | (df["Species"] == 2)]

import matplotlib.colors as clr
from sklearn.tree import DecisionTreeClassifier

cmap = clr.LinearSegmentedColormap.from_list('custom cmap',
                                             [(0,    'blue'),
                                              (0.5,  'white'),
                                              (1,    'red')], N=256)

def plot_decision_boundary(X, y, classifier):
  x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:,0].max() + 0.1
  y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min,x_max, 100), np.linspace(y_min, y_max, 100))
  x_in = np.c_[xx.ravel(), yy.ravel()]
  y_pred = classifier.predict(pd.DataFrame(x_in, columns = X.columns))
  if isinstance(y_pred, pd.Series):
    y_pred = y_pred.to_numpy()
  y_pred = np.round(y_pred).reshape(xx.shape)
  plt.contourf(xx, yy, y_pred, cmap=cmap, alpha=0.7 )
  plt.scatter(X.iloc[:,0], X.iloc[:, 1], c=y, s=40, cmap=cmap)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.xlabel("worst concave points")
  plt.ylabel("worst radius")
  plt.show()

X = df.drop(['Species', 'sepal length (cm)', 'petal width (cm)'], axis=1)
y = df['Species']

# Visualizing decision tree decision bounady
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X, y)
plot_decision_boundary(X, y, tree_classifier)