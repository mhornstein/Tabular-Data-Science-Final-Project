import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from path_visualizer import *
from misclassification_visualizer import *
from decision_path_extractor import *

# Creating a dataset to teach the tree to classify the boolean operation a & b | c.
a = [0] * 4 + [1] * 4
b = (([0] * 2) + [1] * 2) * 2
c = [0,1] * 4
d = [(_a & _b | _c) for _a, _b, _c in zip(a,b,c)]

data = {'a': a, 'b': b, 'c': c, 'Class': d}
df = pd.DataFrame(data)

X_train = df.iloc[:, [0,1,2]]

y_train = df.iloc[:,[3]]

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

# Confusion path plotting - proposal v1

# plot_confusion_path(clf, [[1,1,0], [1,0,1]], [1, 0])
# plot_confusion_path(clf, [[1,1,0], [1,0,1]], [1, 0], present='visited')
# plot_confusion_path(clf, [[1,0,1]], [0], present='visited')
# (clf, X, y, color = '#9900ffff', present = 'all', cm_indices='all', show_cm = True)
# plot_confusion_path(clf, [[1,1,0], [1,0,1]], [1, 0], cm_indices=[(0,1)])
# plot_confusion_path(clf, [[1,1,0], [1,0,1]], [1, 0], cm_indices=[(0,1), (1,1)])
# plot_confusion_path(clf, [[1,1,0], [1,0,1]], [1, 0], cm_indices=[(0,1), (1,0)])

# common_nodes_in_paths(clf, [[1,1,0], [1,0,1], [1,0,0]])
# common_nodes_in_paths(clf, [[1,1,0], [1,0,0]])
# common_nodes_in_paths(clf, [[0,0,0], [0,0,0]])

###############################################

# Misclassification path plotting - proposal v2

X = pd.DataFrame([[1,1,0],[1,0,1],[1,1,1],[0,0,0],[0,1,1],[1,0,0]], columns = ['a','b','c'])
y = [1, 0, 0, 0,1,1]
# plot_misclassification_in_tree(clf, X, y, 0, 1, 'white', 'red')
samples_to_predicates = extract_decision_predicate_by_path(clf, pd.DataFrame(X))
for sample_id in samples_to_predicates:
    print(f'{sample_id}: {samples_to_predicates[sample_id]}')

plot_misclassification(X_train, y_train, X, y, 0,1,'red', 'blue',show_cm = True, present = 'all', max_depth = 4)
# Trying with iris

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Reading the Iris.csv file
data = load_iris()

# Extracting Attributes / Features
X = data.data

# Extracting Target / Class Labels
y = data.target

from sklearn.tree import DecisionTreeClassifier

# Versicolor vs virginica
'''
clf = DecisionTreeClassifier()

_X = X[50:,:]
_y = y[50:]

clf.fit(_X,_y)

plot_misclassification_path(clf, _X, _y, 2, 1, 'orange', 'green', features_names = data.feature_names)
'''

# Versicolor vs virginica - limited
'''
clf = DecisionTreeClassifier(max_depth=20)

_X = X[50:,:]
_y = y[50:]

clf.fit(_X,_y)

plot_misclassification_path(clf, _X, _y, 1,2, 'orange', 'green', features_names = data.feature_names)
'''

# setosa vs versicolor
'''
_X = X[:100,:]
_y = y[:100]

clf.fit(_X,_y)

plot_misclassification_path(clf, _X, _y, 0,1, 'blue', 'orange', data.feature_names)
'''