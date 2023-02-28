from misclassification_visualizer import *
from decision_path_extractor import *

# Creating a dataset to teach the tree to classify the boolean operation a & b | c
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

# Creating the test-set
X = pd.DataFrame([[1,1,0],[1,0,1],[1,1,1],[0,0,0],[0,1,1],[1,0,0]], columns = ['a','b','c'])
y = [1, 0, 0, 0,1,1]

samples_to_predicates = extract_decision_predicate_by_path(clf, pd.DataFrame(X))
for sample_id in samples_to_predicates:
    print(f'{sample_id}: {samples_to_predicates[sample_id]}')

plot_misclassification(X_train, y_train, X, y, 0,1,'red', 'blue',show_cm = True, present = 'all', max_depth = 4)