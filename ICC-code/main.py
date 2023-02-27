import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from visualizer import *
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

X_test = pd.DataFrame([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1], [1, 0, 0]], columns = ['a', 'b', 'c'])
y_test = [1, 0, 0, 0, 1, 1]

# Visualizing - dataset 1: dummy dataset
'''
visualize(X_train, y_train, 0, 1, '#ff8093', '#9ca5ff', show_cm = True, max_depth = None)
'''

# Visualizeing - dataset 2: Iris
'''
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['Species']).astype({'Species': 'int32'})

X = df.drop(['Species'], axis=1)
y = df['Species']

visualize(X, y, 1, 2, '#ff8093', '#9ca5ff', show_cm = True, max_depth = None)
'''

# Visualization - dataset 3: titanic
'''
df = pd.read_csv('data/titanic_data.csv')
df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(),inplace=True)
df.fillna(df.mode().iloc[0], inplace = True)
df['Sex']=df['Sex'].replace({'male':0, 'female':1})
one_hot_encoded_embarked = pd.get_dummies(df['Embarked'],drop_first=False)
df = pd.concat([df,one_hot_encoded_embarked],axis=1).drop('Embarked',axis=1)

X = df.drop(['Survived'],axis=1)
y = df['Survived']

visualize(X, y, 0, 1, '#ff8093', '#9ca5ff', show_cm = True, max_depth = 6)
'''

# Visualization - dataset 4: breast cancer
'''
df = pd.read_csv("data/breast_cancer_wisconsin.csv")
df.drop(columns = ['id', 'Unnamed: 32'], inplace=True)
df['diagnosis']=df['diagnosis'].replace({'B':0, 'M':1})
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
visualize(X, y, 0, 1, '#ff8093', '#9ca5ff', show_cm = True, max_depth = 6)
'''

# visualization - dataset 5: Stratification
np.random.seed(seed=0)
group_size = 350

hidden_group_size = int(group_size / 3)
group_A_size = group_size - hidden_group_size
group_B_size = group_size

group_a_att1 = np.random.normal(loc=0.75, scale=2.25, size=(group_A_size, 1))
group_a_att2 = np.random.normal(loc=2, scale=2.5, size=(group_A_size, 1))

group_hidden_a_att1 = np.random.normal(loc=6.5, scale=1.75, size=(hidden_group_size, 1))
group_hidden_a_att2 = np.random.normal(loc=8, scale=1.75, size=(hidden_group_size, 1))

group_b_att1 = np.random.normal(loc=10, scale=2, size=(group_B_size, 1))
group_b_att2 = np.random.normal(loc=2, scale=2, size=(group_B_size, 1))

noise_values = np.random.normal(loc = 5, scale = 2.5, size = (2 * group_size, 1))

class_values = np.array(['A'] * group_A_size + ['Hidden_A'] * hidden_group_size + ['B'] * group_B_size).reshape(-1,1)

att1_values = np.concatenate((group_a_att1, group_hidden_a_att1, group_b_att1))
att2_values = np.concatenate((group_a_att2, group_hidden_a_att2, group_b_att2))

data = np.concatenate((att1_values, att2_values, noise_values, class_values), axis = 1)

df = pd.DataFrame(data, columns = ['att1', 'att2', 'att3', 'class'])

df = df.astype({'att1': 'float64', 'att2': 'float64', 'att3': 'float64'})

df['class'] = df['class'].replace('Hidden_A', 'A')

X = df.drop(['class'],axis=1)
y = df['class']

visualize(X, y, 'A', 'B', '#ff8093', '#9ca5ff', show_cm = True, max_depth = 2)