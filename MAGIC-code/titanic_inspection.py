from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from misclassification_visualizer import *
from decision_path_extractor import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("data/titanic_data.csv")

##############################
# Data cleaning and organizing

df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1, inplace=True)

df['Age'].fillna(df['Age'].mean(),inplace=True)

df.fillna(df.mode().iloc[0], inplace = True)

df['Sex']=df['Sex'].replace({'male':0, 'female':1})

one_hot_encoded_embarked = pd.get_dummies(df['Embarked'],drop_first=False)
df = pd.concat([df,one_hot_encoded_embarked],axis=1).drop('Embarked',axis=1)

#################
## Classification

X = df.drop(['Survived'],axis=1)
y = df['Survived']
print("All data size: ",X.shape, y.shape)
X_train11, X_test11, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=41)
print("train size:", X_train11.shape, y_train.shape)
print("test size:" , X_test11.shape, y_test.shape)

scaler = StandardScaler()

# fit and transfrom while keeping the original indeces and columns, see: https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
X_train = pd.DataFrame(scaler.fit_transform(X_train11), index=X_train11.index, columns=X_train11.columns)
X_test = pd.DataFrame(scaler.transform(X_test11), index=X_test11.index, columns=X_test11.columns)

##################################
## Analyzing the misclassification

class_index = 0
attributes = list(df.columns[:class_index]) + list(df.columns[class_index + 1:])

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=10)

clf.fit(X_train,y_train)

expected_label = 1
predicted_label = 0

plot_misclassification_in_tree(clf, X_test, y_test, expected_label, predicted_label, 'green', 'red', show_cm = True, present='misclassified')

classification = clf.predict(X_test)
misclassified_samples_indices = [i for i in range(len(classification)) if classification[i] == predicted_label and y_test.iloc[i] == expected_label]

misclassification_to_predicates = extract_decision_predicate_by_path(clf, X_test.iloc[misclassified_samples_indices])
misclassification_stat_lst = []

history = set([])
representing_samples = []

for i in misclassification_to_predicates:
    if misclassification_to_predicates[i] in history: # avoid repeating the analysis for samples with same predicate
        continue
    else:
        history.add(misclassification_to_predicates[i])
        representing_samples.append(i)

    predicate = eval('lambda x: ' + misclassification_to_predicates[i])

    train_samples = X_train[predicate(X_train)]
    train_samples_classification = y_train[train_samples.index]
    train_classified_as_prediced = sum(train_samples_classification == predicted_label)
    train_classified_as_expcetd = sum(train_samples_classification == expected_label)

    test_samples = X_test[predicate(X_test)]
    test_samples_classification = y_test[test_samples.index]
    test_classified_as_prediced = sum(test_samples_classification == predicted_label)
    test_classified_as_expcetd = sum(test_samples_classification == expected_label)

    misclassification_stat_lst.append([train_classified_as_prediced, train_classified_as_expcetd, test_classified_as_prediced, test_classified_as_expcetd, misclassification_to_predicates[i]])

investigation_results = pd.DataFrame(data = misclassification_stat_lst, index = representing_samples, columns = ['train(predicted)', 'train(excpected)', 'test(prediced)', 'test(expcetd)', 'predicate'])
print(investigation_results.iloc[:,:-1])
print("total misclassification: %d" % sum(investigation_results['test(expcetd)']))


print()
