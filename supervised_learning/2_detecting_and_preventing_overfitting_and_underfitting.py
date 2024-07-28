from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

X,y = make_classification(n_samples= 9000, n_features= 18, n_informative= 4, n_redundant= 12, random_state= 4)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3)

# empty list for train_scores and test_scores
train_scores, test_scores = list(), list()

values = [i for i in range(1,21)]

# for loop for model and decision
for i in values:
    model = DecisionTreeClassifier(max_depth= i)
    model.fit(X_train, y_train)
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

#ploting graph
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

# method to prevent overfitting
from sklearn.model_selection import GridSearchCV
param_grid = {'criterion ' : ['gini','entropy'], 'max_depth' : [2,4,6,10,20], 'min_samples_split':[5,10,20,50,100]}
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, n_jobs= -1, scoring= "accuracy")
clf.fit(X_train,y_train)

clf.best_estimator_

print(accuracy_score(y_train, clf.best_estimator_.predict(X_train)))
print(accuracy_score(y_test, clf.best_estimator_.predict(X_test)))

