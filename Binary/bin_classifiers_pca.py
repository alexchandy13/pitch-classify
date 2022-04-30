from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import rbf_kernel
import bin_pca
import pandas as pd
import pickle

df_pca = bin_pca.get_pca()

X = df_pca.drop(['class'], axis = 1)
y = df_pca['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

class_dist = y_train.value_counts()
print("Class Distribution Training:")
print(class_dist / len(y_train))
print()

# KNN
print('k-Nearest Neighbors')
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

acc = knn.score(X_test,y_test)
print("kNN Accuracy: " + str(acc))

y_pred = knn.predict(X_test)

prfs = precision_recall_fscore_support(y_test, y_pred)

scores = pd.DataFrame(prfs[:-1],columns=['Take','Swing'])
scores.index = ['Precision','Recall','F1-Score']

print(scores)
print()

# Gaussian Naive Bayes
print('Gaussian Naive Bayes')

gNB = GaussianNB()

gNB.fit(X_train,y_train)

acc = gNB.score(X_test,y_test)
print("Accuracy: " + str(acc))

y_pred = gNB.predict(X_test)

prfs = precision_recall_fscore_support(y_test, y_pred)

scores = pd.DataFrame(prfs[:-1],columns=['Take','Swing'])
scores.index = ['Precision','Recall','F1-Score']

print(scores)
print()

# Logistic Regression
print('Logistic Regression')
lr = LogisticRegression()

lr.fit(X_train,y_train)

acc = lr.score(X_test,y_test)
print("Accuracy: " + str(acc))

y_pred = lr.predict(X_test)

prfs = precision_recall_fscore_support(y_test, y_pred)

scores = pd.DataFrame(prfs[:-1],columns=['Take','Swing'])
scores.index = ['Precision','Recall','F1-Score']

print(scores)
print()

# Kernelized Logistic Regression
print('Kernelized Logistic Regression')
K_train = rbf_kernel(X_train, X_train, gamma=0.01)
K_test = rbf_kernel(X_test, X_train, gamma=0.01)

klr = LogisticRegression(C=10000)

klr.fit(K_train,y_train)

acc = klr.score(K_test,y_test)
print("Accuracy: " + str(acc))

y_pred = klr.predict(K_test)

prfs = precision_recall_fscore_support(y_test, y_pred)

scores = pd.DataFrame(prfs[:-1],columns=['Take','Swing'])
scores.index = ['Precision','Recall','F1-Score']

print(scores)
print()

# save the models to disk
filename = 'Classifiers/knn_pca.sav'
pickle.dump(knn, open(filename, 'wb')) 

filename = 'Classifiers/gnb_pca.sav'
pickle.dump(gNB, open(filename, 'wb')) 

filename = 'Classifiers/lr_pca.sav'
pickle.dump(lr, open(filename, 'wb')) 

filename = 'Classifiers/klr_pca.sav'
pickle.dump(klr, open(filename, 'wb')) 
