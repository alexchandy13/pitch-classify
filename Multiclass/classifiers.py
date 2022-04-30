from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pca_4
import pandas as pd

df_pca = pca_4.get_pca()

X = df_pca.drop(['class'], axis = 1)
y = df_pca['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

class_dist = y_train.value_counts()
print("Class Distribution:")
print(class_dist / len(y_train))

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

prfs = precision_recall_fscore_support(y_test, y_pred)

scores = pd.DataFrame(prfs[:-1],columns=['Take','Weak/Miss','Strong'])
scores.index = ['Precision','Recall','F1-Score']

print(scores)