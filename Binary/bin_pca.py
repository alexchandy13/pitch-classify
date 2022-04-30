import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('bin-df.csv')
df['class'].replace(['Take', 'Swing'], [0, 1], inplace=True)

y = df['class']
X = df.drop(['class'], axis = 1)
# X = df.drop(['balls','strikes','class'], axis = 1)

class_dist = y.value_counts()
print("Class Distribution:")
print(class_dist)


X['release_speed']=(X['release_speed']-X['release_speed'].min())/(X['release_speed'].max()-X['release_speed'].min())
X['release_spin_rate']=(X['release_spin_rate']-X['release_spin_rate'].min())/(X['release_spin_rate'].max()-X['release_spin_rate'].min())

# X['release_speed'] = (X['release_speed']-X['release_speed'].mean())/X['release_speed'].std()
# X['release_spin_rate'] = (X['release_spin_rate']-X['release_spin_rate'].mean())/X['release_spin_rate'].std()


target_names = ['Take', 'Swing']

pca = PCA(n_components=6)
X_r = pca.fit(X).transform(X)

cum_sum_exp = np.cumsum(pca.explained_variance_ratio_)
plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig('Explained Variance PCA (Binary)')

df_pca = pd.DataFrame(pca.fit_transform(X), columns = [f"Component {co}" for co in range(0,6)])

y_new = pd.DataFrame(y, columns = ["class"]).reset_index(drop=True)

df_pca = pd.concat([df_pca, y_new], axis = 1)
palette = {0:"navy",
           1:"coral"}

sns.pairplot(data=df_pca, hue = 'class', palette=palette)
plt.savefig('PCA Components (Binary)')

def get_pca():
    return df_pca