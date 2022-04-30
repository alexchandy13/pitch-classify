import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('df.csv')
df['class'].replace(['Take', 'Weak/Miss', 'Strong'], [0, 1, 2], inplace=True)
class_dist = df['class'].value_counts()

y = df['class']
X = df.drop(['balls','strikes','class'], axis = 1)

X['release_speed']=(X['release_speed']-X['release_speed'].min())/(X['release_speed'].max()-X['release_speed'].min())
X['release_spin_rate']=(X['release_spin_rate']-X['release_spin_rate'].min())/(X['release_spin_rate'].max()-X['release_spin_rate'].min())

target_names = ['Take', 'Weak/Miss', 'Strong']

pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection = '3d')

ax1.scatter(X_r[y == 0, 0], X_r[y == 0, 1], X_r[y == 0, 2], color='navy', alpha=1, label='Take')
ax1.scatter(X_r[y == 1, 0], X_r[y == 1, 1], X_r[y == 1, 2], color='turquoise', alpha=1, label='Weak/Miss')
ax1.scatter(X_r[y == 2, 0], X_r[y == 2, 1], X_r[y == 2, 2], color='coral', alpha=1, label='Strong')

ax2 = fig.add_subplot(2, 2, 2, projection = '3d')
ax2.scatter(X_r[y == 0, 0], X_r[y == 0, 1], X_r[y == 0, 2], color='navy', alpha=0, label='Take')
ax2.scatter(X_r[y == 1, 0], X_r[y == 1, 1], X_r[y == 1, 2], color='turquoise', alpha=1, label='Weak/Miss')
ax2.scatter(X_r[y == 2, 0], X_r[y == 2, 1], X_r[y == 2, 2], color='coral', alpha=1, label='Strong')

ax3 = fig.add_subplot(2, 2, 3, projection = '3d')
ax3.scatter(X_r[y == 0, 0], X_r[y == 0, 1], X_r[y == 0, 2], color='navy', alpha=0, label='Take')
ax3.scatter(X_r[y == 1, 0], X_r[y == 1, 1], X_r[y == 1, 2], color='turquoise', alpha=0, label='Weak/Miss')
ax3.scatter(X_r[y == 2, 0], X_r[y == 2, 1], X_r[y == 2, 2], color='coral', alpha=1, label='Strong')

plt.savefig('PCA_3D')