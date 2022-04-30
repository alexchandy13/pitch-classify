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

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


fig = plt.figure()
colors = ["navy", "turquoise", "coral"]

ax = fig.add_subplot()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.7, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA (2 Components)")
plt.savefig('PCA_2D')