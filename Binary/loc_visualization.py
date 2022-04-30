from cv2 import transform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv('bin-df.csv')
df['class'].replace(['Take', 'Swing'], [0, 1], inplace=True)
class_dist = df['class'].value_counts()

y = df['class']
X_r = df.drop(["release_speed","balls","strikes","pfx_x","pfx_z","release_spin_rate","class"], axis = 1)

X_r = np.array(X_r)
target_names = ['Take', 'Swing']

fig = plt.figure()
colors = ["navy", "coral"]

ax = fig.add_subplot()
for color, i, target_name in zip(colors, [0, 1], target_names):
    ax.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.7, label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("Plate X vs Plate Z")
plt.savefig('Plate X vs Plate Z')