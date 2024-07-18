import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Função para criar partições fuzzy (manual)
def create_fuzzy_partitions(X):
    partitions = {}
    for feature in X.columns:
        data = X[feature]
        min_val, max_val = data.min(), data.max()
        x_range = np.linspace(min_val, max_val, 100)
        partitions[feature] = {
            'low': np.maximum(0, np.minimum(1, (max_val - x_range) / (max_val - min_val))),
            'medium': np.maximum(0, np.minimum((x_range - min_val) / (max_val - min_val), (max_val - x_range) / (max_val - min_val))),
            'high': np.maximum(0, np.minimum(1, (x_range - min_val) / (max_val - min_val)))
        }
    return partitions

# Função para plotar as partições fuzzy
def plot_fuzzy_partitions(partitions):
    fig, axes = plt.subplots(len(partitions), 1, figsize=(10, len(partitions) * 3))
    for i, feature in enumerate(partitions.keys()):
        ax = axes[i]
        for label, partition in partitions[feature].items():
            ax.plot(np.linspace(X_train[feature].min(), X_train[feature].max(), 100), partition, label=f'{label}')
        ax.set_title(f'Partições Fuzzy para {feature}')
        ax.legend()
    plt.tight_layout()
    plt.show()

# Criar e plotar partições fuzzy
partitions = create_fuzzy_partitions(X_train)
plot_fuzzy_partitions(partitions)
