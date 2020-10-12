from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


irisRaw = load_iris()


iris = pd.DataFrame(data= np.c_[irisRaw['data'], irisRaw['target']],
                     columns= irisRaw['feature_names'] + ['target'])


print("DF has: {} columns and {} rows".format(len(iris.columns), len(iris)))


iris.head()


iris.describe()


iris.groupby("target").count()


iris.groupby("target").mean()


iris.groupby("target").std()


iris_no_na = iris.dropna(axis=1)


print("DF after dropping NaN vals has: {} columns and {} rows".format(len(iris_no_na.columns), len(iris_no_na)))


iris.sort_values(by='sepal width (cm)')


print("Entry with maximum petal len is at index {}\nEntry with minimum petal len is at index {}".format(iris["petal length (cm)"].idxmax(), iris["petal length (cm)"].idxmin()))


iris.std()


iris[iris['sepal length (cm)']>iris['sepal length (cm)'].mean()]


melted = iris.melt(id_vars='target')


f = sns.FacetGrid(melted, col='target', row='variable')
f = f.map(plt.hist, 'value')



