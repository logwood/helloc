from sklearn import datasets		# 存放鸢尾花数据
from sklearn.cluster import KMeans	# 机器学习模型
import matplotlib.pyplot as plt
import pandas as pd

iris = datasets.load_iris()
iris_X = iris.data				# 花朵属性
iris_y = iris.target			# 花朵类别
plt.scatter(iris_X[:50,2],iris_X[:50,3],label='setosa',marker='o')
plt.scatter(iris_X[50:100,2],iris_X[50:100,3],label='versicolor',marker='x')
plt.scatter(iris_X[100:,2],iris_X[100:,3],label='virginica',marker='+')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("actual result")
plt.legend()
plt.show()
km = KMeans(n_clusters=3)			# 设定簇的定值为3 
km.fit(iris_X)						# 对数据进行聚类
num = pd.Series(km.labels_).value_counts()
print(num)
y_train = pd.Series(km.labels_)
y_train.rename('res',inplace=True)
print(y_train)
result = pd.concat([pd.DataFrame(iris_X),y_train],axis=1)
print(result)
Category_one = result[result['res'].values == 0]
k1 = result.iloc[Category_one.index]
Category_two = result[result['res'].values == 1]
k2 = result.iloc[Category_two.index]
Category_three = result[result['res'].values == 2]
k3 =result.iloc[Category_three.index]
plt.scatter(iris_X[:50,2],iris_X[:50,3],label='setosa',marker='o',c='yellow')
plt.scatter(iris_X[50:100,2],iris_X[50:100,3],label='versicolor',marker='o',c='green')
plt.scatter(iris_X[100:,2],iris_X[100:,3],label='virginica',marker='o',c='blue')
plt.scatter(k1.iloc[:,2],k1.iloc[:,3],label='cluster_one',marker='+',c='brown')
plt.scatter(k2.iloc[:,2],k2.iloc[:,3],label='cluster_two',marker='+',c='red')
plt.scatter(k3.iloc[:,2],k3.iloc[:,3],label='cluster_three',marker='+',c='black')
plt.xlabel('petal length')			
plt.ylabel('petal width')			
plt.title("result of KMeans")
plt.legend()
plt.show()
