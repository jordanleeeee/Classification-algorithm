from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf.fit(iris.data, iris.target))

X = [[1,1,0],[0,1,1],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,0,0]]
y = [1,1,1,1,0,0,0,0]
