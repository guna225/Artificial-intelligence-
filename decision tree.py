# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics

# Load a sample dataset (Iris dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create a Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train the Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# Optional: Visualize the Decision Tree (requires graphviz and pydotplus)
# from sklearn.tree import export_graphviz
# from six import StringIO
# from IPython.display import Image
# import pydotplus

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = iris.feature_names, class_names=iris.target_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('iris_decision_tree.png')
# Image(graph.create_png())
