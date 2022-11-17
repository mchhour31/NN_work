import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics # for accuracy calc

from six import StringIO
from IPython.display import Image
import pydotplus

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=0, names=col_names)

print(pima.head())

# feature selection
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
x = pima[feature_cols] # features (independent vars)
y = pima.label # target vars (dependent variables)

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# build model
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test)
print(f"MAE: {metrics.mean_absolute_error(y, y_pred)}")

# evaluate the model
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True, 
                special_characters=True, feature_names=feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes2.png')
Image(graph.create_png())