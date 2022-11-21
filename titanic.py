import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# what sorts of people were more likely to survive?
X_full = pd.read_csv('./data/titanic/train.csv', index_col='PassengerId')
X_test_full = pd.read_csv('./data/titanic/test.csv', index_col='PassengerId')

X = X_full.drop(['Survived'], axis=1)
y = X_full.Survived

X_train_full, X_valid_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# selecting categorical/numerical features
categorical_cols = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == 'object']

numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# print(X_test.shape)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# data preprocessing
numerical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('category', categorical_transformer, categorical_cols)
])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

model = LogisticRegression(random_state=0)
clf = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', model)
])

clf.fit(X_train, y_train)
pred_valid = clf.predict(X_valid)

score = mean_absolute_error(pred_valid, y_test)
print(f"MAE wrt. Validation: {score}")

# confusion matrix
cm = confusion_matrix(y_test, pred_valid)
print(cm)
print(f"Accuracy: {accuracy_score(y_test, pred_valid)}")

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_valid, y_test)
plt.show()

# k fold validation
from sklearn.model_selection import cross_val_score

acc = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(acc.mean()*100))
print("Standard Deviation: {:.2f} %".format(acc.std()*100))





# output = pd.DataFrame({'PassengerId': X_test.index,
#                        'Survived':})

