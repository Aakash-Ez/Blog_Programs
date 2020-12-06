import pandas as pd
from sklearn import svm
import numpy as np
df = pd.read_csv("data.csv",index_col="Index")
print(df.head())

X_value = np.array(df.X).reshape((1000,1))
Y_value = np.array(df.Y).reshape((1000,1))
Labels = np.array(df.Label)
print("X array shape:",X_value.shape)
print("Y array shape:",Y_value.shape)
print("Labels array shape:",Labels.shape)

XY = np.hstack([X_value,Y_value])
print("XY array shape:",XY.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XY, Labels, test_size=0.2, random_state=2)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test,y_test)

clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test,y_test)