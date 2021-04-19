from tensorflow import keras
from keras.datasets import mnist
import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import joblib

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_new, y_train_new = x_train[(y_train == 0) | (y_train == 1)], y_train[(y_train == 0) | (y_train == 1)]

x_train_final = x_train_new.reshape((-1, 784))

x_test_new, y_test_new = x_test[(y_test == 0) | (y_test == 1)], y_test[(y_test == 0) | (y_test == 1)]

x_test_final = x_test_new.reshape((-1, 784))

x_train_final = x_train_final / 255
x_test_final = x_test_final / 255

param = [
  {
    "kernel": ["linear"],
    "C": [1, 10, 100, 1000]
  },
  {
    "kernel": ["rbf"],
    "C": [1, 10, 100, 1000],
    "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
  }
]

svm = SVC(probability = True)
clf = GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)
clf.fit(x_train_final, y_train_new)

print("\nBest Params:")
print(clf.best_params_)
y_predict = clf.predict(x_test_final)
print(classification_report(y_test_new, y_predict))
