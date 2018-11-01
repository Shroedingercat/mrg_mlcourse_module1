import argparse
import pickle
import numpy as np
from Models import *
from sklearn.metrics import classification_report


def pars():
    default_path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_test_dir', default= default_path)
    parser.add_argument('-y_test_dir', default= default_path)
    parser.add_argument('-model_output_dir', default= default_path)
    parser.add_argument('-model', default= "KNN")

    return parser.parse_args()

def read_csv(path):
    training_data_file = open(path, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    X = []
    y = []
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        X.append(np.asfarray(all_values[1:]) / 255.0)
        y.append(all_values[0])

    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    record_num, features_num = X.shape
    bias_f = np.ones((record_num, 1))
    X = np.hstack((bias_f, X))

    return X, y

if __name__ == '__main__':
    print("predict is start")
    pars_arg = pars()
    X_test_dir = pars_arg.x_test_dir
    y_train_dir = pars_arg.y_test_dir
    if pars_arg.model_output_dir:
        weights_dir = pars_arg.model_output_dir + "/"



    X_train, y_train = read_csv(X_test_dir + '/' + "mnist_train.csv")
    X_test, y_test = read_csv(X_test_dir + '/' + "mnist_test.csv")

    with open(weights_dir + "weights.pickle", 'rb') as f:
        weights = pickle.load(f)

    if weights == "KNN":
        knn = KNN(3)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)

    else:
        log = LogisticRegression()
        log.weights = np.array(weights)
        log._multi_class = True
        log._target_num = 10
        pred = log.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=pred))
