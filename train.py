import argparse
import pickle
import numpy as np
from Models import *


def pars():
    default_path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_train_dir', default= default_path)
    parser.add_argument('-y_train_dir', default= default_path)
    parser.add_argument('-model_output_dir', default= default_path)
    parser.add_argument('-model', default= "KNN")

    return parser.parse_args()

#https://pjreddie.com/projects/mnist-in-csv/
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

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
    print("Training is start")
    pars_arg = pars()
    X_train_dir = pars_arg.x_train_dir
    y_train_dir = pars_arg.y_train_dir
    if pars_arg.model_output_dir:
        weights_dir = pars_arg.model_output_dir + "/"
    else:
        weights_dir = pars_arg.model_output_dir
    model = pars_arg.model

    try:
        open(X_train_dir + '/' + "mnist_train.csv")

    except:
        print("data is reading")
        convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
                "mnist_train.csv", 48080)
        convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
                "mnist_test.csv", 10000)

    X, y = read_csv(X_train_dir  +  '/' + "mnist_train.csv")

    if model == "KNN":
        with open(weights_dir +"weights.pickle", 'wb') as f:
            pickle.dump("KNN", f)

    else:
        log = LogisticRegression(iter_num=10, delta=22, alpha=0.00029)
        log.fit(X, y)
        with open(weights_dir + "weights.pickle", 'wb') as f:
            pickle.dump(log.weights, f)

    print("Training is over")

