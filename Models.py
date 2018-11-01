import numpy as np
import scipy as sc


class LogisticRegression():
    def __init__(self, alpha=0.00001, iter_num=1000, delta=10, verbose=True):
        self.weights = None
        self.iter_num = iter_num
        self._target_num = None
        self._multi_class = False
        self.alpha = alpha
        self.delta = delta
        self._verbose = verbose

    def _init_weights(self, features_num, weights_borders):
        a, b = weights_borders
        return a + (b - a) * np.random.random(features_num)

    def fit(self, X, y):
        self._target_num = np.unique(y).shape[0]
        self._gradient_descent(X, y)

    def sigm(self, X, weights):
        return 1 / (1 + np.exp(-np.matmul(X, weights)))

    def predict(self, X, sigma=None):
        prob = []
        if self._multi_class:

            for i in range(self._target_num):
                prob.append(self.sigm(X, self.weights[i, :]))

            prob = np.array(prob)
            prob = prob.T
            if not sigma:
                pred = prob.argmax(1)

            else:

                pred = []
                for i in range(X.shape[0]):
                    tmp = 0
                    for j in range(self._target_num):
                        if X[i][j] >= sigma[j]:
                            if not tmp:
                                tmp = X[i][j]
                                pred.append(j)

                            elif X[i][j] > tmp:
                                pred[len(pred) - 1] = j
                                tmp = X[i][j]
        else:
            if not sigma:
                sigma = 0.5
            prob = self.sigm(X, self.weights)
            pred = []
            for i in range(prob.shape[0]):
                if prob[i] >= sigma:
                    pred.append(1)
                else:
                    pred.append(-1)

        return np.array(pred)

    def _num_gradient_descent(self, X, y, weight):
        last_loss = 0
        alpha = self.alpha
        delta = self.delta
        best_loss = float("inf")
        best_weight = weight

        for Iter in range(self.iter_num):

            scalar = np.matmul(X, weight.T)

            grad = np.matmul(((-(y * np.exp(- y * scalar)) / (np.exp(-y * scalar) + 1))).T, X)
            loss = np.average(self.log_loss(X, y, weight))

            if loss == float("inf"):
                break

            if loss < best_loss:
                best_weight = weight
                best_loss = loss

            if self._verbose and Iter % 100 == 0:
                print(Iter + 1, loss)

            if self.delta:

                if last_loss < loss:
                    weight = best_weight
                    alpha -= alpha / (delta / 10)

                else:
                    alpha += alpha / (delta)

                last_loss = loss

            weight = weight - alpha * np.array(grad)
        print("best loss:", best_loss)
        return best_weight

    def _gradient_descent(self, X, y):
        if self._target_num > 2:
            self._multi_class = True

        if self._multi_class:
            random_weight = self._init_weights((self._target_num, X.shape[1]), (1, -1))
            y_bin = self._change_classification_targets(y)
            self.weights = self._num_gradient_descent(X, y_bin, random_weight)

        elif list(np.unique(y)) == [-1, 1]:
            weight = self._init_weights(X.shape[1], (1, -1))
            self.weights = self._num_gradient_descent(X, y, weight)

        else:
            targets = np.unique(y)
            ind0 = np.where(y == targets[0])
            y[ind0] = -1
            ind1 = np.where(y == targets[1])
            y[ind1] = 1

            weight = self._init_weights(X.shape[1], (1, -1))
            self.weights = self._num_gradient_descent(X, y, weight)

    def log_loss(self, X, y, W):
        return np.sum(np.log(1 + np.exp(-y * (np.matmul(X, W.T)))), axis=1)

    def _change_classification_targets(self, y):
        y_new = np.identity(self._target_num)[y[:]]
        ind = np.where(y_new == 0)
        y_new[ind] = -1

        return y_new


class KNN():
    def __init__(self, K = 5):
        self._X_train = None
        self._y_train = None
        self.K = K

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train

    def predict(self,X):
        return self._num_neighbors(X)

    def _num_neighbors(self, X):
        dist = sc.spatial.distance.cdist(X, self._X_train)
        neighbors = self._y_train[np.argpartition(dist, self.K)[:,:self.K]]

        pred = []
        for i in range(X.shape[0]):
            counts = np.bincount(neighbors[i])
            pred.append(counts.argmax())

        return pred


