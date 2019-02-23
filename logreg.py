from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from numpy import exp, dot, log, ones, zeros, concatenate, linspace
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from numpy import log


def plot_iris(X, y, type_, display=False):
    red = [X[i] for i in range(len(y)) if y[i] == 0]
    blue = [X[i] for i in range(len(y)) if y[i] == 1]
    x1 = [x[0] for x in red]
    x2 = [x[1] for x in red]
    plt.scatter(x1, x2, color="red")
    x1 = [x[0] for x in blue]
    x2 = [x[1] for x in blue]
    plt.scatter(x1, x2, color="blue")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.figtext(0.5, 0, "Fig. 1 - {} data".format(type_), wrap=True, horizontalalignment='center', fontsize=9)
    if display:
        plt.show()


def plot_iris_with_boundaries(X, y, weights, display=False):
    x = linspace(4, 8)
    plt.plot(x, -(weights[1]/weights[2]*x + weights[0]/weights[2]))
    plot_iris(X, y, "Total", display)


class LogisticRegression(object):

    def __init__(self, learning_rate=0.01, num_iterations = 100000, fit_intercept=True, verbose=False):
        self.lr = learning_rate
        self.num_iter = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None
        self.thetas = []

    # we add a vector of ones as intercept in the X matrix  (we have another weight w0 for this intercept "feature")
    @staticmethod
    def __add_intercept(X):
        intercept = ones((X.shape[0], 1))
        return concatenate((intercept, X), axis=1)

    # sigmoid function
    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + exp(-z))

    # dot product between input and weights
    @staticmethod
    def __get_z(X, theta):
        return dot(X, theta)

    # loss function to check how well the predictor / regressor works
    # h = predictions / hypothesis
    # y = real labels
    @staticmethod
    def __loss(h, y):
        return (-y * log(h) - (1 - y) * log(1 - h)).mean()

    # derivative of loss function
    @staticmethod
    def __loss_gradient(X, y, h):
        return dot(X.T, (h - y)) / y.shape[0]

    # updating weights with gradient descent
    def __update_theta(self, gradient):
        self.theta -= self.lr * gradient
        return self.theta

    def fit(self, X, y):
        if self.fit_intercept:
            X = LogisticRegression.__add_intercept(X)
        # weights initialization
        self.theta = zeros(X.shape[1])
        to_save = [int(z * log(z)) for z in range(1, 200)][10:]
        print(to_save)
        for i in range(self.num_iter):
            z = LogisticRegression.__get_z(X, self.theta)
            h = LogisticRegression.__sigmoid(z)
            gradient = LogisticRegression.__loss_gradient(X, y, h)
            self.__update_theta(gradient)
            if self.verbose and i in to_save:
                self.thetas.append(deepcopy(self.theta))
                z = LogisticRegression.__get_z(X, self.theta)
                h = LogisticRegression.__sigmoid(z)
                print(f'loss: {LogisticRegression.__loss(h, y)} \t')
        return self.thetas, to_save

    def predict_prob(self, X):
        if self.fit_intercept:
            X = LogisticRegression.__add_intercept(X)
        return LogisticRegression.__sigmoid(LogisticRegression.__get_z(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


def main():
    iris = load_iris()  # loading Dataset
    X = iris.data[:, :2]  # two dimension out of 4
    y = (iris.target != 0) * 1  # flattening from 3 classes to 2 classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=11)
    #plot_iris(X_train, y_train, "Train")
    #plot_iris(X_test, y_test, "Test")
    lr = LogisticRegression(learning_rate=0.1, num_iterations=300000, verbose=True)
    weights, to_save = lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    print((preds == y_test).mean())
    vector_weights = lr.theta
    print(vector_weights)

    fig, ax = plt.subplots()
    plt.xlabel("x1")
    plt.ylabel("x2")
    red = [X[i] for i in range(len(y)) if y[i] == 0]
    blue = [X[i] for i in range(len(y)) if y[i] == 1]
    x1 = [x[0] for x in red]
    x2 = [x[1] for x in red]
    ax.scatter(x1, x2, color="red")
    x1 = [x[0] for x in blue]
    x2 = [x[1] for x in blue]
    ax.scatter(x1, x2, color="blue")
    x = linspace(4, 8)
    line, = ax.plot(x, -((weights[0][1] / weights[0][2]) * x + weights[0][0] / weights[0][2]), 'b-', linewidth=2)

    def update_plot(i):
        # print(i)
        # print("WEIGHTS")
        # print(-((weights[i][1] / weights[i][2]) * x + weights[i][0] / weights[i][2]))
        print("X+{}".format(i))
        # print(x+i)
        line.set_data(x, -((weights[i][1] / weights[i][2]) * x + weights[i][0] / weights[i][2]))
        plt.title("Iteration {}".format(to_save[i]))
        return line, ax

    anim = FuncAnimation(fig, update_plot, frames=range(len(weights)), interval=200)
    anim.save("test.gif", dpi=80, writer='imagemagick')
    plt.show()
    # plot_iris_with_boundaries(X, y, vector_weights, True)


main()
