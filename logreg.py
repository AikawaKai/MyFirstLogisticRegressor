from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from numpy import exp, dot, log, ones, zeros, concatenate, linspace
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from numpy import log, logspace, meshgrid, arange, full, array
from mpl_toolkits.mplot3d import Axes3D


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
        self.losses = []

    # we add a vector of ones as intercept in the X matrix  (we have another weight w0 for this intercept "feature")
    @staticmethod
    def add_intercept(X):
        intercept = ones((X.shape[0], 1))
        return concatenate((intercept, X), axis=1)

    # sigmoid function
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + exp(-z))

    # dot product between input and weights
    @staticmethod
    def get_z(X, theta):
        return dot(X, theta)

    # loss function to check how well the predictor / regressor works
    # h = predictions / hypothesis
    # y = real labels
    @staticmethod
    def loss(h, y):
        return (-y * log(h) - (1 - y) * log(1 - h)).mean()

    # derivative of loss function
    @staticmethod
    def loss_gradient(X, y, h):
        return dot(X.T, (h - y)) / y.shape[0]

    # updating weights with gradient descent
    def __update_theta(self, gradient):
        self.theta -= self.lr * gradient
        return self.theta

    def fit(self, X, y):
        if self.fit_intercept:
            X = LogisticRegression.add_intercept(X)
        # weights initialization
        self.theta = zeros(X.shape[1])
        self.theta = array([0.0, -5.0, 2.0])
        to_save = [int(z * log(z)) for z in range(1, 200)]
        print(to_save)

        for i in range(self.num_iter):
            z = LogisticRegression.get_z(X, self.theta)
            h = LogisticRegression.sigmoid(z)
            gradient = LogisticRegression.loss_gradient(X, y, h)
            self.__update_theta(gradient)
            if self.verbose and i in to_save:
                self.thetas.append(deepcopy(self.theta))
                z = LogisticRegression.get_z(X, self.theta)
                h = LogisticRegression.sigmoid(z)
                loss = LogisticRegression.loss(h, y)
                print(f'loss: {loss} \t')
                print("theta: ", self.theta[1])
                self.losses.append(loss)
        return self.thetas, self.losses, to_save

    def predict_prob(self, X):
        if self.fit_intercept:
            X = LogisticRegression.add_intercept(X)
        return LogisticRegression.sigmoid(LogisticRegression.get_z(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


def func_z(X, y, w1, w2, w3):
    n1, n2 = w1.shape
    res = full((n1, n2), None)
    for i in range(n1):
        for j in range(n2):
            curr_w1 = w1[i, j]
            curr_w2 = w2[i, j]
            curr_w3 = w3[i, j]
            z = LogisticRegression.get_z(X, array([curr_w1, curr_w2, curr_w3]))
            h = LogisticRegression.sigmoid(z)
            y_ = LogisticRegression.loss(h, y)
            res[i, j] = y_
    return res


def _3d_plot(weights, X_train, y_train, losses):
    w1_ = [weights[i][0] for i in range(len(weights))]
    w2_ = [weights[i][1] for i in range(len(weights))]
    w3_ = [weights[i][2] for i in range(len(weights))]
    print(max(w2_), min(w2_))
    print(max(w3_), min(w3_))
    intercept = weights[-1][0]
    w2, w3 = meshgrid(arange(-6, 5, 0.1), arange(-6, 5, 0.1))
    lato = len(arange(-6, 5, 0.1))
    w1 = full((lato, lato), intercept)
    X_ = LogisticRegression.add_intercept(X_train)
    print(X_.shape)
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    _3d = ax1.plot_surface(w2, w3, func_z(X_, y_train, w1, w2, w3), edgecolor='blue', rstride=8,
                           cstride=5, cmap='jet')
    ax1.plot([w2_[-1]], [w3_[-1]], losses[-1], 'r*', markersize=10)
    line, = ax1.plot([], [], [], 'r-', label='Gradient descent', lw=1.5)
    point, = ax1.plot([], [], [], 'bo')
    display_value = ax1.text(2., 2., 27.5, '', transform=ax1.transAxes)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        display_value.set_text('')

        return line, point, display_value

    def animate(i):
        # Animate line
        line.set_data(w2_[:i], w3_[:i])
        line.set_3d_properties(losses[:i])

        # Animate points
        point.set_data(w2_[i], w3_[i])
        point.set_3d_properties(losses[i])

        # Animate display value
        display_value.set_text('Min = ' + str(losses[i]))

        return line, point, display_value

    ax1.legend(loc=1)

    anim = FuncAnimation(fig1, animate, init_func=init,
                         frames=len(losses), interval=120,
                         repeat_delay=60, blit=True, repeat=True)
    anim.save("3d.gif", dpi=120, writer='imagemagick')
    plt.show()


def _2d_plot(weights, losses, to_save, X, y, filter_):
    weights = weights[filter_:]
    losses = losses[filter_:]
    to_save = to_save[filter_:]

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
        line.set_data(x, -((weights[i][1] / weights[i][2]) * x + weights[i][0] / weights[i][2]))
        plt.title("Iteration {}".format(to_save[i]))
        return line, ax

    anim = FuncAnimation(fig, update_plot, frames=range(len(weights)), interval=200)
    anim.save("2d.gif", dpi=80, writer='imagemagick')
    plt.show()


def main():
    iris = load_iris()  # loading Dataset
    X = iris.data[:, :2]  # two dimension out of 4
    y = (iris.target != 0) * 1  # flattening from 3 classes to 2 classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=11)
    lr = LogisticRegression(learning_rate=0.1, num_iterations=300000, verbose=True)
    weights, losses, to_save = lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    print((preds == y_test).mean())
    vector_weights = lr.theta
    print(vector_weights)
    _3d_plot(weights, X_train, y_train, losses)

    # plot_iris_with_boundaries(X, y, vector_weights, True)


main()
