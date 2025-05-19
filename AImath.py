import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split


# Input argument is weight and a tuple (train_data, target)
def grad_mse(w, xy):
    (x, y) = xy
    (rows, cols) = x.shape

    # Compute the output
    o = np.sum(x * w, axis=1)
    diff = y - o
    diff = diff.reshape((rows, 1))
    diff = np.tile(diff, (1, cols))
    grad = diff * x
    grad = -np.sum(grad, axis=0)
    return grad


# Input argument is weight and a tuple (train_data, target)
def mse(w, xy):
    (x, y) = xy

    # Compute output
    # keep in mind that we're using mse and not mse/m
    # because it would be relevant to the end result
    o = np.sum(x * w, axis=1)
    mse = np.sum((y - o) * (y - o))
    mse = mse / 2
    return mse


# (xy) is the (training_set,target) pair
def batch_stochastic_gradient_descent(max_epochs, threshold, w_init,
                                      obj_func, grad_func, xy, batch_size,
                                      learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    batch_rows = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    mse = 0

    # if batch size == 10, data == 1000, learn with 100 datas and update with mean of that mse, with 1 epoch, 10 batch comesout.
    # Run epochs
    while i < max_epochs and diff > threshold:
        for k in range(batch_size):
            j = rows // batch_size
            if (k + 1) * j > rows:
                x_batch_train = x_train[k * j:, :]
                y_batch_train = y_train[k * j:]
            else:
                x_batch_train = x_train[(k) * j:((k + 1) * j), :]
                y_batch_train = y_train[(k) * j:((k + 1) * j)]

            for x, y in zip(x_batch_train, y_batch_train):
                mse += grad_func(w, (np.array([x]), y))

            mse_mu = mse / x_batch_train.shape[0]
            delta_w = -learning_rate * mse_mu + momentum * delta_w
            w = w + delta_w
            mse = 0

        i += 1

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


def NAG(max_epochs, threshold, w_init, obj_func, grad_func, xy, batch_size, learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    batch_rows = 0
    diff = 1.0e10
    rows = x_train.shape[0]
    mse = 0

    # if batch size == 10, data == 1000, learn with 100 datas and update with mean of that mse, with 1 epoch, 10 batch comesout.
    # Run epochs
    while i < max_epochs and diff > threshold:
        for k in range(batch_size):
            j = rows // batch_size
            if (k + 1) * j > rows:
                x_batch_train = x_train[k * j:, :]
                y_batch_train = y_train[k * j:]
            else:
                x_batch_train = x_train[(k) * j:((k + 1) * j), :]
                y_batch_train = y_train[(k) * j:((k + 1) * j)]

            for x, y in zip(x_batch_train, y_batch_train):
                mse += grad_func(w, (np.array([x]), y))

            mse_mu = mse + (momentum * delta_w) / x_batch_train.shape[0]
            delta_w = -learning_rate * mse_mu + momentum * delta_w
            w = w + delta_w
            mse = 0

        i += 1

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


if __name__ == "__main__":
    # Load the digits dataset with two classes
    digits, target = dt.load_digits(n_class=2, return_X_y=True)

    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        digits, target, test_size=0.2, random_state=10)

    # Add a column of ones to account for bias in train and test
    x_train = np.hstack((np.ones((y_train.size, 1)), x_train))
    x_test = np.hstack((np.ones((y_test.size, 1)), x_test))

    rand = np.random.RandomState(19)
    w_init = rand.uniform(-1, 1, x_train.shape[1]) * .000001
    batch_size1 = 288
    batch_size2 = 100
    batch_size3 = 50
    w_history_stoch1, mse_history_stoch1 = batch_stochastic_gradient_descent(
        100, 0.1, w_init,
        mse, grad_mse, (x_train, y_train), batch_size=batch_size1,
        learning_rate=1e-6, momentum=0.7)
    w_history_stoch2, mse_history_stoch2 = batch_stochastic_gradient_descent(
        100, 0.1, w_init,
        mse, grad_mse, (x_train, y_train), batch_size=batch_size2,
        learning_rate=1e-6, momentum=0.7)
    w_history_stoch3, mse_history_stoch3 = batch_stochastic_gradient_descent(
        100, 0.1, w_init,
        mse, grad_mse, (x_train, y_train), batch_size=batch_size3,
        learning_rate=1e-6, momentum=0.7)

    w_history_NAG1, mse_history_NAG1 = NAG(
        100, 0.1, w_init,
        mse, grad_mse, (x_train, y_train), batch_size=batch_size1,
        learning_rate=1e-6, momentum=0.9)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    # Plot the MSE
    ax[0][0].plot(np.arange(mse_history_stoch1.size), mse_history_stoch1)
    ax[0][0].set_xlabel('Iteration No.')
    ax[0][0].set_ylabel('Mean Square Error')
    ax[0][0].set_title(f'Gradient Descent on Digits Data (BatchSGD with Batch size ={batch_size1})')

    ax[0][1].plot(np.arange(mse_history_stoch2.size), mse_history_stoch2)
    ax[0][1].set_xlabel('Iteration No.')
    ax[0][1].set_ylabel('Mean Square Error')
    ax[0][1].set_title(f'Gradient Descent on Digits Data (BatchSGD with Batch size ={batch_size2})')

    ax[0][2].plot(np.arange(mse_history_stoch3.size), mse_history_stoch3)
    ax[0][2].set_xlabel('Iteration No.')
    ax[0][2].set_ylabel('Mean Square Error')
    ax[0][2].set_title(f'Gradient Descent on Digits Data (BatchSGD with Batch size ={batch_size3})')

    ax[1][0].plot(np.arange(mse_history_NAG1.size), mse_history_NAG1)
    ax[1][0].set_xlabel('Iteration No.')
    ax[1][0].set_ylabel('Mean Square Error')
    ax[1][0].set_title(f'NAG on Digits Data (BatchSGD with Batch size ={batch_size1})')

    plt.show()
