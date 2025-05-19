import numpy as np
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
# if momentum is 0, 1 < batch_size < rows: Mini-Batch SGD(without momentum),
# if momentum is 0, batch_size = 1: SGD
def batch_stochastic_gradient_descent(max_epochs, threshold, w_init,
                                      obj_func, grad_func, xy, batch_size,
                                      learning_rate, momentum):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]

    # if batch size == 10, data == 1000, learn with 10 datas and update with mean of that mse; For 1 epoch, 100 batches come out with 10 data.
    # Run epochs
    while i < max_epochs and diff > threshold:
        # Shuffle the datas
        permutation = np.random.permutation(rows)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        # calculate the number of batches, number of iterations
        batch_num = (rows + batch_size - 1) // batch_size
        for k in range(batch_num):
            start = k * batch_size
            end = min(start + batch_size, rows)

            x_batch = x_train_shuffled[start:end, :]
            y_batch = y_train_shuffled[start:end]

            # grad_mse returns sum of mse
            mse_sum = grad_func(w, (x_batch, y_batch))
            mse_avg = mse_sum / x_batch.shape[0]

            delta_w = -learning_rate * mse_avg + momentum * delta_w
            w = w + delta_w

        i += 1

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history

# NAG optimizer; implemented from 'batch_stochastic_gradient_descent' function.
def NAG(max_epochs, threshold, w_init, obj_func, grad_func, xy, batch_size, learning_rate, momentum):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]

    # if batch size == 10, data == 1000, learn with 10 datas and update with mean of that mse; For 1 epoch, 100 batches come out with 10 data.
    # Run epochs
    while i < max_epochs and diff > threshold:
        # Shuffle the datas
        permutation = np.random.permutation(rows)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        # calculate the number of batches; number of iterations
        batch_num = (rows + batch_size - 1) // batch_size
        for k in range(batch_num):
            start = k * batch_size
            end = min(start + batch_size, rows)

            x_batch = x_train_shuffled[start:end, :]
            y_batch = y_train_shuffled[start:end]

            # pred = w_t + mu * m_{t-1}
            pred = w + momentum * delta_w

            # grad_mse returns sum of mse
            mse_sum = grad_func(pred, (x_batch, y_batch))
            mse_avg = mse_sum / x_batch.shape[0]

            delta_w = -learning_rate * mse_avg + momentum * delta_w  # m_t
            w = w + delta_w  # w_{t+1}
        i += 1

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


# Returns error rate of classifier
# total misclassifications/total*100
def error(w, xy):
    (x, y) = xy

    o = np.sum(x * w, axis=1)

    # map the output values to 0/1 class labels
    ind_1 = np.where(o > 0.5)
    ind_0 = np.where(o <= 0.5)
    o[ind_1] = 1
    o[ind_0] = 0
    return np.sum((o - y) * (o - y)) / y.size * 100


if __name__ == "__main__":
    # Load the digit dataset with two classes
    digits, target = dt.load_digits(n_class=2, return_X_y=True)

    # Split into train and a test set
    x_train, x_test, y_train, y_test = train_test_split(
        digits, target, test_size=0.2, random_state=10)

    # Add a column of ones to account for bias in train and test
    x_train = np.hstack((np.ones((y_train.size, 1)), x_train))
    x_test = np.hstack((np.ones((y_test.size, 1)), x_test))

    rand = np.random.RandomState(19)
    w_init = rand.uniform(-1, 1, x_train.shape[1]) * .000001

    # calculate the number of batches in mini-batch SGD without momentum
    batch_size = [288, 100, 50, 10, 5, 1]
    w_history_stoch = []
    mse_history_stoch = []

    for i, size in enumerate(batch_size):
        w_hist, mse_hist = batch_stochastic_gradient_descent(
            100, 0.1, w_init,
            mse, grad_mse, (x_train, y_train), batch_size=size,
            learning_rate=1e-6, momentum=0)
        w_history_stoch.append(w_hist)
        mse_history_stoch.append(mse_hist)

    # calculate with different momentums in NAG, Mini-batch SGD with momentum, with batchsize = 1
    w_history_NAG = []
    mse_history_NAG = []
    w_history_batch_momentum = []
    mse_history_batch_momentum = []
    momentum = np.arange(0.0, 1.1, 0.1)

    for i, m in enumerate(momentum):
        w_hist, mse_hist = NAG(
            100, 0.1, w_init,
            mse, grad_mse, (x_train, y_train), batch_size=batch_size[5],
            learning_rate=1e-6, momentum=m)
        w_history_NAG.append(w_hist)
        mse_history_NAG.append(mse_hist)

    for i, m in enumerate(momentum):
        w_hist, mse_hist = batch_stochastic_gradient_descent(
            100, 0.1, w_init,
            mse, grad_mse, (x_train, y_train), batch_size=batch_size[5],
            learning_rate=1e-6, momentum=m)
        w_history_batch_momentum.append(w_hist)
        mse_history_batch_momentum.append(mse_hist)

    # Plot the MSE with different batch sizes at Mini-batch SGD without momentum
    idx = 0
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    for i in range(3):
        for j in range(2):
            ax[i][j].plot(np.arange(mse_history_stoch[idx].size), mse_history_stoch[idx])
            ax[i][j].set_xlabel('Iteration No.')
            ax[i][j].set_ylabel('Mean Square Error')
            ax[i][j].set_title(f'BatchSGD with Batch size ={batch_size[idx]}')
            idx += 1
    plt.tight_layout()
    plt.show()

    # Plot the MSE with different batch sizes at Mini-batch SGD without momentum, batchsize = 1
    idx = 0
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 25))
    for i in range(3):
        for j in range(4):
            if idx < 11:
                ax[i][j].plot(np.arange(mse_history_NAG[idx].size), mse_history_NAG[idx], label='NAG')
                ax[i][j].plot(np.arange(mse_history_batch_momentum[idx].size), mse_history_batch_momentum[idx],
                              label='SGD')
                ax[i][j].legend(loc='upper right')

                ax[i][j].set_xlabel('Iteration No.')
                ax[i][j].set_ylabel('Mean Square Error')
                ax[i][j].set_title(f'NAG and SGD with m ={momentum[idx]:.1f} (bs=1)')

                train_error_NAG = error(w_history_NAG[idx][-1], (x_train, y_train))
                test_error_NAG = error(w_history_NAG[idx][-1], (x_test, y_test))

                train_error_stochastic = error(w_history_batch_momentum[idx][-1], (x_train, y_train))
                test_error_stochastic = error(w_history_batch_momentum[idx][-1], (x_test, y_test))

                print(f'Momentum = {momentum[idx]:.1f}')
                print(f'\tNAG:')
                print(f'\t\tTrain error: {train_error_NAG:.2f}')
                print(f'\t\tTest error: {test_error_NAG:.2f}')
                print(f'\tStochastic:')
                print(f'\t\tTrain error: {train_error_stochastic:.2f}')
                print(f'\t\tTest error: {test_error_stochastic:.2f}')
                idx += 1
            else:
                ax[i][j].axis('off')

    plt.tight_layout()
    plt.show()

    # with the plots above, we can see the convergence result with SGD with momentum and NAG.
    # Final errors of SGD and NAG are 0.35, 1.39, but m = 0.9 reach that value fast.
