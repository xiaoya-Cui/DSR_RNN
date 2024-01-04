from Train import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def main():
    X_constants, X_rnn, y_constants, y_rnn = get_data()
    results = train(
        X_constants,y_constants,X_rnn,y_rnn,
        operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin','tan','exp','ln','c', 'var_x'],
        min_length = 2,
        max_length = 10,
        type = 'rnn',
        num_layers = 1,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 25,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 2000,
        scale_initial_risk = True,
        batch_size = 2000,
        num_batches = 500,
        use_gpu = False,
        live_print = True,
        summary_print = True
    )

    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]
    loss_total = results[4]


    plt.plot([i+1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.show()

    plt.plot([i+1 for i in range(len(loss_total))], loss_total)
    plt.xlabel('Epoch')
    plt.ylabel('loss_total')
    plt.title('loss over Time')
    plt.show()

def get_data():
    X = np.arange(-1,1,0.1)
    #y = 1.452*X + 0.012
    #y = np.cos(X)*np.sin(X*X)
    #y = 3.1415 * X * np.sin(2*X)
    y = X*X*X + X*X +X
    #y = 1.57 + 24.3*X
    #y = np.log(X*X+1.3)

    X = X.reshape(X.shape[0], 1)
    comb = list(zip(X,y))
    random.shuffle(comb)
    X,y = zip(*comb)

    training_proportion = 0.3
    div = int(training_proportion * len(X))

    X_constants,X_rnn = np.array(X[:div]),np.array(X[div:])
    y_constants,y_rnn = np.array(y[:div]),np.array(y[div:])
    X_constants,X_rnn = torch.Tensor(X_constants),torch.Tensor(X_rnn)
    y_constants,y_rnn = torch.Tensor(y_constants),torch.Tensor(y_rnn)

    return X_constants,X_rnn,y_constants,y_rnn

if __name__=='__main__':
    main()
