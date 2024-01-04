import time
import random
import torch
import torch.nn as nn
import numpy as np
from Operators import Operators
from Rnn import DSRRNN
from Expression_utils import *


###############################################################################
# Main Training loop
###############################################################################

def train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin','tan','exp','ln','c', 'var_x'],
        #operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin', 'c', 'var_x'],
        min_length = 2,
        max_length = 10,
        type = 'rnn',
        num_layers = 1,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 15,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 2000,
        scale_initial_risk = True,
        batch_size = 100,
        num_batches = 500,
        hidden_size = 500,
        use_gpu = False,
        live_print = True,
        summary_print = True
    ):


    epoch_best_rewards = []
    epoch_best_expressions = []
    loss_total = []

    # Establish GPU device if necessary
    if (use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Initialize operators, RNN, and optimizer
    operators = Operators(operator_list, device)
    dsr_rnn = DSRRNN(operators, hidden_size, device, min_length=min_length,
                     max_length=max_length, type=type, num_layers=num_layers,dropout=dropout
                     ).to(device)
    if (optimizer == 'adam'):
        optim = torch.optim.Adam(dsr_rnn.parameters(),lr=lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)

    Best_expression,best_performance = None, float('-inf')

    start = time.time()

    sequences,sequence_lengths,log_probabilities,entropies = dsr_rnn.sample_sequence(initial_batch_size,min_length,
        max_length)

    for i in range(num_batches):
        expressions = []
        for j in range(len(sequences)):
        #Replace expression with an expression
            expressions.append(
                Expression(operators,sequences[j].long().tolist(),sequence_lengths[j].long().tolist()).to(device)
            )
        # constant optimization
        optimize_constants(expressions,X_constants,y_constants,inner_lr,inner_num_epochs,inner_optimizer)


        rewards = []
        for expression in expressions:
            rewards.append(benchmark(expression,X_rnn,y_rnn))
        rewards = torch.tensor(rewards)

        best_epoch_expression = expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())

        if (max(rewards) > best_performance):
        #训练阶段最好的
            best_performance = max(rewards)
            best_expression = best_epoch_expression

        #Early stopping criteria
        if (best_performance >= 0.98):
            best_str = str(best_expression)
            if(live_print):
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

        #计算 risk threshold
        if (i == 0 and scale_initial_risk):
            threshold = np.quantile(rewards,1-(1-risk_factor)/(initial_batch_size/batch_size))
        else:
            threshold = np.quantile(rewards,risk_factor)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j]>threshold])
        #挑选出要保留表达式的序号

        if (len(indices_to_keep) == 0 and summary_print):
        #如果没有满足条件的表达式，中止迭代
            print('Threshold removes all expressions.')
            break
        #筛选符合条件的，沿着维度dim选择序号indices_to_keep
        rewards = torch.index_select(rewards,0,indices_to_keep)
        log_probabilities = torch.index_select(log_probabilities,0,indices_to_keep)
        entropies = torch.index_select(entropies,0,indices_to_keep)

        #计算风险寻求策略梯度与熵梯度
        risk_seeking_grad = torch.sum((rewards-threshold) * log_probabilities,axis=0)
        entropy_grad = torch.sum(entropies,axis=0)
        #剪切梯度，防止爆炸
        risk_seeking_grad = torch.clip(risk_seeking_grad/len(rewards),-1e6,1e6)
        entropy_grad = entropy_coefficient * torch.clip(entropy_grad/len(rewards),-1e6,1e6)

        #计算loss
        loss = -1 * lr * (risk_seeking_grad + entropy_grad)
        loss_total.append(loss.item())
        loss.backward()
        optim.step()

        #每个epoch打印
        if (live_print):
            print(f"""Epoch: {i+1} ({round(float(time.time() - start), 2)}s elapsed)
            Entropy Loss: {entropy_grad.item()}
            Risk-Seeking Loss: {risk_seeking_grad.item()}
            Total Loss: {loss.item()}
            Best Performance (Overall): {best_performance}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Overall): {best_expression}
            Best Expression (Epoch): {best_epoch_expression}
            """)


        #采样下一个batch
        sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_sequence(batch_size,min_length,max_length)

    if (summary_print):
        print(f"""y1
        Time Elapsed: {round(float(time.time()-start),2)}
        Epochs Required: {i+1}
        Best Performance: {round(best_performance.item(),4)}
        Best Expression: {best_expression}
        """)
    return [epoch_best_rewards,epoch_best_expressions,best_performance,best_expression,loss_total]


def benchmark(expression,X_rnn,y_rnn):

    with torch.no_grad():
        y_pred = expression(X_rnn)
        return reward_nrmse(y_pred,y_rnn)

def reward_nrmse(y_pred,y_rnn):
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    val = torch.std(y_rnn) * val # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()









