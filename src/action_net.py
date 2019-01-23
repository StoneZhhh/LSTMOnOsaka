import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


torch.set_default_tensor_type('torch.DoubleTensor')
np.random.seed(42)


class ActionNet(nn.Module):

    def __init__(self, time_length, input_dim, hidden_dim, num_layers=1):
        super(ActionNet, self).__init__()
        self.time_length = time_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.class_num = 4

        # Define the LSTM layer
        self.lstm0 = nn.LSTM(self.input_dim // self.class_num, self.hidden_dim, self.num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(self.input_dim // self.class_num, self.hidden_dim, self.num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(self.input_dim // self.class_num, self.hidden_dim, self.num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(self.input_dim // self.class_num, self.hidden_dim, self.num_layers, batch_first=True)

        # define fully connected net
        self.fcn1 = nn.Linear(self.hidden_dim * self.class_num, self.class_num)

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.fcn1.weight)

    def forward(self, input):
        sensorsize = self.input_dim // self.class_num
        batch_size = input.size()[0]

        (h0, c0) = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

        # shape of lstm_out: (time_length, batch_size, hidden_dim)
        lstm_out0, hidden0 = self.lstm0(input[:, :, : sensorsize].view(batch_size, self.time_length, -1),  (h0, c0))
        lstm_out1, hidden1 = self.lstm1(input[:, :, sensorsize: 2*sensorsize].view(batch_size, self.time_length, -1), (h0, c0))
        lstm_out2, hidden2 = self.lstm2(input[:, :, 2*sensorsize: 3*sensorsize].view(batch_size, self.time_length, -1), (h0, c0))
        lstm_out3, hidden3 = self.lstm3(input[:, :, 3*sensorsize:].view(batch_size, self.time_length, -1), (h0, c0))

        lstm_out0 = lstm_out0[:, -1, :].view(batch_size, -1)
        lstm_out1 = lstm_out1[:, -1, :].view(batch_size, -1)
        lstm_out2 = lstm_out2[:, -1, :].view(batch_size, -1)
        lstm_out3 = lstm_out3[:, -1, :].view(batch_size, -1)

        # shape of lstm_cat: (batch_size, hidden_dim * 4)
        lstm_cat = torch.cat((lstm_out0, lstm_out1, lstm_out2, lstm_out3), 1).view(batch_size, -1)

        f1 = self.fcn1(lstm_cat)

        return f1


def read_np_data(filepath):
    return np.load(filepath)


def save_np_data(filename):
    np.save('../data/' + filename)


def fit(learning_rate, epoch, hidden_dim, train_model=True, load_model=False):
    xfilepath = '../data/OSAKA_x.npy'
    yfilepath = '../data/OSAKA_y.npy'
    checkpoint_path = '../data/action_net_4.p'

    # load training data
    xdata = read_np_data(xfilepath)
    ydata = read_np_data(yfilepath)
    batch_size, time_length, features_num = xdata.shape  # N, T, D

    # divide 80% of the data for train and 20% for test
    div = int(np.round(xdata.shape[0] * 0.8))
    train_idx = np.random.choice(list(range(batch_size)), div, replace=False)
    test_idx = [i for i in range(batch_size) if i not in list(train_idx)]

    x_train = xdata[train_idx]
    x_test = xdata[test_idx]
    y_train = ydata[train_idx]
    y_test = ydata[[i for i in range(batch_size) if i not in list(train_idx)]]

    # print(x_train.shape)
    # print(x_test.shape)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # init nn model, loss function and optimizer
    model = ActionNet(time_length, features_num, hidden_dim)
    if not load_model:
        model.init_weight()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    prev_epoch = 0

    # load model
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']

    # train EPOCH times
    if train_model:
        model.train()
        for i in range(epoch):
            # model.zero_grad()
            # model.hidden0, model.hidden1, model.hidden2, model.hidden3 = model.init_hidden()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, Variable(y_train).double())
            print('epoch', i, '; loss', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == epoch - 1:
                torch.save({
                    'epoch': i + prev_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, checkpoint_path)

    model.eval()

    # evaluate prediction accuracy
    with torch.no_grad():
        y_train_pred = model(x_train)
        correct = 0
        for i in range(y_train_pred.size()[0]):
            p = torch.argmax(y_train_pred[i])
            g = torch.argmax(y_train[i])
            if torch.eq(p, g):
                correct += 1
        print('train accuracy', correct * 1.0 / y_train_pred.size()[0])

        y_test_pred = model(x_test)
        correct = 0
        for i in range(y_test_pred.size()[0]):
            p = torch.argmax(y_test_pred[i])
            g = torch.argmax(y_test[i])
            if torch.eq(p, g):
                correct += 1
        print('test accuracy', correct * 1.0 / y_test_pred.size()[0])


if __name__ == '__main__':
    params = sys.argv[1:]
    learning_rate = params[0]
    epoch = params[1]
    hidden_dim = params[2]
    train_model = (params[3] == 'True')
    load_model = (params[4] == 'True')
    fit(float(learning_rate), int(epoch), int(hidden_dim), train_model=bool(train_model), load_model=bool(load_model))
