import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pickle


torch.set_default_tensor_type('torch.DoubleTensor')

'''
When describing the shape of all the tensors, B is short of batch_size, T is short for max sequence length
and H is short for hidden_dim / hidden_size.
'''


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.attn.weight)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions, B, H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B, T, H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask)  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, class_num=4, layer_num=1):
        super(ALSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_num = class_num

        self.layer_num = layer_num

        self.bilstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True, bidirectional=True)

        self.attn = Attention('concat', hidden_size)
        self.linear = nn.Linear(hidden_size, self.class_num)

    def init_weight(self):
        self.attn.init_weight()
        torch.nn.init.xavier_uniform(self.linear.weight)

    def forward(self, input_seqs, input_lens, hidden=None):
        '''
        :param input_seqs:
            Has shape (batch_size, max_seq_length, input_dim) where input_dim = 21
        :param input_lens:
            Has shape (batch_size, 1) recording the actual length of every input seq
        :param hidden:
            initial state of lstm
        :returns:
            outputs in shape (batch_size, self.class_num)
        '''
        batch_size = input_seqs.size()[0]
        x = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens, batch_first=True)
        (h0, c0) = (Variable(torch.zeros(2 * self.layer_num, batch_size, self.hidden_size)),
                    Variable(torch.zeros(2 * self.layer_num, batch_size, self.hidden_size)))

        outputs, (h, c) = self.bilstm(x, (h0, c0))
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print(outputs.size())
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # (batch_size, max_seq_len, hidden_dim)
        # print('original hidden size: ', h.size())
        hidden = h.view(1, 2, batch_size, -1)[:, 1, :, :]  # (layer_num, batch_size, hidden_dim)
        # print('original hidden size: ', hidden.size())
        attn_weights = self.attn(hidden, outputs, src_len=input_lens)

        attn_out = attn_weights.bmm(outputs).squeeze(1)  # (B, 1, T) dot (B, T, H) -> (B, 1, H) -> (B, H)

        linear_out = self.linear(attn_out)

        return linear_out


def preprocessing(data):
    '''
    :param data:
        A list of [[[21 dim], [21 dim], ..., [21 dim]], label]
    :param seq_lens:
        A sorted list of int in a reversed order, recording the sequence lengths of the elements in data
    :returns:
        sorted_x:
        sorted_y: labels
        sorted_lens
    '''
    trans_dict = {
        'Walk1.csv': [1, 0, 0, 0],
        'Walk2.csv': [0, 1, 0, 0],
        'SlopeUp.csv': [0, 0, 1, 0],
        'SlopeDown.csv': [0, 0, 0, 1]
    }

    seq_lens = []
    x = []
    y = []
    for d in data:
        seq_lens.append(len(d[0]))
        x.append(d[0])
        y.append(trans_dict[d[1]])

    sorted_idx = list(np.argsort(seq_lens))
    sorted_idx.reverse()

    sorted_lens = Variable(torch.tensor([seq_lens[i] for i in sorted_idx]))

    sorted_x = [Variable(torch.tensor(x[i])) for i in sorted_idx]
    sorted_x = torch.nn.utils.rnn.pad_sequence(sorted_x, batch_first=True)

    sorted_y = Variable(torch.tensor([y[i] for i in sorted_idx]))

    return sorted_x, sorted_y, sorted_lens


def fit(filepath, checkpoint_path, learning_rate, epoch, hidden_dim, load_model=False, train_model=True):
    data = pickle.load(open(filepath, 'rb'))
    train = [data['data'][i] for i in data['train_ids']]
    test = [data['data'][i] for i in data['test_ids']]
    train_x, train_y, train_x_len = preprocessing(train)
    test_x, test_y, test_x_len = preprocessing(test)

    model = ALSTM(21, hidden_dim)

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

    # print(train_x)
    # train EPOCH times
    if train_model:
        model.train()
        for i in range(epoch):
            # model.zero_grad()
            # model.hidden0, model.hidden1, model.hidden2, model.hidden3 = model.init_hidden()
            y_pred = model(train_x, train_x_len)
            loss = loss_fn(y_pred, train_y.double())
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
        y_train_pred = model(train_x, train_x_len)
        correct = 0
        for i in range(y_train_pred.size()[0]):
            p = torch.argmax(y_train_pred[i])
            g = torch.argmax(train_y[i])
            if torch.eq(p, g):
                correct += 1
        print('train accuracy', correct * 1.0 / y_train_pred.size()[0])

        y_test_pred = model(test_x, test_x_len)
        correct = 0
        for i in range(y_test_pred.size()[0]):
            p = torch.argmax(y_test_pred[i])
            g = torch.argmax(test_y[i])
            if torch.eq(p, g):
                correct += 1
        print('test accuracy', correct * 1.0 / y_test_pred.size()[0])


if __name__ == '__main__':
    params = sys.argv[1:]
    if not params:
        learning_rate = 1e-2
        epoch = 100
        hidden_dim = 10
        train_model = True
        load_model = False
    else:
        learning_rate = params[0]
        epoch = params[1]
        hidden_dim = params[2]
        train_model = (params[3] == 'True')
        load_model = (params[4] == 'True')

    filepath = '../data/final_osaka_data.pkl'
    checkpoint_path = '../data/bi_net.p'
    fit(filepath, checkpoint_path, float(learning_rate), int(epoch), int(hidden_dim),
        train_model=bool(train_model), load_model=bool(load_model))
    # data = pickle.load(open('../data/final_osaka_data.pkl', 'rb'))
    # test = [data['data'][i] for i in data['train_ids']]
    # for d in test:
    #     print(d[1])
    # print(len(test[0][0]))  # time length
    # print(len(test[0][0][0]))  # input dim
    # print(len(test[0][0][1]))
    # print(len(test[1][0]))
    # print(len(test[1][0][0]))
    # print(len(test[2][0]))
    # print(preprocessing(test)[0])
