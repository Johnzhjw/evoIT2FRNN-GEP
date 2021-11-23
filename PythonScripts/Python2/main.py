import time
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
import os
import pandas as pd
from torchvision import transforms
import json
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 重写Dataset
class Mydataset(Dataset):

    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1.reshape(len(x1), -1)), y1
        return x1.reshape(len(x1), -1), y1

    def __len__(self):
        return len(self.x)


def get_dataset(file_name='../dataset/Stk.0941.HK.all.csv', train_ratio=2 / 3, col_name_tar="Close", flag_multi=False,
                flag_fuse=False, tag_norm='min-max', trn_batch_size=12, vld_batch_size=12, shuffle=True):
    df = pd.read_csv(file_name)
    # df = df.sort_index(ascending=True)
    print(df.head(5))

    train_len = int(len(df) * train_ratio)

    # 提取open,close,high,low,vol 作为feature,并做标准化
    # df = df[["Open", "Close", "High", "Low", "Volume", "Adjusted"]]
    df = df[[col for i, col in enumerate(df.columns) if i]]
    tar_min = df[col_name_tar][:train_len].min()
    tar_max = df[col_name_tar][:train_len].max()
    tar_mean = df[col_name_tar][:train_len].mean()
    tar_std = df[col_name_tar][:train_len].std()
    if tag_norm == 'min-max':
        df = df.apply(lambda x: (x - min(x[:train_len])) / (max(x[:train_len]) - min(x[:train_len])))
    elif tag_norm == 'Z-score':
        df = df.apply(lambda x: (x - x[:train_len].mean()) / x[:train_len].std())
    elif tag_norm == 'none':
        df = df.apply(lambda x: x)
    else:
        print('Invalid norm type.')
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    if flag_fuse:
        train_inds = [i for i, col in enumerate(df.columns) if col != col_name_tar]
    else:
        train_inds = [i for i, col in enumerate(df.columns)]
    if flag_multi:
        n_ft = len(train_inds)
    else:
        n_ft = 1
        train_inds = [i for i, col in enumerate(df.columns) if col == col_name_tar]
    test_inds = [i for i, col in enumerate(df.columns) if col == col_name_tar]
    total_len = df.shape[0]
    sequence = 3
    X = []
    Y = []
    for i in range(df.shape[0] - sequence):
        X.append(np.array(df.iloc[i:(i + sequence), train_inds], dtype=np.float32).reshape(sequence, -1))
        Y.append(np.array(df.iloc[(i + sequence), test_inds], dtype=np.float32).reshape(-1, ))

    print(X[0])
    print(Y[0])

    # # 构建batch
    trainx, trainy = X[:(train_len - sequence)], Y[:(train_len - sequence)]
    testx, testy = X[(train_len - sequence):], Y[(train_len - sequence):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()),
                              batch_size=trn_batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=vld_batch_size, shuffle=shuffle)

    return {'tar_min': tar_min, 'tar_max': tar_max, 'tar_mean': tar_mean, 'tar_std': tar_std, 'n_ft': n_ft,
            'total_len': total_len, 'train_len': train_len, 'sequence': sequence, 'tag_norm': tag_norm,
            'col_name_tar': col_name_tar, 'df': df, 'train_loader': train_loader, 'test_loader': test_loader}


def save_net_config(path, net, config_name='net.config'):
    """ dump run_config and net_config to the model_folder """
    net_save_path = os.path.join(path, config_name)
    json.dump(net.config, open(net_save_path, 'w'), indent=4)
    print('Network configs dump to %s' % net_save_path)


def save_net(path, net, model_name):
    """ dump net weight as checkpoint """
    if isinstance(net, torch.nn.DataParallel):
        checkpoint = {'state_dict': net.module.state_dict()}
    else:
        checkpoint = {'state_dict': net.state_dict()}
    model_path = os.path.join(path, model_name)
    torch.save(checkpoint, model_path)
    print('Network model dump to %s' % model_path)


class lstm(nn.Module):

    def __init__(self, input_size=5, hidden_size=10, output_size=1):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a, b, c = hidden.shape
        out = self.linear(hidden.reshape(a * b, c))
        return out


class CNNNetwork(nn.Module):
    def __init__(self, sequence=3, n_ft=1):
        super(CNNNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=(sequence, 1))
        self.relu = nn.ReLU(inplace=True)
        self.Linear1 = nn.Linear(64 * n_ft, 50)
        self.Linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.relu(x)
        x = x.flatten(1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x


class TSA_uni():
    def __init__(self, model_name, data, flag_print=False, sequence=3, n_ft=1):
        super(TSA_uni, self).__init__()
        self.model_name = model_name
        self.sequence = sequence
    def pred(self, data, n_known):
        preds = []
        labels = []
        for i, ind in enumerate(range(n_known, len(data))):
            if 'naive' in self.model_name:
                yhat = data[ind-1]
            elif 'avg' in self.model_name:
                yhat = np.mean(data[:n_known])
            elif 'mov_win' in self.model_name:
                yhat = np.mean(data[ind-self.sequence:ind])
            preds.append(yhat)
            labels.append(data.tolist()[ind])
        return preds, labels


class get_model():

    def __init__(self, args, name_model='lstm', lstm_hid_size=10, lr=0.001):
        self.args = args
        self.name_model = name_model
        self.criterion = nn.MSELoss()
        if name_model == 'lstm':
            self.model = lstm(input_size=args['n_ft'], hidden_size=lstm_hid_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif name_model == 'cnn':
            self.model = CNNNetwork(sequence=args['sequence'], n_ft=args['n_ft'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif 'TSA_uni' in name_model:
            self.TSA_agent = TSA_uni(model_name=name_model,
                                     data=args['df'][args['col_name_tar']][:args['train_len']],
                                     flag_print=False, sequence=args['sequence'], n_ft=args['n_ft'])
        else:
            print('Invalid network type')
        if 'TSA_uni' not in name_model and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

    def train(self, i_run, n_epochs=200, flag_info=False):
        if 'TSA_uni' in self.name_model:
            return
        if flag_info:
            nBatch = len(self.args['train_loader'])
            for i in range(n_epochs):
                if self.name_model == 'cnn':
                    self.model.train()
                losses = AverageMeter()
                data_time = AverageMeter()
                total_loss = 0
                with tqdm(total=nBatch,
                          desc='Run #{} Train Epoch #{}'.format(i_run, i + 1)) as t:
                    end = time.time()
                    for idx, (data, label) in enumerate(self.args['train_loader']):
                        data_time.update(time.time() - end)
                        data, label = data.to(self.device), label.to(self.device)
                        data1 = data.squeeze(1)
                        if self.name_model == 'cnn':
                            pred = self.model(data1)
                        else:
                            pred = self.model(Variable(data1))
                        loss = self.criterion(pred, label)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()

                        # measure accuracy and record loss
                        losses.update(loss.item(), data.size(0))

                        t.set_postfix({
                            'loss': losses.avg,
                            # 'lr': new_lr,
                            # 'loss_type': loss_type,
                            'data_time': data_time.avg,
                        })
                        t.update(10)
                        end = time.time()
        else:
            for i in range(n_epochs):
                if self.name_model == 'cnn':
                    self.model.train()
                for idx, (data, label) in enumerate(self.args['train_loader']):
                    data, label = data.to(self.device), label.to(self.device)
                    data1 = data.squeeze(1)
                    if self.name_model == 'cnn':
                        pred = self.model(data1)
                    else:
                        pred = self.model(Variable(data1))
                    loss = self.criterion(pred, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def test(self):
        # 开始测试
        if 'TSA_uni' not in self.name_model:
            if self.name_model == 'cnn':
                self.model.eval()
            preds = []
            labels = []
            for idx, (x, label) in enumerate(self.args['test_loader']):
                x, label = x.to(self.device), label.to(self.device)
                x = x.squeeze(1)  # batch_size,seq_len,input_size
                pred = self.model(x)
                preds.extend(pred.data.squeeze(1).tolist())
                labels.extend(label.squeeze(1).tolist())
        else:
            preds, labels = self.TSA_agent.pred(self.args['df'][self.args['col_name_tar']], self.args['train_len'])

        # print(len(preds[0:50]))
        # print(len(labels[0:50]))

        if self.args['tag_norm'] == 'min-max':
            res_final = np.sqrt(self.criterion(torch.Tensor(preds), torch.Tensor(labels))) * \
                        (self.args['tar_max'] - self.args['tar_min'])
            print(res_final)
            plt.plot(
                [ele * (self.args['tar_max'] - self.args['tar_min']) + self.args['tar_min'] for ele in preds[0:50]],
                "r", label="pred")
            plt.plot(
                [ele * (self.args['tar_max'] - self.args['tar_min']) + self.args['tar_min'] for ele in labels[0:50]],
                "b", label="real")
        elif self.args['tag_norm'] == 'Z-score':
            res_final = np.sqrt(self.criterion(torch.Tensor(preds), torch.Tensor(labels))) * self.args['tar_std']
            print(res_final)
            plt.plot([ele * self.args['tar_std'] + self.args['tar_mean'] for ele in preds[0:50]], "r", label="pred")
            plt.plot([ele * self.args['tar_std'] + self.args['tar_mean'] for ele in labels[0:50]], "b", label="real")
        elif self.args['tag_norm'] == 'none':
            res_final = np.sqrt(self.criterion(torch.Tensor(preds), torch.Tensor(labels)))
            print(res_final)
            plt.plot([ele for ele in preds[0:50]], "r", label="pred")
            plt.plot([ele for ele in labels[0:50]], "b", label="real")
        else:
            print('Invalid norm type.')
        plt.show()

        return res_final


def main(args):
    all_res = []
    os.makedirs(args.save, exist_ok=True)
    for i in range(args.iterations):
        data_args = get_dataset(file_name=args.file_name, col_name_tar=args.col_name_tar,
                                flag_multi=args.flag_multi, flag_fuse=args.flag_fuse,
                                tag_norm=args.tag_norm, train_ratio=args.train_ratio,
                                trn_batch_size=args.trn_batch_size, vld_batch_size=args.vld_batch_size,
                                shuffle=args.shuffle)
        print('Iter: ', i + 1)
        engine = get_model(data_args, name_model=args.name_model, lstm_hid_size=args.lstm_hid_size, lr=args.lr)
        engine.train(i_run=i, n_epochs=args.n_epochs)
        res_final = engine.test()
        all_res.append(res_final.tolist())
        if 'TSA_uni' not in engine.name_model:
            save_net(args.save, engine.model, engine.name_model + '-{}'.format(i + 1) + '.pkl')
        # 保存图片到本地
        plt.savefig(os.path.join(args.save, engine.name_model + '-{}'.format(i + 1) + '_plot.eps'),
                    format='eps', bbox_inches='tight')
    save_path = os.path.join(args.save, args.name_model)
    del data_args['train_loader']
    del data_args['test_loader']
    del data_args['df']
    data_args['test_err'] = all_res
    data_args['file_name'] = args.file_name
    with open(save_path, 'w') as handle:
        json.dump(data_args, handle)
    print(all_res)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='../dataset/_TS/traffic.csv', help='location of data')
    parser.add_argument('--col_name_tar', type=str, default='Slowness_in_traffic', help='target feature name')
    parser.add_argument('--flag_multi', action='store_true', default=False, help='whether use multiple input features')
    parser.add_argument('--flag_fuse', action='store_true', default=False, help='whether fuse input features')
    parser.add_argument('--shuffle', action='store_true', default=False, help='whether fuse input features')
    parser.add_argument('--tag_norm', type=str, default='min-max', help='Normalization type: none/min-max/Z-score')
    parser.add_argument('--train_ratio', type=float, default=2/3, help='The ratio of train samples')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--name_model', type=str, default='lstm', help='model name - lstm/cnn/TSA_uni_naive/TSA_uni_avg/TSA_uni_mov_win')
    parser.add_argument('--save', type=str, default='pred', help='location of dir to save')
    parser.add_argument('--str_time', type=str, default=None, help='time string')
    parser.add_argument('--iterations', type=int, default=10, help='number of search iterations')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for dataloader per evaluation job')
    parser.add_argument('--trn_batch_size', type=int, default=12, help='train batch size for training')
    parser.add_argument('--vld_batch_size', type=int, default=12, help='test batch size for inference')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs for CNN training')
    parser.add_argument('--lstm_hid_size', type=int, default=3, help='hidden size of lstm')
    cfgs = parser.parse_args()
    cfgs.str_time = time.strftime("%Y%m%d-%H%M%S")
    cfgs.save = '{}-{}-{}'.format(cfgs.save, cfgs.name_model, cfgs.str_time)
    print(cfgs)
    main(cfgs)
