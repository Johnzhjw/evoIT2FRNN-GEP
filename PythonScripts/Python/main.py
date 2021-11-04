import tushare as ts
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
import os
import pandas as pd
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchsummary import summary

train_ratio = 2 / 3

# 获取代号为000300的股票价格
# cons=ts.get_apis()
# df=ts.bar('000001', conn=cons, asset='INDEX', start_date='2018-01-01', end_date='')
# df = df.sort_index(ascending=True)

df = pd.read_csv('../dataset/Stk.0941.HK.all.csv')
df = df.sort_index(ascending=True)
print(df.head(5))

train_len = int(len(df)*train_ratio)

# 提取open,close,high,low,vol 作为feature,并做标准化

df = df[["Open", "Close", "High", "Low", "Volume", "Adjusted"]]
close_min = df['Close'][:train_len].min()
close_max = df["Close"][:train_len].max()
close_mean = df["Close"][:train_len].mean()
close_std = df["Close"][:train_len].std()
df = df.apply(lambda x: (x - min(x[:train_len])) / (max(x[:train_len]) - min(x[:train_len])))
# df = df.apply(lambda x: (x - x[:train_len].mean()) / x[:train_len].std())

# 定义X和Y: 根据前n天的数据，预测当天的收盘价
# 例如根据 1月1日，1月2日，1月3日 的价格预测 1月4日的收盘价
# X=[ ["open1","close1","high1","low1","vol1"] ,["open2","close2","high2","low2","vol2"] ,["open3","close3","high3","low3","vol3"] ]
# Y=[ close4 ]
# 那么X对应的sequence=3 ， [ input_size=5(5维度) ，这tm就是nlp中每个词的embedding ]

n_ft = 1 # df.shape[1]
total_len = df.shape[0]
sequence = 3
X = []
Y = []
for i in range(df.shape[0] - sequence):
    X.append(np.array(df.iloc[i:(i + sequence), 1], dtype=np.float32))
    Y.append(np.array(df.iloc[(i + sequence), 1], dtype=np.float32))

print(X[0])
print(Y[0])


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
            return self.tranform(x1.reshape(len(x1),-1)), y1
        return x1.reshape(len(x1),-1), y1

    def __len__(self):
        return len(self.x)


# # 构建batch
trainx, trainy = X[:train_len], Y[:train_len]
testx, testy = X[train_len:], Y[train_len:]
train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=12,
                          shuffle=True)
test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=12, shuffle=True)


class lstm(nn.Module):

    def __init__(self, input_size=5, hidden_size=3, output_size=1):
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


model = lstm(input_size=n_ft)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

preds = []
labels = []
for i in range(200):
    total_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        data1 = data.squeeze(1)
        pred = model(Variable(data1))
        label = label.unsqueeze(1)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# 开始测试


preds = []
labels = []
for idx, (x, label) in enumerate(test_loader):
    x = x.squeeze(1)  # batch_size,seq_len,input_size
    pred = model(x)
    preds.extend(pred.data.squeeze(1).tolist())
    labels.extend(label.tolist())

# print(len(preds[0:50]))
# print(len(labels[0:50]))

print(np.sqrt(criterion(torch.Tensor(preds), torch.Tensor(labels))))

import matplotlib.pyplot as plt

plt.plot([ele * (close_max - close_min) + close_min for ele in preds[0:50]], "r", label="pred")
plt.plot([ele * (close_max - close_min) + close_min for ele in labels[0:50]], "b", label="real")
# plt.plot([ele * close_std + close_mean for ele in preds[0:50]], "r", label="pred")
# plt.plot([ele * close_std + close_mean for ele in labels[0:50]], "b", label="real")
plt.show()

