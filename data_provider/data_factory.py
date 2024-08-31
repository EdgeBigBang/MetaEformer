import os
import numpy as np
from data_provider.data_loader import Dataset_ECL, Dataset_Traffic, Dataset_Cloud, Dataset_ECW, PPIO_Dataset_test
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Pred
from sklearn import preprocessing

data_dict = {
    'ECL': Dataset_ECL,
    'Traffic': Dataset_Traffic,
    'Cloud': Dataset_Cloud
}


def data_provider(args, flag):
    data_class = args.dataset
    Data = data_dict[data_class]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        data_path = args.data
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
        data_path = args.data
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        data_path = args.data

    data_set = Data(
        args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.enc_len, args.label_len, args.pred_len],
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def get_ppio(args):
    file_path = os.path.join(args.root_path, args.data)
    print(file_path)
    # with open(file_path, 'rb') as f:
    #     content = f.read()
    #     print(len(content))  # 应该不为 0
    X = np.load(open(file_path, 'rb'), allow_pickle=True)
    # y 只需要实际的流量值即可
    y = X[:, :, 0]
    Xtr, ytr, Xte, yte = train_test_split(X, y)
    num_ts, num_periods, num_features = Xte.shape

    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    # TODO yscaler 为什么用的Xtr的数据？
    yscaler.fit(ytr.reshape(-1, 1))

    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    # pickle.dump([xscaler, yscaler], open('8_scalers.pkl', 'wb'))
    # 这边载入的loader
    Xtr_loader = DataLoader(Dataset_ECW(Xtr, args.enc_len, args.label_len, args.pred_len), batch_size=args.batch_size)
    Xte_loader = DataLoader(Dataset_ECW(Xte, args.enc_len, args.label_len, args.pred_len), batch_size=args.batch_size)

    return Xtr_loader, Xte_loader, yscaler

def train_test_split(X, y, train_ratio=0.8, test_ratio=0.2):
    num_ts, num_periods, num_features = X.shape
    # train_periods = int(num_periods * train_ratio)
    train_periods = int(num_periods * train_ratio)
    test_periods = int(num_periods * test_ratio)

    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, -test_periods:, :]
    yte = y[:, -test_periods:]
    return Xtr, ytr, Xte, yte

def train_test_split_test(X, y, train_ratio=1, test_ratio=1):
    num_ts, num_periods, num_features = X.shape
    # train_periods = int(num_periods * train_ratio)
    train_periods = int(num_periods * train_ratio)
    test_periods = int(num_periods * test_ratio)

    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, -test_periods:, :]
    yte = y[:, -test_periods:]
    return Xtr, ytr, Xte, yte

def get_ppio_test(raw_X, raw_y, X, y, batch_size, args):
    Xtr, ytr, Xte, yte = train_test_split_test(X, y)
    num_ts, num_periods, num_features = Xte.shape

    result = np.concatenate((raw_y.reshape(-1,1), yte.reshape(-1,1)), axis=0)
    combined_data = np.concatenate((raw_X.reshape(-1,1), Xte.reshape(-1,1)), axis=0)
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    # TODO yscaler 为什么用的Xtr的数据？
    yscaler.fit(result.reshape(-1, 1))
    xscaler.fit(combined_data.reshape(-1, num_features))
    Xtr = xscaler.transform(Xtr.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    # pickle.dump([xscaler, yscaler], open('8_scalers.pkl', 'wb'))
    # 这边载入的loader
    Xtr_loader = DataLoader(PPIO_Dataset_test(Xtr, args.enc_len, args.label_len, args.pred_len), batch_size=batch_size)
    Xte_loader = DataLoader(PPIO_Dataset_test(Xte, args.enc_len, args.label_len, args.pred_len), batch_size=batch_size)

    return Xtr_loader, Xte_loader, yscaler