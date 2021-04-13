# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
#import  matplotlib.pyplot as plt

np.random.seed(10)
def prepro(d_path, test_path=None,length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28,SNR=None):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名


    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        # import locale
        # locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
        filenames = os.listdir(original_path)
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(original_path, i) #路径拼接
            file = loadmat(file_path) #读取第i个.mat文件
            file_keys = file.keys()
            for key in file_keys: #寻找名字包含DE的键
                if 'DE' in key:
                    files[i] = file[key].ravel()#files[i]中的i即为文件名，同时也是“键”名
        print('capture done')
        return files

    def add_noise(data,SNR):
        keys = data.keys()  # 对应files里的每一个文件名
        data1={}
        P_noise=np.zeros(len(keys))
        for step,i in enumerate (keys):
            signal=data[i]
            P_signal=((signal)**2).sum()/len(signal)  #功率 ，约等于方差
            P_noise[step]=P_signal/(10**(SNR/10))
            # noise=np.random.normal(0,P_noise**0.5,len(signal))#因此把功率作为方差，功率开根号就是标准差
            # data1[i]=signal+noise
            # print('signal.var()={:.4f}'.format(signal.var()))
            # print('noise.var()={:.4f}'.format(noise.var()))
        sigma=(P_noise.mean())**0.5
        for i in keys:
            signal=data[i]

            noise = np.random.normal(0, sigma, len(signal))
            data1[i] = signal + noise
        return data1

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()#对应files里的每一个文件名
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]#对应名字的数据
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:#如果增强数据
                enc_time = length // enc_step #次数=长度/间隔
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):

                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):

                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):

                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test,path):
        X = []
        Y = []
        label = 0
        filenames = os.listdir(path)
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        #Y=np.asarray(Y).reshape(-1,1)
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 加噪声
    if SNR != None:
        data1 = add_noise(data, SNR)
    else:
        data1 = data
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data1)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train, path=d_path)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test, path=d_path)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X1, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X1 = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    #Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)

    if test_path != None:
        data_2 = capture(original_path=test_path)
        if SNR != None:
            data2 = add_noise(data_2, SNR)
        else:
            data2 = data_2
        # 将数据切分为训练集、测试集
        train, test = slice_enc(data2)
        # 为训练集制作标签，返回X，Y
        Train_X_, Train_Y_ = add_labels(train, path=test_path)
        # 为测试集制作标签，返回X，Y
        Test_X_, Test_Y_ = add_labels(test, path=test_path)
        # 为训练集Y/测试集One-hot标签
        Train_Y_, Test_Y = one_hot(Train_Y_, Test_Y_)
        # 训练数据/测试数据 是否标准化.
        if normal:
            Train_X_, Test_X = scalar_stand(Train_X, Test_X_)
        else:
            # 需要做一个数据转换，转换成np格式.
            Train_X_ = np.asarray(Train_X_)
            Test_X = np.asarray(Test_X_)
        # 将测试集切分为验证集合和测试集.
        #Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X_, Test_Y_)

    return Train_X1, Train_Y, Test_X, Test_Y

def getdata(trainth,testh,length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28,SNR=None):

    x_train, y_train, x_test, y_test=prepro(d_path=trainth, test_path=testh,length=length,
                                            number=number, normal=normal, rate=rate, enc=enc,
                                            enc_step=enc_step,SNR=SNR)
    import torch
    import torch.utils.data as Data
    # 输入卷积的时候还需要修改一下，增加通道数目
    x_train, x_test = x_train[:, np.newaxis, :], x_test[:, np.newaxis, :]
    # x_train,  x_test = x_train[:,:,np.newaxis],  x_test[:,:,np.newaxis]
    # 输入数据的维,
    input_shape = x_train.shape[1:]

    print('训练样本维度:', x_train.shape)
    print(x_train.shape[0], '训练样本个数')
    print('测试样本的维度', x_test.shape)
    print(x_test.shape[0], '测试样本个数')
    print('SNR=',SNR)

    # 使用GPU计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_xt = torch.from_numpy(x_train.astype(np.float32)).cuda(device)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).cuda(device)
    train_yt = torch.topk(train_yt, 1)[1].squeeze(1)  # one_hot转回普通

    test_xt = torch.from_numpy(x_test.astype(np.float32)).cuda(device)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).cuda(device)
    test_yt = torch.topk(test_yt, 1)[1].squeeze(1)  # one_hot转回普通

    train_data = Data.TensorDataset(train_xt, train_yt)
    test_data = Data.TensorDataset(test_xt, test_yt)

    return train_data,test_data


if __name__ == "__main__":
    path = 'data/0HP'
    testh='data/1HP'
    # train_X, train_Y, test_X, test_Y = prepro(d_path=path,
    #                                                             test_path=testh,
    #                                                             length=864,
    #                                                             number=1000,
    #                                                             normal=False,
    #                                                             rate=[0.5, 0.25, 0.25],
    #                                                             enc=False,
    #                                                             enc_step=28,
    #                                                             SNR=-10)
    dataset_train, dataset_valid =getdata(trainth=path,testh=testh,length=2048,number=1000,
                                           normal=True,rate=[0.8, 0.1, 0.1],enc=True,
                                          enc_step=28, SNR=None)
