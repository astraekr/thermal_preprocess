import datetime
import pandas as pd
import sklearn.preprocessing as skpp
import matplotlib.pyplot as plt
import re
from pandas import concat


non_decimal = re.compile(r'[^\d.]+')


def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Credit to Jason Brownlee at Machine Learning Mastery

    :param data:
    :param n_in:
    :param n_out:
    :param dropnan:
    :return:
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def removeText(textDataset):
    #Takes pandas dataframe as an argument
    #Remove all text from the dataset
    #returns a matrix of the values - not a dataframe
    numDataset = textDataset.values
    for col in range(0, len(textDataset.columns) - 1):
        # inn[index,col] = ([re.sub(non_decimal, "", elem) for index, elem in enumerate(inn[:,col])])
        # https://stackoverflow.com/questions/12116717/does-pythons-re-sub-take-an-array-as-the-input-parameter
        for index, elem in enumerate(numDataset[:, col]):
            if type(elem) == float: continue
            if type(elem) == int: continue
            numDataset[index, col] = re.sub(non_decimal, "", elem)

    return numDataset

def normalise_and_plot(csv_file):
    #columns
    #datetime,dn, dm, dx, sn, sm, sz, Ta, Ua, Pa, fan, current, voltage
    # 0, 1, 2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 10, 11 ,12
    dataset = pd.read_csv(csv_file, parse_dates=True, header=0, index_col=0, date_parser=parser)
    dataset.index.name = 'datetime'
    dataset.columns = ['dn', 'dm','dx', 'sn', 'sm', 'sx', 'ta', 'ua', 'pa', 'fan', 'currnet', 'voltage']

    dataset.drop(['dn'], axis=1, inplace=True)
    dataset.drop(['dm'], axis=1, inplace=True)
    dataset.drop(['dx'], axis=1, inplace=True)
    dataset.drop(['sn'], axis=1, inplace=True)
    dataset.drop(['sm'], axis=1, inplace=True)
    dataset.drop(['sx'], axis=1, inplace=True)
    dataset.drop(['fan'], axis=1, inplace=True)
    dataset.drop(['voltage'], axis=1, inplace=True)
    #dataset.drop(['dn', 'dm', 'sn', 'sm', 'sx', 'fan', 'voltage'], axis=1)

    data_points_numerical = removeText(dataset)
    values = data_points_numerical.astype('float32')
    plt.plot(values[:, 0], label='ta')
    plt.plot(values[:, 1] + 2, label='ua')
    plt.plot(values[:, 2] + 4, label='pa')
    plt.plot(values[:, 3] + 5, label='current')
    plt.legend()
    plt.savefig('/home/alastair/Data/3d/overnight2/analysis/datatimeseries.png')

    return
    numLags = 1
    #data_points = dataset.values
    data_points_numerical = removeText(dataset)
    values = data_points_numerical.astype('float32')
    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    plt.plot(scaled[:, 0], label='ta')
    plt.plot(scaled[:, 1] + 2, label='ua')
    plt.plot(scaled[:, 2] + 4, label='pa')
    plt.plot(scaled[:, 3] + 5, label='current')


    plt.legend()
    # pyplot.savefig('predict'+str(epoch)+'_'+str(numLags)+'lags'+str(batchsize)+'batchsize_reg_dropoutRMSE'+str(rmse)+'.png')
    plt.savefig('/home/alastair/Data/3d/overnight2/analysis/datatimeseries.png')