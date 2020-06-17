import datetime
import pandas as pd
import sklearn.preprocessing as skpp
import re
import numpy as np
from pandas import concat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as itgrt


non_decimal = re.compile(r'[^\d.]+')


def parser(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


class CsvPreprocess():
    def __init__(self, working_dir, csv_filename):
        self.parent_folder = working_dir
        self.csv_file = csv_filename
        self.csv_dataframe = pd.read_csv(self.csv_file, parse_dates=True, header=0, index_col=0, date_parser=parser)


    def load_variables_csv(self, csv_name):
        """Loads csv into pandas dataframe, returns it

        :param csv_name: path to and including the csv
        :return: pandas dataframe of the csv
        """


    def resample_csv(self, csv_name):
        """Resamples the csv to higher frequency, as per the image resampling

        :param csv: path to and including the csv
        :return:
        """
        df = self.load_variables_csv(csv_name)

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

    def remove_text(self, text_dataset):
        # Takes Pandas DataFrame as an argument
        # Remove all text from the dataset
        # returns a matrix of the values - not a dataframe
        numDataset = text_dataset.values
        for col in range(0, len(text_dataset.columns) - 1):
            # inn[index,col] = ([re.sub(non_decimal, "", elem) for index, elem in enumerate(inn[:,col])])
            # https://stackoverflow.com/questions/12116717/does-pythons-re-sub-take-an-array-as-the-input-parameter
            for index, elem in enumerate(numDataset[:, col]):
                if type(elem) == float: continue
                if type(elem) == int: continue
                numDataset[index, col] = re.sub(non_decimal, "", elem)

        return numDataset

    def plot_fft_column_timeseries(self, column_index, column_name, plot_name):
        """Gets and plots fft of a column (single variable) of the csv dataset


        :param column_index: index of the column to be analysed
        :param column_name: name of the column to be analysed
        :param plot_name: what the final plot should be saved as
        :type column_index int
        :type column_name str
        :type plot_name str
        :return:
        """
        dataset = self.csv_dataframe

        dataset.index.name = 'datetime'
        dataset.columns = ['dn', 'dm', 'dx', 'sn', 'sm', 'sx', 'ta', 'ua', 'pa', 'fan', 'current', 'voltage']
        data_points_numerical = self.remove_text(dataset)
        values = data_points_numerical.astype('float32')

        ts = values[:, column_index]
        n = len(values[:, column_index])
        d = 60.0  # samples are approx once per minute #TODO sure this up
        fig, ax = plt.subplots()
        sample_freqs = np.fft.rfftfreq(n, d)
        fourier = np.fft.rfft(ts)
        ax.plot(sample_freqs[1:], fourier.real[1:(len(ts) / 2) + 1], label='real, ' + column_name)
        ax.plot(sample_freqs[1:], fourier.imag[1:(len(ts) / 2) + 1], label='imag, ' + column_name)
        ax.legend()
        fig.savefig(
            self.parent_folder + 'analysis/'+plot_name+'.png')
        fig.savefig(
            self.parent_folder + 'analysis/'+plot_name+'.svg')
        fig.clf()

    def get_column_integral(self, column_index=10):
        """
            default to current, it's the variable this was designed for
        :param column_index:
        :return:
        """

        dataframe = self.csv_dataframe
        column = dataframe.values[:, column_index]
        length_of_dataset = len(column)
        for i in range(0, length_of_dataset):
            integrals = []

            #print column[0:i]
            integral_this_interval = np.cumsum(column[0:i])

            integrals.append(integral_this_interval)

        print(integrals)


    def normalise_and_plot(self):
        #columns
        #datetime,dn, dm, dx, sn, sm, sz, Ta, Ua, Pa, fan, current, voltage
        # 0, 1, 2 ,3 ,4 ,5 ,6 ,7 ,8 ,9, 10, 11 ,12
        dataset = pd.read_csv(self.csv_file, parse_dates=True, header=0, index_col=0, date_parser=parser)
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

        data_points_numerical = self.removeText(dataset)
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