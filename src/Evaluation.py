import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.externals import joblib

'''
TODO: Model only returns True for buy and sell. Check it
'''


class Evaluator:

    def __init__(self, model_path, eval_data_path, scaler_path=None):
        self.model = self.load_model(model_path)
        self.evaluation_data = self.load_evaluation_data(eval_data_path)
        if scaler_path is not None:
            self.scaler = self.load_scaler(scaler_path)
        self.framesize = self.file_to_framesize()
        self.y_test = None
        self.X_test = None
        self.predictions = None

    @staticmethod
    def load_model(path):
        # use tensorflow load_model method to load model
        return load_model(path)

    @staticmethod
    def load_scaler(path):
        # use sklearn.externals.joblib load method to load scaler which was used to transform training data of model
        return joblib.load(path)

    @staticmethod
    def load_evaluation_data(path):
        evaluation_data = pd.read_csv(path)
        # add new column Open_before which contains the open values of the previous day
        evaluation_data["Open_before"] = evaluation_data["Open"].shift(1)

        # calculate the procentual change of the open value of the current day to the
        # open value of the day before
        evaluation_data["Open_changes"] = (evaluation_data["Open"] / evaluation_data["Open_before"]) - 1

        # throw out the first line which has NaN as value because of the previous shift of values
        evaluation_data = evaluation_data.dropna()
        # reset index to original
        evaluation_data = evaluation_data.reset_index(drop=True)
        # drop old index
        evaluation_data = evaluation_data.drop(columns=['Unnamed: 0'])

        # resort data frame by start backwards
        return evaluation_data[::-1]

    def file_to_framesize(self):
        # round evaluation data length to a multiple of 2500
        # e.g. len(evaluation_data) is 11.237 -> then it will return 10.000
        return int(len(self.evaluation_data) / 2500) * 2500

    def normalize(self):
        # changes describe the changes from the day before to the current day
        changes = self.evaluation_data["Open_changes"]

        # reshape changes to target shape of scaler
        changes = np.array(changes).reshape(-1, 1)

        # normalize test data accordingly to the train data
        # scaler was trained by training data which was used for training the model
        test = self.scaler.transform(changes)

        # reshape test data back to original format
        test = test.reshape(-1)

        # return normalized test data
        return test

    def create_prediction_data(self, normalize=True):
        # normalize the test data when normalize is true
        if normalize:
            test = self.normalize()
        # if normalize is false just load evaluation data
        else:
            test = self.evaluation_data["Open_changes"]

        # correct regression values stored in y_test
        self.y_test = np.array(test)

        # empty X_test which will be used for predition
        self.X_test = []

        # create X_test values by iterating target values "test"
        # for each round build an array row which contains
        # 1. the 20 previous changes at positions 0-19
        # 2. the sentiment at this specific day at position 20
        #
        # Therefore position i+1 to i+21 describe the changes value for the 20 previous days
        # position i describes the current day
        for i in range(0, len(test) - 20):
            # add at position 0 - 19 the changes of the previous 20 days (i+1, i+21)
            to_add = test[i + 1:i + 21].tolist()
            # add at position 20 the sentiment of the current day (i)
            to_add.append(self.evaluation_data['Sentiment'][i])
            # append row with previous values and sentiment to X_test
            self.X_test.append(to_add)

        # reshape test values to target format of the model
        self.X_test = np.array(self.X_test).reshape(-1, 21, 1)

    def buy_or_sell(self):
        # array contains for each day whether to buy or to sell
        # naive implementation is used. Buy when model prediction is greater than 0. Otherwise sell
        buy_or_sell = []

        # predict whether to buy or to sell using the loaded model and preprocessed dataset
        self.predictions = self.model.predict(self.X_test)

        # reshape predictions to 1D numpy array
        self.predictions = np.array(self.predictions).reshape(-1)

        # Iterate predictions and buy stock (append True to buy_or_sell array) when prediction is >1
        # otherwise sell stock (append False to buy_or_sell array)
        for change in self.predictions:
            if change > 0:
                buy_or_sell.append(True)
            else:
                buy_or_sell.append(False)

        # append 20 zeros to predictions because these are the start values of the frames
        self.predictions = np.append(self.predictions, np.array([0 for i in range(0,20)]))
        # save predictions in evalutation data
        self.evaluation_data["predictions"] = self.predictions
        # save predicted open stock price in evaluation data
        self.evaluation_data["Open_predicted"] = self.evaluation_data["Open_before"] * \
                                                 (1 + self.evaluation_data["predictions"])

        # return resorted buy_or_sell array
        # Resorting is necessary because first entry in buy or sell describes oldest datapoint
        return buy_or_sell[::-1]

    def market_return(self):
        # rendite describes the actual market return
        # Market return is given by day N stock price divided by day 0 stock price
        # where N is the newest date of the dataset
        rendite = self.evaluation_data["Open"][len(self.evaluation_data) - 1] / self.evaluation_data["Open"][0]
        a = self.evaluation_data["Open"][0]
        # return market return with 100€ start capital
        return 100 * rendite

    def model_return(self, kapital=100):
        # amount of stocks in depot at day -1
        depot = 0

        # Naive assumption: Always sell complet depot or buy as many stocks as money is available
        #
        # Iterate buy_or_sell predictions and evaluation data
        # Buy stock when prediction to buy is True AND depot is liquid (kapital > 0)
        # Sell all stocks in depot when prediction is False and there are stocks in depot
        # Do nothing (hold stocks or dont buy new) in every other case
        for (prediction, i) in zip(self.buy_or_sell(),
                                   range(0, len(self.evaluation_data) - 20)):
            if prediction == True and kapital > 0:
                # buy stock with opening price at day i
                kurs = self.evaluation_data["Open"][i]
                # depot value is given by
                # depot[Aktien] = Kapital[Euro] * (1 [Aktie] / Kurs [Euro pro Aktie])
                depot += kapital * (1 / kurs)
                # kapital is 0 after buying all stocks
                kapital = 0
            elif depot > 0:
                # sell all stocks at day i
                kurs = self.evaluation_data["Open"][i]
                # capital is given by the amount of stocks in depot multiplied with the current open day price at day i
                kapital = depot * kurs
                # depot is zero after selling all stocks
                depot = 0

        # return of the model with 100€ start capital
        # return capital if we have sold our stocks in the last iteration
        # otherwise calculate capital as in the depot > 0 iteration with the stock price of the last day
        return kapital if kapital > 0 else depot * self.evaluation_data["Open"][len(self.evaluation_data ) - 1]

    def plot_predictions(self):
        # plot the actual open price
        plt.plot(self.evaluation_data.index, self.evaluation_data["Open"], label="Real Open")
        # plot the predicted open price
        plt.plot(self.evaluation_data.index, self.evaluation_data["Open_predicted"],
                 label="Predicted Open")
        # set x axis label to index dax
        plt.xlabel("Index")
        # set y axis label to open courses
        plt.ylabel("Open course")
        # plot the legend to distinguish both graphs
        plt.legend()
        # show plot
        plt.show()
