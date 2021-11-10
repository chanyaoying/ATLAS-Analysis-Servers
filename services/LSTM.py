import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utility import get_price, get_start_end_date
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def train_model(tickers):

    tickers = tickers.split(',')
    for ticker in tickers:
        close = get_price(ticker, "Close", 5)
        close_array = close.values
        sc = MinMaxScaler(feature_range=(0, 1))
        scaled_close = sc.fit_transform(close_array)

        train = scaled_close[0:int(len(scaled_close)*0.8)]
        valid = scaled_close[int(len(scaled_close)*0.8):]

        X_train = []
        y_train = []

        for i in range(180, len(train)):
            # predict stock price based on past 6 months of data
            X_train.append(scaled_close[i-180:i])
            y_train.append(scaled_close[i])


        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()

        #Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        model.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(units = 1))

        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Fitting the RNN to the Training set
        model.fit(X_train, y_train, epochs = 100, batch_size = 32)

        # Testing on validation set
        df_scaled_close=pd.DataFrame(data=scaled_close, 
                                    index=close.index, 
                                    columns=['Close'])
        inputs = df_scaled_close[len(df_scaled_close) - len(valid) - 180:].values
        inputs = inputs.reshape(-1,1)

        X_test = []
        for i in range(180,inputs.shape[0]):
            X_test.append(inputs[i-180:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        # Predict the stock price
        close_price = model.predict(X_test)

        # unscaling the predicted value
        close_price = sc.inverse_transform(close_price)

        # Store validation data in cache
        a, b , key = get_start_end_date(5)
        del a
        del b
        valid_key = f"LTSM_{ticker}_{key}_valid"

        previous_180=close[-180:].values
        date = close[-1:].index[0]
        dates = []
        forecast_days = 60


