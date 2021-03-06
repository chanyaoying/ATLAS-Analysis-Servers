# standard packages
import pandas as pd
from datetime import datetime, timedelta
from math import *
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# price cache
from utility import get_price

# Statistics
from pmdarima import auto_arima

# Flask
from flask import Flask
from flask_cors import CORS

port = 5002

app = Flask(__name__)
CORS(app)


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/favicon.ico")
def favicon_fix():
    return "Okay", 200


@app.route("/<string:tickers>", methods=['GET'])
def home(tickers):

    # Parameters
    tickers = tickers.split(',')
    days_to_predict = 60
    output = []

    for ticker in tickers:

        # train size = 80% of the dataset
        # model_train = df.iloc[:int(df.shape[0]*0.80)]

        # test size = 20%
        # valid = df.iloc[int(df.shape[0]*0.80):]

        # train size = 100%
        model_train = get_price(ticker, 'Close')

        two_years = model_train.copy()
        two_years.columns = ['predictionPrice']
        two_years['ticker'] = ticker
        two_years['date'] = two_years.index
        # two_years['date'] = two_years['date'].apply(lambda epoch_time: epoch_time.strftime('%Y-%m-%d'))
        two_years = two_years.to_json(orient="records")

        output += json.loads(two_years)
        del two_years

        # train model
        model_arima = auto_arima(model_train, trace=False, error_action='ignore',
                                 start_p=1, start_q=1, max_p=3, max_q=3,
                                 suppress_warnings=True, stepwise=False, seasonal=False)

        model_arima.fit(model_train, disp=-1)

        model_predictions = pd.DataFrame(model_arima.predict(n_periods=days_to_predict), index=(
            datetime.strptime(model_train.index[-1], "%Y-%m-%d") + timedelta(days=i) for i in range(1, days_to_predict + 1)))
        model_predictions.columns = ['predictionPrice']
        model_predictions['ticker'] = ticker
        model_predictions['date'] = model_predictions.index
        model_predictions['date'] = model_predictions['date'].apply(
            lambda epoch_time: epoch_time.strftime('%Y-%m-%d'))
        model_predictions = model_predictions.to_json(orient="records")

        output += json.loads(model_predictions)

    return {
        "meta": {
            "table_name": f"Auto ARIMA table of {','.join(tickers)}",
            "columns": ["ticker", "date", "predictionPrice"]
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
