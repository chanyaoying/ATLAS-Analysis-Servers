from datetime import datetime
import json

from numpy import NaN, nan
from numpy.lib.type_check import nan_to_num
from werkzeug.wrappers import ResponseStreamMixin
import yahoo_fin.stock_info as si
import pandas as pd
from flask import Flask
from flask_cors import CORS

port = 5004

app = Flask(__name__)
CORS(app)


def parse_epoch_time(df: pd.core.frame.DataFrame, name: str):
    cols = df.columns.to_list()
    new_cols = []
    month_sorted = sorted(list(map(lambda n: (n.strftime(
        '%Y-%m-%d'), n.strftime('%Y-%m-%d')[5:7]), cols)), key=lambda n: n[1])
    month_order = [(item[0], rank) for rank, item in enumerate(month_sorted)]
    month_order = dict(month_order)
    for epoch_time in cols:
        date = epoch_time.strftime('%Y-%m-%d')
        if 'yearly' in name:
            date = date[:4]
        elif 'quarterly' in name:
            date = date[:4] + f" Q{month_order[date] + 1}"
        new_cols.append(date)
    return new_cols


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200
    

@app.route("/favicon.ico")
def favicon_fix():
    return "Okay", 200


@app.route("/<string:tickers>", methods=['GET'])
def home(tickers):
    tickers_list = tickers.split(',')
    output = []

    for ticker in tickers_list:
        try:
            financials = si.get_financials(ticker)
            stock_financials = []
            for name, item in financials.items():
                periods = parse_epoch_time(item, name)
                row = []

                for j, period in enumerate(periods):
                    for i, breakdown in enumerate(item.index):
                        value = item.iloc[i, j]
                        try:
                            value = float(nan_to_num(value)) if nan_to_num(value) else 0.0
                            row.append(
                            {
                                'ticker': ticker,
                                'type': name,
                                'period': period,
                                'breakdown': breakdown,
                                'value': value
                            }
                        )
                        except Exception as e:
                            print(f"{e} :: {value}")
                            {
                                'ticker': ticker,
                                'type': name,
                                'period': period,
                                'breakdown': breakdown,
                                'value': 0
                            }
                        
                stock_financials += row  # for each type of statement
            output += stock_financials  # for each ticker
        except Exception as e:
            print(f"ERROR: {e}")
            pass

    return {
            "meta": {
                "table_name": f"Financial Statement of {tickers}",
                "columns": ["ticker", "type", "period", "breakdown", "value"],
            },
            "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)
