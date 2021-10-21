import json
from flask import Flask
from flask_cors import CORS
import yfinance as yf  # 2000 requests per hour per IP
from typing import Dict, List, Any
import pandas as pd

port = 5003

app = Flask(__name__)
CORS(app)


def dict_to_df(dictionary: Dict[str, Any], columns: List[str] = ["Key", "Value"]) -> pd.core.frame.DataFrame:
    """
    Converts a vertical python dictionary into a 2D pandas Dataframe, with 2 columns.
    Each key corresponds to a row in the dataframe.

    Usage:
    df = dict_to_df(price_dict)
    """
    data = {columns[0]: list(dictionary.keys()),
            columns[1]: list(dictionary.values())}
    df = pd.DataFrame.from_dict(data)
    return df


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/<string:tickers>")
def home(tickers):

    tickers = tickers.split(',')

    output = []

    for ticker in tickers:
        share_info = yf.Ticker(ticker)

        filter = ['sector', 'country', 'companyOfficers', 'totalAssets', 'bookValue',
                  'profitMargins', 'lastSplitDate', 'lastSplitFactor', 'lastDividendDate', 'lastCapGain']
        info_dict = {k: v for k, v in share_info.info.items() if k in filter} 

        df = dict_to_df(info_dict, ["field", "value"])
        df['value'] = df['value'].apply(str)
        df['ticker'] = ticker
        df = df.to_json(orient='records')
        output += json.loads(df)

    print(output)
    return {
        "meta": {
            "table_name": f"Company info of {','.join(tickers)}",
            "columns": ["ticker", "field", "value"]
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
