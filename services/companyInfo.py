from flask import Flask
from flask_cors import CORS
import yfinance as yf  # 2000 requests per hour per IP

port = 5003

app = Flask(__name__)
CORS(app)


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/favicon.ico")
def favicon_fix():
    return "Okay", 200


@app.route("/<string:tickers>")
def home(tickers):

    tickers = tickers.split(',')

    output = []
    column_names = ['ticker', 'company_name','sector','website','summary']
    xstr = lambda s: s or ""

    return_tickers = []

    for ticker in tickers:

        try:

            share_info = yf.Ticker(ticker).info
            company_name = xstr(share_info['shortName'])
            sector = xstr(share_info['sector'])
            symbol = xstr(share_info['symbol'])
            website = xstr(share_info['website'])
            summary = xstr(share_info['longBusinessSummary'])

            new_row = {'ticker': symbol, 'company_name': company_name, 'sector': sector, 'website': website,'summary':summary}
            output.append(new_row)
            return_tickers.append(ticker)

        except:
            pass

    return {
        "meta": {
            "table_name": f"Company info of {','.join(return_tickers)}",
            "columns": column_names
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
