import json
import scipy.optimize as sco
from pandas_datareader import data
import pandas as pd
import numpy as np
from utility import get_price, get_optimal_allocation

# Flask
from flask import Flask
from flask_cors import CORS

port = 5006

app = Flask(__name__)
CORS(app)


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/favicon.ico")
def favicon_fix():
    return "Okay", 200

sector_etf = ['XLE', 'XLF', 'XLK', 'XLRE', 'XLY',
              'XLI', 'XLB', 'XLC', 'XLV', 'XLP', 'XLU']

sector_dict = {
    'XLB': ['LIN', 'ECL', 'GOLD'], 'XLC': ['GOOGL', 'FB', 'NFLX'], 'XLY': ['AMZN', 'TSLA', 'NKE'], 'XLP': ['KO', 'WMT', 'PEP'], 'XLE': ['XOM', 'CVX', 'PSX'], 'XLF': ['V', 'MA', 'JPM'], 'XLV': ['JNJ', 'PFE', 'UNH'], 'XLI': ['HON', 'GE', 'FDX'], 'XLR': ['AMT', 'CCI', 'PSA'], 'XLK': ['MSFT', 'AAPL', 'CRM'], 'XLU': ['NEE', 'DUK', 'XEL']
}


@app.route("/<string:tickers>/<string:allocation>")
def home(tickers, allocation):

    # Get SPY benchmark (return series)
    benchmark_price = get_price('SPY', 'Adj Close')
    benchmark_return_series = (benchmark_price.pct_change() + 1).cumprod() - 1
    benchmark_title = 'Benchmark (SPY)'
    
    portfolio_tickers = tickers.split(',')
    allocation = list(map(float, allocation.split(',')))
    total = sum(allocation)

    portfolio_weights = list(
        map(
            lambda amount: amount / total,
            allocation
        )
    )

    portfolio_2y_adj_close = pd.DataFrame()
    portfolio_2y_close = pd.DataFrame()

    for ticker in portfolio_tickers:
        col_adj_close = get_price(ticker, 'Adj Close')
        col_adj_close.columns = [ticker]
        portfolio_2y_adj_close.index = col_adj_close.index
        portfolio_2y_adj_close = portfolio_2y_adj_close.join(col_adj_close)

        col_close = get_price(ticker, 'Close')
        col_close.columns = [ticker]
        portfolio_2y_close.index = col_close.index
        portfolio_2y_close = portfolio_2y_close.join(col_close)

    # get 3 month close (just to perform correlation analysis)
    # cus we're assuming the cycle is 3 months
    portfolio_3m_close = portfolio_2y_close.iloc[(
        len(portfolio_2y_close.index) // 6) * 5:-1]

    sector_2y_adj_close = pd.DataFrame()
    sector_2y_close = pd.DataFrame()

    for etf in sector_etf:
        col_adj_close = get_price(etf, 'Adj Close')
        col_adj_close.columns = [etf]
        sector_2y_adj_close.index = col_adj_close.index
        sector_2y_adj_close = sector_2y_adj_close.join(col_adj_close)

        col_close = get_price(etf, 'Close')
        col_close.columns = [etf]
        sector_2y_close.index = col_close.index
        sector_2y_close = sector_2y_close.join(col_close)

    sector_3m_close = sector_2y_close.iloc[(
        len(sector_2y_close.index) // 6) * 5:-1]

    # get the weighted return series
    weighted_portfolio_3m_close = pd.DataFrame(
        (portfolio_weights * portfolio_3m_close).sum(axis=1))
    weighted_portfolio_3m_close = weighted_portfolio_3m_close.rename(columns={
        0: 'portfolio'})
    all_3m_close = sector_3m_close.join(weighted_portfolio_3m_close)
    all_3m_close_return_series = (all_3m_close.pct_change() + 1).cumprod() - 1
    correlation = pd.DataFrame(
        all_3m_close_return_series.corr().tail(1).round(3))

    
    correlation = correlation.sort_values(by="portfolio", axis=1)
    top3_inverse_sectors = correlation.columns.tolist()[0:2]

    lagging_tickers = []
    [lagging_tickers.extend(sector_dict[etf]) for etf in top3_inverse_sectors]

    new_portfolios = [portfolio_tickers + [ticker]
                      for ticker in lagging_tickers]
    new_portfolios_weights = {lagging_tickers[i]: get_optimal_allocation(
        new_portfolios[i]) for i in range(len(lagging_tickers))}

    output = []
    for suggested_ticker, weights in new_portfolios_weights.items():

        types = ['max_sharpe', 'min_vol', 'max_sortino', 'equal_weight', 'ticker_only']
        suggested_ticker_2y_adj_close = get_price(
            suggested_ticker, 'Adj Close')

        for i in range(len(types)):
            suggested_portfolio = portfolio_2y_adj_close.copy().join(
                suggested_ticker_2y_adj_close)
            suggested_portfolio = (
                suggested_portfolio.pct_change() + 1).cumprod() - 1
            suggested_portfolio = pd.DataFrame((suggested_portfolio * weights[i]).sum(axis=1)).rename(columns={
                0: 'returns'})
            suggested_portfolio['allocation_type'] = types[i]
            suggested_portfolio['allocation_weights'] = json.dumps(weights[i])
            suggested_portfolio['title'] = f"portfolio + {suggested_ticker}"
            suggested_portfolio['date'] = suggested_portfolio.index
            suggested_portfolio = suggested_portfolio.to_json(orient="records")
            output += json.loads(suggested_portfolio)

    benchmark_return_series.columns = ['returns']
    benchmark_return_series['allocation_type'] = 'benchmark'
    benchmark_return_series['allocation_weights'] = json.dumps(weights[i])
    benchmark_return_series['title'] = benchmark_title
    benchmark_return_series['date'] = benchmark_return_series.index
    benchmark_return_series = benchmark_return_series.to_json(orient="records")
    output += json.loads(benchmark_return_series)

    return {
        "meta": {
            "table_name": f"Sector Rotation Strategy for {','.join(portfolio_tickers)}",
            "columns": ["title", "date", "returns", "allocation_type", "allocation_weights"]
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
