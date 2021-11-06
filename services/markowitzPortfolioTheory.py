import pandas as pd
import numpy as np

from utility import get_price

import scipy.optimize as sco
from pandas_datareader import data

# import json
from flask import Flask
from flask_cors import CORS


port = 5005

app = Flask(__name__)
CORS(app)

np.random.seed(777)


@app.route("/favicon.ico")
def favicon_fix():
    return "Okay", 200


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def downside_risk(weights, original_returns):
    daily_combined = (original_returns * weights).sum(axis=1)
    downside_deviation = np.clip(daily_combined, np.NINF, 0).std()
    return downside_deviation

########################################################################################################
# Efficient Frontier
########################################################################################################

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


########################################################################################################

def neg_sortino_ratio(weights, mean_returns, original_returns, risk_free_rate):
    returns = np.sum(mean_returns*weights)
    downside_deviation = downside_risk(weights, original_returns)
    neg_sortino = -(returns - risk_free_rate/252) * np.sqrt(252) / downside_deviation
    return neg_sortino


def max_sortino_ratio(mean_returns, original_returns, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, original_returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sortino_ratio, num_assets*[1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


########################################################################################################

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


########################################################################################################


def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, table, returns):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    max_sortino = max_sortino_ratio(mean_returns, returns, risk_free_rate)
    chosen_weights = max_sortino['x']
    sdp_sort = 0
    rp_sort = 0
    max_sortino_allocation = pd.DataFrame(max_sortino.x, index=table.columns, columns=['allocation'])
    max_sortino_allocation.allocation = [round(i*100, 2)for i in max_sortino_allocation.allocation]
    max_sortino_allocation = max_sortino_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    # annual vol/rt
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252

    output = []

    for i, weightage in enumerate(max_sharpe_allocation.iloc[0]):
        ticker = max_sharpe_allocation.columns.tolist()[i]
        output.append(
            {
                "ticker": ticker,
                "weight": weightage,
                "annualised_return": round(rp, 2),
                "annualised_volatility": round(sdp, 2),
                "type": "max_sharpe"
            }
        )

    for i, weightage in enumerate(min_vol_allocation.iloc[0]):
        ticker = min_vol_allocation.columns.tolist()[i]
        output.append(
            {
                "ticker": ticker,
                "weight": weightage,
                "annualised_return": round(rp_min, 2),
                "annualised_volatility": round(sdp_min, 2),
                "type": "min_volatility"
            }
        )

    for i, weightage in enumerate(max_sortino_allocation.iloc[0]):
        ticker = max_sortino_allocation.columns.tolist()[i]
        output.append(
            {
                "ticker": ticker,
                "weight": weightage,
                "annualised_return": round(rp_sort, 2),
                "annualised_volatility": round(sdp_sort, 2),
                "type": "max_sortino"
            }
        )

    return output


@app.route("/<string:tickers>", methods=['GET'])
def home(tickers):

    table = pd.DataFrame()
    tickers = tickers.split(',')

    for ticker in tickers:
        col = get_price(ticker, 'Adj Close')
        col.columns = [ticker]
        table.index = col.index
        table = table.join(col)

    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.0178
    output = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, table, returns)

    return {
        "meta": {
            "table_name": f"Allocation (Markowitz) table if {','.join(tickers)}",
            "columns": ['ticker', 'weight', 'annualised_return', 'annualised_volatility', 'type']
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
