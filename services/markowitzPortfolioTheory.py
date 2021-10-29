import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.optimize as sco

from pandas_datareader import data
# import matplotlib.pyplot as plt

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
    std = np.sqrt(np.dot(weights.T, np.dot(
        cov_matrix, weights))) * np.sqrt(252)
    return std, returns


# Efficient Frontier
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(
        weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


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


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[
                          1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, table, returns
                             ):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(
        max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(
        max_sharpe.x, index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [
        round(i*100, 2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(
        min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(
        min_vol.x, index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [
        round(i*100, 2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

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

    return output

    # print("-"*80)
    # print("Maximum Sharpe Ratio Portfolio Allocation\n")
    # print("Annualised Return:", round(rp, 2))
    # print("Annualised Volatility:", round(sdp, 2))
    # print("\n")
    # print(max_sharpe_allocation)
    # print("-"*80)
    # print("Minimum Volatility Portfolio Allocation\n")
    # print("Annualised Return:", round(rp_min, 2))
    # print("Annualised Volatility:", round(sdp_min, 2))
    # print("\n")
    # print(min_vol_allocation)

    # print("Individual Stock Returns and Volatility\n")
    # for i, txt in enumerate(table.columns):
    #     print(txt, ":", "annuaised return", round(
    #         an_rt[i], 2), ", annualised volatility:", round(an_vol[i], 2))

    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.scatter(an_vol,an_rt,marker='o',s=200)

    # for i, txt in enumerate(table.columns):
    #     ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    # ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    # ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    # target = np.linspace(rp_min, 0.34, 50)
    # efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    # ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    # ax.set_title('Portfolio Optimization with Individual Stocks')
    # ax.set_xlabel('annualised volatility')
    # ax.set_ylabel('annualised returns')
    # ax.legend(labelspacing=0.8)


# tickers = ['BABA', 'KWEB', 'SPOT', 'OIL', 'VOO', 'FB']

@app.route("/<string:tickers>", methods=['GET'])
def home(tickers):

    tickers = tickers.split(',')
    #!! to standardise time series with a price getter service (then cache?)
    start_date = '2019-9-1'
    end_date = '2021-8-31'
    panel_data = pd.DataFrame()

    for t in tickers:
        t_data = data.DataReader(t, 'yahoo', start_date, end_date)
        t_data = t_data.loc[start_date: end_date]
        t_data = t_data[['Adj Close']]
        t_data = t_data.reset_index()
        t_data["ticker"] = t
        t_data = t_data.rename(
            columns={"Date": "date", t_data.columns[1]: "adj_close"})
        panel_data = panel_data.append(t_data)

    df = panel_data.set_index('date')
    table = df.pivot(columns='ticker')
    # By specifying col[1] in below list comprehension
    # You can select the stock names under multi-level column
    table.columns = [col[1] for col in table.columns]

    # plt.figure(figsize=(14, 7))
    # for c in table.columns.values:
    #     plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
    # plt.legend(loc='upper left', fontsize=12)
    # plt.ylabel('price in $')

    returns = table.pct_change()

    # plt.figure(figsize=(14, 7))
    # for c in returns.columns.values:
    #     plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
    # plt.legend(loc='upper right', fontsize=12)
    # plt.ylabel('daily returns')

    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.0178
    output = display_ef_with_selected(mean_returns, cov_matrix,
                                      risk_free_rate, table, returns)

    return {
        "meta": {
            "table_name": f"Allocation (Markowitz) table if {','.join(tickers)}",
            "columns": ['ticker', 'weight', 'annualised_return', 'annualised_volatility', 'type']
        },
        "data": output,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)
