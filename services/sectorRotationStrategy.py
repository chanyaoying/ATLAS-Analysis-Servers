import json
import scipy.optimize as sco
from pandas_datareader import data
import pandas as pd
import numpy as np
from utility import get_price, get_optimal_allocation

# Flask
from flask import Flask
from flask_cors import CORS


portfolio_tickers = ['SPY', 'QQQ', 'TLT', 'OIL']  # current portfolio
portfolio_weights = [0.25, 0.25, 0.25, 0.25]  # current weights

portfolio_2y_adj_close = pd.DataFrame()
portfolio_2y_close = pd.DataFrame()

benchmark_price = get_price('SPY', 'Adj Close')
benchmark_return_series = (benchmark_price.pct_change() + 1).cumprod() - 1
benchmark_title = 'Benchmark (SPY)'

for ticker in portfolio_tickers:
    col_adj_close = get_price(ticker, 'Adj Close')
    col_adj_close.columns = [ticker]
    portfolio_2y_adj_close.index = col_adj_close.index
    portfolio_2y_adj_close = portfolio_2y_adj_close.join(col_adj_close)

    col_close = get_price(ticker, 'Close')
    col_close.columns = [ticker]
    portfolio_2y_close.index = col_close.index
    portfolio_2y_close = portfolio_2y_close.join(col_close)

portfolio_3m_close = portfolio_2y_close.iloc[(
    len(portfolio_2y_close.index) // 6) * 5:-1]

sector_etf = ['XLE', 'XLF', 'XLK', 'XLRE', 'XLY',
              'XLI', 'XLB', 'XLC', 'XLV', 'XLP', 'XLU']

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


weighted_portfolio_3m_close = pd.DataFrame(
    (portfolio_weights * portfolio_3m_close).sum(axis=1))
weighted_portfolio_3m_close = weighted_portfolio_3m_close.rename(columns={
                                                                 0: 'portfolio'})
all_3m_close = sector_3m_close.join(weighted_portfolio_3m_close)
all_3m_close_return_series = (all_3m_close.pct_change() + 1).cumprod() - 1
correlation = pd.DataFrame(all_3m_close_return_series.corr().tail(1).round(3))


# top3_inverse_sectors
correlation = correlation.sort_values(by="portfolio", axis=1)
top3_inverse_sectors = correlation.columns.tolist()[0:3]

# select 3 representative tickers by sectors;
sector_dict = {
    'XLB': ['LIN', 'ECL', 'GOLD'], 'XLC': ['GOOGL', 'FB', 'NFLX'], 'XLY': ['AMZN', 'TSLA', 'NKE'], 'XLP': ['KO', 'WMT', 'PEP'], 'XLE': ['XOM', 'CVX', 'PSX'], 'XLF': ['V', 'MA', 'JPM'], 'XLV': ['JNJ', 'PFE', 'UNH'], 'XLI': ['HON', 'GE', 'FDX'], 'XLR': ['AMT', 'CCI', 'PSA'], 'XLK': ['MSFT', 'AAPL', 'CRM'], 'XLU': ['NEE', 'DUK', 'XEL']
}

lagging_tickers = []
[lagging_tickers.extend(sector_dict[etf]) for etf in top3_inverse_sectors]

new_portfolios = [portfolio_tickers + [ticker] for ticker in lagging_tickers]
new_portfolios_weights = {lagging_tickers[i]: get_optimal_allocation(
    new_portfolios[i]) for i in range(len(lagging_tickers))}


output = []
for suggested_ticker, weights in new_portfolios_weights.items():

    types = ['sharpe', 'min_vol', 'sortino', 'equal', 'ticker_only']
    suggested_ticker_2y_adj_close = get_price(suggested_ticker, 'Adj Close')

    for i in range(len(types)):
        suggested_portfolio = portfolio_2y_adj_close.copy().join(
            suggested_ticker_2y_adj_close)
        suggested_portfolio = (suggested_portfolio.pct_change() + 1).cumprod() - 1
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

print(output)


# # use 2y for rs
# start_date = '2019-01-01'
# end_date = '2020-12-31'
# panel_data = data.DataReader(
#     inverse_sector_tickers, 'yahoo', start_date, end_date)

# sector_closes_2y = panel_data[['Adj Close']]
# sector_closes_2y = sector_closes_2y.loc[start_date: end_date]


# ###########################################################
# # portfolio_2y_return_series_adj
# ###########################################################
# # portfolio_tickers = ['SPY','AMZN','AAPL']
# # weights = [0.5, 0.5, 0.0]
# # start_date = '2019-01-01'
# # end_date = '2020-12-31'

# panel_data = data.DataReader(portfolio_tickers, 'yahoo', start_date, end_date)

# closes_2y = panel_data[['Adj Close']]
# closes_2y = closes_2y.loc[start_date: end_date]

# # // return series for the period and plot the returns on a single chart.
# return_series_adj = (closes_2y['Adj Close'].pct_change() + 1).cumprod() - 1

# # portfolio returns
# weighted_return_series_adj_2y = weights * (return_series_adj)
# return_series_adj_2y = weighted_return_series_adj_2y.sum(axis=1)


# ###########################################################
# # Efficient Frontiner from Modern Portfolio Theory
# ###########################################################

# np.random.seed(777)


# def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
#     returns = np.sum(mean_returns*weights) * 252
#     std = np.sqrt(np.dot(weights.T, np.dot(
#         cov_matrix, weights))) * np.sqrt(252)
#     return std, returns


# def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
#     p_var, p_ret = portfolio_annualised_performance(
#         weights, mean_returns, cov_matrix)
#     return -(p_ret - risk_free_rate) / p_var


# def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
#     num_assets = len(mean_returns)
#     args = (mean_returns, cov_matrix, risk_free_rate)
#     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bound = (0.0, 1.0)
#     bounds = tuple(bound for asset in range(num_assets))
#     result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets, ], args=args,
#                           method='SLSQP', bounds=bounds, constraints=constraints)
#     return result


# def portfolio_volatility(weights, mean_returns, cov_matrix):
#     return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


# def min_variance(mean_returns, cov_matrix):
#     num_assets = len(mean_returns)
#     args = (mean_returns, cov_matrix)
#     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bound = (0.0, 1.0)
#     bounds = tuple(bound for asset in range(num_assets))

#     result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets, ], args=args,
#                           method='SLSQP', bounds=bounds, constraints=constraints)

#     return result


# def efficient_return(mean_returns, cov_matrix, target):
#     num_assets = len(mean_returns)
#     args = (mean_returns, cov_matrix)

#     def portfolio_return(weights):
#         return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

#     constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
#                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bounds = tuple((0, 1) for asset in range(num_assets))
#     result = sco.minimize(portfolio_volatility, num_assets*[
#                           1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
#     return result


# def efficient_frontier(mean_returns, cov_matrix, returns_range):
#     efficients = []
#     for ret in returns_range:
#         efficients.append(efficient_return(mean_returns, cov_matrix, ret))
#     return efficients


# def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
#     max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
#     sdp, rp = portfolio_annualised_performance(
#         max_sharpe['x'], mean_returns, cov_matrix)
#     max_sharpe_allocation = pd.DataFrame(
#         max_sharpe.x, index=table.columns, columns=['allocation'])
#     max_sharpe_allocation.allocation = [
#         round(i*100, 2)for i in max_sharpe_allocation.allocation]
#     max_sharpe_allocation = max_sharpe_allocation.T
#     max_sharpe_allocation

#     min_vol = min_variance(mean_returns, cov_matrix)
#     sdp_min, rp_min = portfolio_annualised_performance(
#         min_vol['x'], mean_returns, cov_matrix)
#     min_vol_allocation = pd.DataFrame(
#         min_vol.x, index=table.columns, columns=['allocation'])
#     min_vol_allocation.allocation = [
#         round(i*100, 2)for i in min_vol_allocation.allocation]
#     min_vol_allocation = min_vol_allocation.T

#     an_vol = np.std(returns) * np.sqrt(252)
#     an_rt = mean_returns * 252

#     ret_list = [round(rp, 2), round(sdp, 2), max_sharpe_allocation.iloc[0, 0], max_sharpe_allocation.iloc[0, 1], round(
#         rp_min, 2), round(sdp_min, 2), min_vol_allocation.iloc[0, 0], min_vol_allocation.iloc[0, 1]]

#     for i, txt in enumerate(table.columns):
#         ret_list.append(round(an_rt[i], 2))
#         ret_list.append(round(an_vol[i], 2))
#     ret_list.append(pd.DataFrame(mean_returns).index.tolist()[0])
#     return ret_list


# # 2y
# # portfolio_tickers = ['SPY','AMZN','AAPL']
# start_date = '2019-01-01'
# end_date = '2020-12-31'
# panel_data = data.DataReader(portfolio_tickers, 'yahoo', start_date, end_date)
# closes_2y = panel_data[['Adj Close']]
# closes_2y = closes_2y.loc[start_date: end_date]


# weighted_closes_2y = pd.DataFrame(
#     (weights * closes_2y["Adj Close"]).sum(axis=1))
# weighted_closes_2y = weighted_closes_2y.reset_index()
# weighted_closes_2y = weighted_closes_2y.rename(
#     columns={"Date": "date", 0: "adj_close"})
# weighted_closes_2y["ticker"] = "portfolio"


# sector_res = []
# for t in inverse_sector_tickers:

#     panel_data = pd.DataFrame()
#     t_data = data.DataReader(t, 'yahoo', start_date, end_date)

#     t_data = t_data.loc[start_date: end_date]
#     t_data = t_data[['Adj Close']]

#     t_data = t_data.reset_index()
#     t_data["ticker"] = t
#     t_data = t_data.rename(
#         columns={"Date": "date", t_data.columns[1]: "adj_close"})
#     panel_data = panel_data.append(t_data)

#     panel_data = panel_data.append(weighted_closes_2y)

#     df = panel_data.set_index('date')
#     df.head()

#     table = df.pivot(columns='ticker')
#     table.columns = [col[1] for col in table.columns]
#     table.head()

#     returns = table.pct_change()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()

#     risk_free_rate = 0.0178
#     sector_res.append(display_ef_with_selected(
#         mean_returns, cov_matrix, risk_free_rate))


# ######################################################################
# # content for each sector in ret_list
# #####################################################################
# # Maximum Sharpe Ratio Portfolio Allocation

# # 1) Annualised Return: 0.39
# # 2) Annualised Volatility: 0.29
# #              JPM  portfolio
# # 3) allocation- sector ticker  6.81
# # 4) allocation- portfolio 93.19

# #####################################################################
# # Minimum Volatility Portfolio Allocation

# # 5) Annualised Return: 0.36
# # 6) Annualised Volatility: 0.28
# #              JPM  portfolio
# # 7) allocation- sector ticker  28.11
# # 8) allocation- portfolio 71.89
# #####################################################################
# # Individual Stock Returns and Volatility

# # 9) annuaised return- sector ticker;
# # 10) annualised volatility- sector ticker;
# # 11) annuaised return- portfolio;
# # 12) annualised volatility- portfolio;

# # 13) ticker_symbol
# #####################################################################

# # print(sector_res)
# # import json
# # export_sector_res = pd.DataFrame(sector_res).to_json(orient="records")
# # parsed = json.loads(export_sector_res)
# # json.dumps(parsed, indent=4)

# ###########################################################
# # Maximum Sharpe Ratio Portfolio Allocation
# ###########################################################
# max_sharpe_w_inverse_sectors = pd.DataFrame()
# max_sharpe_w_inverse_sectors["current portfolio"] = return_series_adj_2y
# for sector in sector_res:
#     # portfolio returns
#     weights = [sector[2], sector[3]]
#     sector_closes_2y_return_series_adj = (
#         sector_closes_2y['Adj Close'].pct_change() + 1).cumprod() - 1
#     return_series_adj_2y_new_sector = pd.DataFrame(sector_closes_2y_return_series_adj[sector[12]]).join(
#         pd.DataFrame(return_series_adj_2y)).rename(columns={0: 'portfolio'})
#     weighted_return_series_adj = weights * (return_series_adj_2y_new_sector)

#     # Sum the weighted returns for portfolio
#     inverse_portfolio_rs_adj = weighted_return_series_adj.sum(axis=1)/100
#     max_sharpe_w_inverse_sectors["+ " + sector[12]] = inverse_portfolio_rs_adj


# # print(max_sharpe_w_inverse_sectors)
# # export_max_sharpe = max_sharpe_w_inverse_sectors.to_json(orient="records")
# # parsed = json.loads(export_max_sharpe)
# # json.dumps(parsed, indent=4)


# ############################################################
# # Minimum Volatility Portfolio Allocation
# ###########################################################

# min_vol_w_inverse_sectors = pd.DataFrame()
# min_vol_w_inverse_sectors["current portfolio"] = return_series_adj_2y

# for sector in sector_res:
#     # portfolio returns
#     weights = [sector[6], sector[7]]

#     sector_closes_2y_return_series_adj = (
#         sector_closes_2y['Adj Close'].pct_change() + 1).cumprod() - 1
#     # include ticker;
#     return_series_adj_2y_new_sector = pd.DataFrame(sector_closes_2y_return_series_adj[sector[12]]).join(
#         pd.DataFrame(return_series_adj_2y)).rename(columns={0: 'portfolio'})

#     weighted_return_series_adj = weights * (return_series_adj_2y_new_sector)

#     # Sum the weighted returns for portfolio
#     inverse_portfolio_rs_adj = weighted_return_series_adj.sum(axis=1)/100
#     min_vol_w_inverse_sectors["+ " + sector[12]] = inverse_portfolio_rs_adj


# # print(min_vol_w_inverse_sectors)
# # export_min_vol = min_vol_w_inverse_sectors.to_json(orient="records")
# # parsed = json.loads(export_min_vol)
# # json.dumps(parsed, indent=4)
