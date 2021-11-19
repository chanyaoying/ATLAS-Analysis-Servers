import numpy as np
import pandas as pd
import scipy.optimize as sco


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(
        cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def downside_risk(weights, original_returns):
    daily_combined = (original_returns * weights).sum(axis=1)
    downside_deviation = np.clip(daily_combined, np.NINF, 0).std()
    return downside_deviation


def annualised_returns(weights, returns):
    return_series = (returns + 1).cumprod() - 1
    final_weighted_returns = np.sum(return_series.tail(1).mean() * weights)
    annualised_ret = (final_weighted_returns + 1)**(1/2) - \
        1  # 2 represents 2 years
    return annualised_ret


def annualised_volatility(series: pd.core.series.Series) -> float:
    return np.sqrt(np.log(series / series.shift(1)).var()) * np.sqrt(252)


def combined_annualised_volatility(weights, table):
    return np.sqrt(np.sum(np.square(table.apply(annualised_volatility)) * weights))

########################################################################################################
# Efficient Frontier
########################################################################################################


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
    result = sco.minimize(neg_sharpe_ratio, num_assets*[
                          1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


########################################################################################################

def neg_sortino_ratio(weights, mean_returns, original_returns, risk_free_rate):
    returns = np.sum(mean_returns*weights)
    downside_deviation = downside_risk(weights, original_returns)
    neg_sortino = -(returns - risk_free_rate/252) * \
        np.sqrt(252) / downside_deviation
    return neg_sortino


def max_sortino_ratio(mean_returns, original_returns, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, original_returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sortino_ratio, num_assets*[
                          1./num_assets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
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
