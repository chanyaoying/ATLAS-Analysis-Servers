from typing import Tuple, List
import json
from datetime import datetime
import pandas as pd
from pandas_datareader import data
import redis
from MPT_functions import max_sharpe_ratio, min_variance, max_sortino_ratio


def df_to_dict(df: pd.core.frame.DataFrame, key: str) -> dict:
    df.columns = [key]
    df.index = df.index.map(lambda epoch_time: epoch_time.strftime('%Y-%m-%d'))
    df = json.loads(df.to_json(orient="columns"))
    return df


def dict_to_df(dict_from_cache: dict, column: str) -> pd.core.frame.DataFrame:
    processed = {date.decode('utf-8'): float(price.decode('utf-8'))
                 for date, price in dict_from_cache.items()}
    data = {column: processed}
    return pd.DataFrame(data)


def get_start_end_date(period: int) -> Tuple[str]:
    """
    Returns as a tuple,
    1. start date: 2 years ago from the start of this month
    2. end date: the start of this month
    3. current year-month
    """
    current_month, current_year = datetime.now().strftime("%m,%Y").split(',')
    start_date = f"{int(current_year)-period}-{current_month}-01"
    end_date = f"{current_year}-{current_month}-01"
    return start_date, end_date, f"{current_year}-{current_month}_{period}years"


def get_price(ticker: str, price_type: str, period: int = 2) -> pd.core.frame.DataFrame:
    """
    Checks redis if the data for the past 2 years exist.
    Get the data if found.
    If not, use pandas data reader to store the result and return it.

    Returns a pandas dataframe of the close prices for the past 2 or 5 years from this month.
    """
    print(f"Getting price data for {ticker}:")
    cache = redis.Redis(host="ATLAS_price_cache", port=6379)
    start_date, end_date, key = get_start_end_date(period)
    key = f"{ticker}_{key}_{price_type}"
    query = cache.hgetall(key)

    if not query:  # if price data not found in redis
        print("Price data not found in cache. Pandas data reader used.")
        df = data.DataReader([ticker], 'yahoo', start_date, end_date)[price_type]
        price_dict = df_to_dict(df, key)
        with cache.pipeline() as pipe:
            for key, price_data in price_dict.items():
                pipe.hmset(key, price_data)
                print(f"{key} inserted into cache.")
            pipe.execute()
        print("Insertion complete.")
        return df
    else:
        print("Price data found in cache.")
        return dict_to_df(query, key)


def rounded_float_list(array: List[float]) -> List[float]:
    return list(
        map(
            lambda value: round(value, 2),
            array
        )
    )


def get_optimal_allocation(tickers: List[str]) -> List[float]:
    """
    Uses Post Modern Portfolio Theory to get the optimal weights for
    - Maximum Sharpe Ratio
    - Minimum Volatility
    - Maximum Sortino Ratio
    - Even weightage
    - Only new stock
    """
    table = pd.DataFrame()
    n = len(tickers)
    
    for ticker in tickers:
        col = get_price(ticker, 'Adj Close')
        col.columns = [ticker]
        table.index = col.index
        table = table.join(col)

    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.0178

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sharpe_weights = rounded_float_list(max_sharpe['x'])

    min_volatility = min_variance(mean_returns, cov_matrix)
    min_vol_weights = rounded_float_list(min_volatility['x'])

    max_sortino = max_sortino_ratio(mean_returns, returns, risk_free_rate)
    sortino_weights = rounded_float_list(max_sortino['x'])

    return sharpe_weights, min_vol_weights, sortino_weights, [1/n] * n, [0] * (n-1) + [1]

