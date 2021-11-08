from typing import Tuple
import json
from datetime import datetime
import pandas as pd
from pandas_datareader import data
import redis


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


def get_start_end_date(period: int = 2) -> Tuple[str]:
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


def get_price(ticker: str, price_type: str) -> pd.core.frame.DataFrame:
    """
    Checks redis if the data for the past 2 years exist.
    Get the data if found.
    If not, use pandas data reader to store the result and return it.

    Returns a pandas dataframe of the close prices for the past 2 or 5 years from this month.
    """
    cache = redis.Redis(host="ATLAS_price_cache", port=6379)
    start_date, end_date, key = get_start_end_date()
    print(start_date, end_date)
    key = f"{ticker}_{key}"
    query = cache.hgetall(key)

    if not query:  # if price data not found in redis
        print("pandas data reader used")
        df = data.DataReader([ticker], 'yahoo', start_date, end_date)[price_type]
        price_dict = df_to_dict(df, key)
        with cache.pipeline() as pipe:
            for key, price_data in price_dict.items():
                pipe.hmset(key, price_data)
                print("inserted!")
            pipe.execute()
        print("insert complete")
        return df
    else:
        print("Price data found.")
        return dict_to_df(query, key)

    
