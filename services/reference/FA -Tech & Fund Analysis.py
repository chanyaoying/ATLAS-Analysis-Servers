#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas_datareader import data 
import matplotlib.pyplot as plt 

import pandas as pd
import numpy as np
from scipy.stats import norm 
import pyfolio as pf 

from matplotlib.ticker import FuncFormatter
get_ipython().run_line_magic('matplotlib', 'inline')
import json


# In[2]:


#Get data for adj. close, close, high, low,open, volume
tickers = ['FB','AMZN','AAPL'] 
start_date = '2020-01-01'
end_date = '2020-12-31'
panel_data = data.DataReader(tickers,'yahoo', start_date, end_date) 
#Use loc method to ensure we do not consider extra days of data panel_data = panel_data.loc['2020-01-01':'2020-12-31']
panel_data.head()


# In[3]:


#convert multi-index column to single-index colum with insertion of new ticker column
all = []
for name, data in panel_data.groupby(level=1, axis=1):
    data.columns = data.columns.droplevel(1)
    data.insert(6, 'Tickers', name)
    data=data.reset_index()
    all.append(data)

result_df = pd.concat(all)
result_df


# In[4]:


import os
path = os.getcwd() # get directory of this notebook
path += "/json_data"


# In[5]:


#Export data to json
result_df.to_json(path_or_buf= path+"/trade_data.json",orient="records")


# ## Return Series

# In[ ]:


closes_1y = panel_data[['Close', 'Adj Close']]
return_series_adj = (closes_1y['Adj Close'].pct_change()+ 1).cumprod() - 1 #formula for return series
return_series_adj


# ## Volatility & Annual Return

# In[ ]:


#volatility
volatility = np.sqrt(np.log(closes_1y['Close'] / closes_1y['Close'].shift(1)).var()) * np.sqrt(252)
vol_df = volatility.to_frame(name="volatility").reset_index()

#annualized return
annualized_return = return_series_adj.tail(1).reset_index()

date_value =  annualized_return.iloc[0,0]

annualized_return.drop(columns = ['Date'],inplace=True) #drop date column
annualized_return = annualized_return.T.reset_index() #change to single index

#create date df
dup_date = [date_value] * len(annualized_return)
date_df = pd.DataFrame({"Date":dup_date})

#combine date df with ticker_annu_df
frames = [annualized_return,date_df]
annu_df = pd.concat(frames,axis= 1,join='outer',ignore_index=False,sort=False)

#Export vol_df & annu_df
vol_df.to_json(path_or_buf= path+ "/volatility.json",orient="records")
annu_df.to_json(path_or_buf= path+ "/annual_return.json",orient="records")


# ## Sharpe Ratio

# In[ ]:


# Risk Adjusted Return Calculations
#5. Calculate the Sharpe ratio for each fund. For the risk-adjusted return, assume that the risk-free rate is 1.0%.
# -> Which fund has the highest/ lowest ratio

#assume a risk free rate of 1%. Use 0% to match results with pyfolio
risk_free_ann_ret_rate = 0.01
returns_ts = closes_1y['Adj Close'].pct_change().dropna()
avg_daily_ret = returns_ts.mean()

#create the risk free rate column
returns_ts['RiskFree_Rate'] = risk_free_ann_ret_rate/252 
avg_rf_ret = returns_ts['RiskFree_Rate'].mean()
#calculate sharpe ratio
#Add the excess return columns for each ETF
returns_ts['Excess_ret_FB'] = returns_ts["FB"] - returns_ts['RiskFree_Rate'] 
returns_ts['Excess_ret_AMZN'] = returns_ts["AMZN"] - returns_ts['RiskFree_Rate'] 
returns_ts['Excess_ret_AAPL'] = returns_ts["AAPL"] - returns_ts['RiskFree_Rate']


# In[ ]:


#calculate sharpe ratio
sharpe_FB = ((avg_daily_ret['FB'] - avg_rf_ret) /returns_ts['Excess_ret_FB'].std())*np.sqrt(252)
print("Sharpe Ratio FB :\n", sharpe_FB.round(3))

sharpe_AMZN = ((avg_daily_ret['AMZN'] - avg_rf_ret) /returns_ts['Excess_ret_AMZN'].std())*np.sqrt(252) 
print("Sharpe Ratio AMZN :\n" , sharpe_AMZN.round(3))

sharpe_AAPL = ((avg_daily_ret['AAPL'] - avg_rf_ret) /returns_ts['Excess_ret_AAPL'].std())*np.sqrt(252) 
print("Sharpe Ratio AAPL :\n" , sharpe_AAPL.round(3))

sharpe_df = pd.DataFrame({"Tickers":['FB','AMZN','AAPL'],"Sharpe Ratio":[sharpe_FB,sharpe_AMZN,sharpe_AAPL]})

#Export sharpe df
sharpe_df.to_json(path_or_buf= path+"/sharpe_ratio.json",orient="records")


# ## Portfolio Return

# In[ ]:


closes_1y = panel_data[['Close', 'Adj Close']]
return_series_adj = (closes_1y['Adj Close'].pct_change()+ 1).cumprod() - 1 #formula for return series


# In[ ]:


# Portfolio Calculations
# 6. Portfolio: 50% FB and 50% AMZN
# Calculate the annualized return, volatility & risk-adjusted return for the portfolio.
# -> How does the total (annual) return for the combination of funds compare with taking the sum of 0.5 of the annual return of each fund?
# -> How does the annualized volitility for the combination of funds compare with taking the sum of 0.5 of the annualized volatiity of each fund?


#portfolio returns
weights= [0.2, 0.5, 0.3]
weighted_return_series_adj = weights * (return_series_adj) 

#Sum the weighted returns for FB,APPL,SPY
weighted_return_series_adj['Sum_Weighted_Return'] = weighted_return_series_adj.sum(axis=1)


#Remove adj_return_series cols as it has already been used to calculate for sum_weighted_returns
weighted_return_series_adj.drop(['FB','AMZN','AAPL'],inplace=True,axis=1) 
weighted_return_series_adj = weighted_return_series_adj.reset_index()

#rename adjusted_return_series
return_series_adj = return_series_adj.reset_index()
return_series_adj= return_series_adj.rename(columns={"FB": "FB_Adjusted_Return", "AMZN": "AMZ_Adjusted_Return","AAPL": "APPL_Adjusted_Return"})
return_series_adj

#Merge and export adjusted_retun_weighted_df
result_return_df = return_series_adj.merge(weighted_return_series_adj, left_on='Date', right_on='Date')
result_return_df.to_json(path_or_buf= path+"/portfolio_return.json",orient="records")


# ### Portfolio Volatility & Portfolio Return

# In[ ]:


#for portfolio volatility
#Note that to be able to apply weights, we cannot use raw prices for volatility calculation
return_series_close = (closes_1y['Close'].pct_change()+ 1).cumprod() - 1 
weighted_return_series_close= weights * (return_series_close) 
return_series_close_FB_AAPL_AMZN = weighted_return_series_adj.sum(axis=1)

ret_FB_APPL_AMZN = return_series_close_FB_AAPL_AMZN.tail(1) 
print(ret_FB_APPL_AMZN.round(3))

#print("Portfolio Return fb-AMZN:", ret_series_close[-1].round(3))
vol_FB_APPL_AMZN = np.sqrt(252) * np.log((return_series_close_fb_AMZN+1)/(return_series_close_fb_AMZN+1).shift(1)).std()
#print("Portfolio Volatility fb-AMZN:", ret_series_close.round(3))

result = pd.DataFrame({'Portfolio Return APPL-FB-AMZN':ret_FB_APPL_AMZN,'Portfolio Volatility APPL-FB-AMZN':vol_FB_APPL_AMZN})
result = result.reset_index()
result.to_json(path_or_buf= path+"/portfolio_vol_return.json",orient="records")


# In[ ]:


# Not sure of -> 6.1 RAR - Sharpe ratio for portfolio - SPY 50%, AMZN 50%


# In[ ]:


returns_ts
#keep only the first 3 columns of pct.change() from returns_ts for finding portfolio return
returns_ts = returns_ts[['FB', 'AMZN', 'AAPL']] 

# portfolio_weights = [1/3, 1/3, 1/3]
#these weights are for SPY, for SPY-AMZN 50% each use [0.5, 0.5, 0]. Try using [1,0,0] to ge 
portfolio_weights = [0, 0.5 , 0.5]

#apply the weights to returns
wt_portfolio_ret = returns_ts * portfolio_weights 
wt_portfolio_ret.head()

#sum up the weighted returns
portfolio_returns1 = wt_portfolio_ret.sum(axis = 1) 
portfolio_returns1.head()

#export porfolio_return1
portfolio_returns1.to_json(path_or_buf= path+"/portfolio_return1.json",orient="records")

#plot the returns
#ret_ax = portfolio_returns1.plot(title= "Portfolio Returns- AAPL (0.5), SPY (0.5)")
#ret_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


# ### Tear Sheet

# In[ ]:


#Create pyfolio tear sheet
result = pf.create_simple_tear_sheet(portfolio_returns1)

result.to_json(path_or_buf= path+"/tear_sheet.json")


# In[ ]:


#Export Return Series_Close for correlation scatter plot
return_series_close.to_json(path_or_buf= path+"/close_series_correlation.json")


# ### Moving Average (20,50,100)

# In[ ]:


close_price= closes_1y['Close']
close_price


# #### Simple Moving Avg

# In[ ]:


#################################### 
# 6) “moving avg;”
###window= # days;
## (opt;) close_price= xclose_price_1y/ long;
##################################

#.rename(columns={"Symbols": "Tickers", 0: "Annualised Return"})

#20days moving avg
sma_20_close = close_price.rolling(window= 20).mean().reset_index()
sma_20_close = sma_20_close.rename(columns={"FB": "FB_20", "AMZN": "AMZN_20","AAPL": "AAPL_20",})
sma_20_close

#50days moving avg
sma_50_close = close_price.rolling(window= 50).mean().reset_index()
sma_50_close = sma_50_close.rename(columns={"FB": "FB_50", "AMZN": "AMZN_50","AAPL": "AAPL_50",})
sma_50_close

#100days moving avg
sma_100_close = close_price.rolling(window = 100).mean().reset_index()
sma_100_close = sma_100_close.rename(columns={"FB": "FB_100", "AMZN": "AMZN_100","AAPL": "AAPL_100",})
sma_100_close

sma_df = pd.concat([sma_20_close,sma_50_close,sma_100_close])

sma_df.to_json(path_or_buf= path+"/Simple_moving_Average.json",orient="records")


# In[ ]:





# #### Weighted Moving Average

# In[ ]:


weights_20 = np.arange(1,21) 

wma_20_close = close_price.rolling(20).apply(lambda prices: np.dot(prices, weights_20)/weights_20.sum(), raw=True)
wma_20_close = wma_20_close.rename(columns={"FB": "FB_20", "AMZN": "AMZN_20","AAPL": "AAPL_20",})
wma_20_close

weights_50 = np.arange(1,51) 
wma_50_close = close_price.rolling(50).apply(lambda prices: np.dot(prices, weights_50)/weights_50.sum(), raw=True)
wma_50_close = wma_50_close.rename(columns={"FB": "FB_50", "AMZN": "AMZN_50","AAPL": "AAPL_50",})
wma_50_close

weights_100 = np.arange(1,101) 
wma_100_close = close_price.rolling(100).apply(lambda prices: np.dot(prices, weights_100)/weights_100.sum(), raw=True)
wma_100_close = wma_100_close.rename(columns={"FB": "FB_100", "AMZN": "AMZN_100","AAPL": "AAPL_100",})
wma_100_close

wma_df = pd.concat([wma_20_close,wma_50_close,wma_100_close])
wma_df.to_json(path_or_buf= path+"/Weighted_moving_Average.json",orient="records")


# #### Exponential Moving Average

# In[ ]:


# 8) ema**

### 20days ###
# derived from wma formula;
sma20 = close_price.rolling(20).mean()

# p2
modPrice20 = close_price.copy()
modPrice20.iloc[0:20] = sma20[0:20]

# If we want to emulate the EMA as in our spreadsheet using our modified price series, we don’t need this adjustment. We then set adjust=False:
ema_20_close = modPrice20.ewm(span= 20, adjust= False).mean()
ema_20_close = ema_20_close.rename(columns={"FB": "FB_20", "AMZN": "AMZN_20","AAPL": "AAPL_20",})
ema_20_close

### 50days ###
# derived from wma formula;
sma50 = close_price.rolling(50).mean()

# p2
modPrice50 = close_price.copy()
modPrice50.iloc[0:50] = sma50[0:50]

ema_50_close = modPrice50.ewm(span= 50, adjust= False).mean()
ema_50_close = ema_50_close.rename(columns={"FB": "FB_50", "AMZN": "AMZN_50","AAPL": "AAPL_50",})
ema_50_close

### 100days ###
# derived from wma formula;
sma100 = close_price.rolling(100).mean()

# p2
modPrice100 = close_price.copy()
modPrice100.iloc[0:100] = sma100[0:100]

ema_100_close = modPrice100.ewm(span= 100, adjust= False).mean()
ema_100_close = ema_100_close.rename(columns={"FB": "FB_100", "AMZN": "AMZN_100","AAPL": "AAPL_100",})
ema_100_close

ema_df = pd.concat([ema_20_close,ema_50_close,ema_100_close])
ema_df.to_json(path_or_buf= path+"/Exponential_moving_Average.json",orient="records")


# ## Fundamental Data

# In[ ]:


# Import packages
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data
from matplotlib.ticker import FuncFormatter

# render the figures in this notebook 
get_ipython().run_line_magic('matplotlib', 'inline')

import yfinance as yf # to get basic share info
import yahoo_fin.stock_info as si # get fundamentals data


# In[ ]:


# Helper functions

def dict_to_df(dictionary: Dict[str, Any], columns: List[str]=["Key", "Value"]) -> pd.core.frame.DataFrame:
    """
    Converts a vertical python dictionary into a 2D pandas Dataframe, with 2 columns.
    Each key corresponds to a row in the dataframe.
    
    Usage:
    df = dict_to_df(price_dict)
    """
    data = {columns[0]: list(dictionary.keys()), columns[1]: list(dictionary.values())}
    df = pd.DataFrame.from_dict(data)
    return df


# ### Tickers here

# In[ ]:


# Input tickers here
tickers = ['AMZN', 'AAPL', 'FB']


# In[ ]:


start_date = '2020-01-01'
end_date = '2020-12-31'

panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

# Create yf ticker object for each ticker input
Tickers = list(map(lambda t: yf.Ticker(t), tickers))
Tickers


# ### Retrieve Company Basic Information 

# In[ ]:


# Get basic company info (keys:shortName,industry,sector,symbol,website,longBusinessSummary )

#all = pd.DataFrame(columns=['industry'])

column_names = ['company_name','industry','sector','symbol','website','summary']
df = pd.DataFrame(columns = column_names)

row_data = []

#return empty string if no value found
xstr = lambda s: s or ""

for ticker in Tickers:
    ticker_dict = ticker.info
    company_name = xstr(ticker_dict['shortName'])
    industry = xstr(ticker_dict['industry'])
    sector = xstr(ticker_dict['sector'])
    symbol = xstr(ticker_dict['symbol'])
    website = xstr(ticker_dict['website'])
    summary = xstr(ticker_dict['longBusinessSummary'])

    new_row = {'company_name': company_name, 'industry': industry, 'sector': sector, 'symbol': symbol,'website': website,'summary':summary}
    df = df.append(new_row,ignore_index=True)
    
#df.head()

df.to_json(path_or_buf= path+"/company_basic_info.json",orient="records")


# ### Company CashFlow

# In[ ]:


all = []

for ticker in Tickers:
    #convert multi-index to single index with index date shift as date column
    ticker_df = ticker.cashflow.T.rename_axis('Date').reset_index()
    symbol = ticker.info['symbol']
    
    #include symbol column
    ticker_df['symbols'] = symbol
    
    all.append(ticker_df)

result_df = pd.concat(all)

result_df.to_json(path_or_buf= path+"/company_cashflow.json",orient="records")


# ### Company StockSplit

# In[ ]:


ticker_list = []

for ticker in Tickers:
    ticker_split_df = ticker.splits.to_frame().reset_index()
    symbol = ticker.info['symbol']
    
    #include symbol column
    ticker_split_df['symbols'] = symbol


    
    ticker_list.append(ticker_split_df)

split_df = pd.concat(ticker_list)
    
split_df.to_json(path_or_buf= path+"/company_stocksplit.json",orient="records")
    


# ### Quotes

# In[ ]:


ticker_quote_list = []

for ticker in tickers:

    quote = si.get_quote_table(ticker)
    dict_to_df = pd.DataFrame([quote]) 
    
    #include symbol column
    dict_to_df['symbols'] = ticker

    ticker_quote_list.append(dict_to_df)
    
quote_df = pd.concat(ticker_quote_list)
    
split_df.to_json(path_or_buf= path+"/company_quote.json",orient="records")


# ### Financial Statement

# In[ ]:


ticker_finstat_list = []
for ticker in tickers:
    try:
        financials = si.get_financials(ticker)
        
        for name, item in financials.items():
            #print(f"{name} for {ticker}")
            fin_stat_df = item.T
            
            #include symbol column
            fin_stat_df['symbols'] = ticker
            
            ticker_finstat_list.append(fin_stat_df)

    except KeyError:
        print(f"Unable to get financial statements for {ticker}.")


fin_df = pd.concat(ticker_finstat_list)
split_df.to_json(path_or_buf= path+"/company_financial_statement.json",orient="records")


# ### Technical Stats (issue with transforming data)

# In[ ]:


ticker_techstat_list = []

for ticker in tickers:
    stats = si.get_stats(ticker)
    new_stat = stats.reset_index().iloc[:,1:]
    new_stat = new_stat.T
    new_stat.columns = new_stat.iloc[0,:]
    print(new_stat)
    #stats_df = stats.reset_index().T
    #stats_df.columns = stats_df.iloc[0,:]
    #stats_df = stats_df.iloc[1:,:]
    
    #Set first row as column header
    #stats_df.columns = stats_df.iloc[1]
    
    #include symbol column
    #stats_df['symbols'] = ticker
    #print(stats_df.iloc[:,:4])
    #print(stats_df.iloc[1:,:3] )
    break
    
    #ticker_techstat_list.append(stats_df)
    
#tech_stat_df = pd.concat(ticker_techstat_list) 
#tech_stat_df.head()
#tech_stat_df.to_json(path_or_buf= path+"/company_tech_stats.json",orient="records")


# ### Earnings data

# In[ ]:


tickers_dfs = []
#conso_dfs(ticker) -> [df1,df2,df3]
#tickers_dfs -> [[df1,df2,df3],[]]

for ticker in tickers:
    conso_dfs = []
    earnings = si.get_earnings(ticker)
    print(f"Earnings for {ticker}")
    for name, item in earnings.items():
        #print(f"{name} for {ticker}")
        item['symbols'] = ticker
        
        conso_dfs.append(item)
    tickers_dfs.append(conso_dfs)
    
        #display(item)
print(tickers_dfs[0])


# In[ ]:


file_names = ['Earnings_Actual_Estimate','Earnings_Year','Earnings_Year_Quarterly']
all = []
index = 0
for ticker1,ticker2,ticker3 in zip(*tickers_dfs):
    all.append(ticker1)
    all.append(ticker2)
    all.append(ticker3)
    result = pd.concat(all)
    result.to_json(path_or_buf= path+"/"+ file_names[index] +".json",orient="records")
    index += 1
    


# ### Sector Performance

# In[ ]:


#!pip install plotly
#!pip install alpha_vantage


# In[7]:


from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.fundamentaldata import FundamentalData
import matplotlib.pyplot as plt


# In[3]:


api_key = "2P5DGRRQM0DPPDYN"


# In[4]:


sp = SectorPerformances(key=api_key, output_format='pandas')
data, meta_data = sp.get_sector()
data.describe()


# In[5]:


meta_data


# In[6]:


data['Rank A: Real-Time Performance'].plot(kind='bar')
plt.title('Real Time Performance (%) per Sector')
plt.tight_layout()
plt.grid()
plt.show()


# In[ ]:




