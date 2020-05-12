import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import date, timedelta
from portfolioanalytics.portfolio_return import Portfolio

# download google stock historical data
start_date = date(2019, 12, 31)
end_date = date.today() - timedelta(days=1)

# WEIGHTS AND RETURNS FRAMEWORK
ids = ["RACE", "TSLA", "AAPL"]

prices = web.DataReader(ids, data_source='yahoo', start=start_date, end=end_date)
prices = prices["Adj Close"]

prices.index = [d.date() for d in prices.index]

# items = "P_PRICE"
# df = get_stock_historical_prices(ids, items, dcast=True)
prices.head()
prices.tail()

weights = {date(2019, 12, 31): [.5, .3, .2],
           date(2020, 1, 31): [.3, .5, .2],
           date(2020, 2, 28): [.2, .3, .5],
           date(2020, 3, 31): [.2, .7, .1],
           date(2020, 4, 30): [.3, .1, .6]}
weights = pd.DataFrame(weights)
weights = weights.transpose()
weights.columns = ids

# subset prices to match weights.columns
# prices = prices[weights.columns.values.tolist()]


ptf = Portfolio(prices)
ptf.compute_returns().apply(lambda x: np.cumprod(1 + x)).plot()
ptf.portfolio_returns(weights=weights)

ptf_ret, ptf_ts = ptf.portfolio_returns(weights=weights, verbose=False)
ptf_ret2, ptf_ts2, contrib,  V, V_bop = ptf.portfolio_returns(weights=weights, verbose=True)

ptf_ret.equals(ptf_ret2)

# check
ew_ret, ew_ts = ptf.portfolio_returns()
overall_ret = ew_ts.tail(1).values[0] / ew_ts.head(1).values[0] - 1

w = [1 / 3, 1 / 3, 1 / 3]
#cumret = prices.tail(1).values[0] / prices.head(1).values[0] - 1
ratios = prices.tail(1).values[0] / prices.head(1).values[0]  # VT/V0
tot_ret = sum(w * ratios) - 1  # VT/V0 - 1, since VT = 1/3 (V1/V01 + V2/V02 + V3/V03)
# devono corrispondere
tot_ret - overall_ret

# devono corrispondere
# sum(w * cumret) - overall_ret

