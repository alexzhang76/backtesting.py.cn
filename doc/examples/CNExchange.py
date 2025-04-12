# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# _Backtesting.py_ Quick Start User Guide
# =======================
#
# This tutorial shows some of the features of *backtesting.py*, a Python framework for [backtesting](https://www.investopedia.com/terms/b/backtesting.asp) trading strategies.
#
# _Backtesting.py_ is a small and lightweight, blazing fast backtesting framework that uses state-of-the-art Python structures and procedures (Python 3.6+, Pandas, NumPy, Bokeh). It has a very small and simple API that is easy to remember and quickly shape towards meaningful results. The library _doesn't_ really support stock picking or trading strategies that rely on arbitrage or multi-asset portfolio rebalancing; instead, it works with an individual tradeable asset at a time and is best suited for optimizing position entrance and exit signal strategies, decisions upon values of technical indicators, and it's also a versatile interactive trade visualization and statistics tool.
#
#
# ## Data
#
# _You bring your own data._ Backtesting ingests _all kinds of 
# [OHLC](https://en.wikipedia.org/wiki/Open-high-low-close_chart)
# data_ (stocks, forex, futures, crypto, ...) as a
# [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/10min.html)
# with columns `'Open'`, `'High'`, `'Low'`, `'Close'` and (optionally) `'Volume'`.
# Such data is widely obtainable, e.g. with packages:
# * [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/),
# * [Quandl](https://www.quandl.com/tools/python),
# * [findatapy](https://github.com/cuemacro/findatapy),
# * [yFinance](https://github.com/ranaroussi/yfinance),
# * [investpy](https://investpy.readthedocs.io/),
#   etc.
#
# Besides these columns, **your data frames can have additional columns which are accessible in your strategies in a similar manner**.
#
# DataFrame should ideally be indexed with a _datetime index_ (convert it with [`pd.to_datetime()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html));
# otherwise a simple range index will do.

# +
# Example OHLC daily data for Google Inc.
from backtesting.test import GOOG

GOOG.tail()
# -

# ## Strategy
#
# Let's create our first strategy to backtest on these Google data, a simple [moving average (MA) cross-over strategy](https://en.wikipedia.org/wiki/Moving_average_crossover).
#
# _Backtesting.py_ doesn't ship its own set of _technical analysis indicators_. Users favoring TA should probably refer to functions from proven indicator libraries, such as
# [TA-Lib](https://github.com/TA-Lib/ta-lib-python) or
# [Tulipy](https://tulipindicators.org),
# but for this example, we can define a simple helper moving average function ourselves:

# +
import pandas as pd


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


# -

# A new strategy needs to extend 
# [`Strategy`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy)
# class and override its two abstract methods:
# [`init()`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.init) and
# [`next()`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.next).
#
# Method `init()` is invoked before the strategy is run. Within it, one ideally precomputes in efficient, vectorized manner whatever indicators and signals the strategy depends on.
#
# Method `next()` is then iteratively called by the
# [`Backtest`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest)
# instance, once for each data point (data frame row), simulating the incremental availability of each new full candlestick bar.
#
# Note, _backtesting.py_ cannot make decisions / trades _within_ candlesticks — any new orders are executed on the next candle's _open_ (or the current candle's _close_ if
# [`trade_on_close=True`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest.__init__)).
# If you find yourself wishing to trade within candlesticks (e.g. daytrading), you instead need to begin with more fine-grained (e.g. hourly) data.

# +
from backtesting import Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):

        # if self.position:
        #     print(f"position size: {self.position.size}")
            
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            # debug 
            #print("Available cash:", self.available_cash)
            #print("Available cash leveraged:", self.available_cash_leveraged)

            size = 400
            #print(self.data.index[-1].date(), "buy size:", size)
            order_id = self.buy_ex(size=size)
            if not order_id:
                print("Not enough margin to place estimated minimal order")
            #else:
            #    print(f"order size: {order_id.size}")

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            if self.position:
                #print(self.data.index[-1].date(), "sell size:", self.position.size)
                self.position.close_ex(self.position.size, self.data.index[-1].date())


# -

# In `init()` as well as in `next()`, the data the strategy is simulated on is available as an instance variable
# [`self.data`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.data).
#
# In `init()`, we declare and **compute indicators indirectly by wrapping them in 
# [`self.I()`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.I)**.
# The wrapper is passed a function (our `SMA` function) along with any arguments to call it with (our _close_ values and the MA lag). Indicators wrapped in this way will be automatically plotted, and their legend strings will be intelligently inferred.
#
# In `next()`, we simply check if the faster moving average just crossed over the slower one. If it did and upwards, we close the possible short position and go long; if it did and downwards, we close the open long position and go short. Note, we don't adjust order size, so _Backtesting.py_ assumes _maximal possible position_. We use
# [`backtesting.lib.crossover()`](https://kernc.github.io/backtesting.py/doc/backtesting/lib.html#backtesting.lib.crossover)
# function instead of writing more obscure and confusing conditions, such as:

# + magic_args="echo" language="script"
#
#     def next(self):
#         if (self.sma1[-2] < self.sma2[-2] and
#                 self.sma1[-1] > self.sma2[-1]):
#             self.position.close()
#             self.buy()
#
#         elif (self.sma1[-2] > self.sma2[-2] and    # Ugh!
#               self.sma1[-1] < self.sma2[-1]):
#             self.position.close()
#             self.sell()
# -

# In `init()`, the whole series of points was available, whereas **in `next()`, the length of `self.data` and all declared indicators is adjusted** on each `next()` call so that `array[-1]` (e.g. `self.data.Close[-1]` or `self.sma1[-1]`) always contains the most recent value, `array[-2]` the previous value, etc. (ordinary Python indexing of ascending-sorted 1D arrays).
#
# **Note**: `self.data` and any indicators wrapped with `self.I` (e.g. `self.sma1`) are NumPy arrays for performance reasons. If you prefer pandas Series or DataFrame objects, use `Strategy.data.<column>.s` or `Strategy.data.df` accessors respectively. You could also construct the series manually, e.g. `pd.Series(self.data.Close, index=self.data.index)`.
#
# We might avoid `self.position.close()` calls if we primed the
# [`Backtest`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest)
# instance with `Backtest(..., exclusive_orders=True)`.

# ## Backtesting
#
# Let's see how our strategy performs on historical Google data. The
# [`Backtest`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest)
# instance is initialized with OHLC data and a strategy _class_ (see API reference for additional options), and we begin with 10,000 units of cash and set broker's commission to realistic 0.2%.

# +
from backtesting import Backtest

bt = Backtest(GOOG, SmaCross, cash=100_000, commission=.002)
stats = bt.run()
print(stats)

# The columns should be self-explanatory.
print(stats['_trades'])  # Contains individual trade data

# [`Backtest.run()`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest.run)
# method returns a pandas Series of simulation results and statistics associated with our strategy. We see that this simple strategy makes almost 600% return in the period of 9 years, with maximum drawdown 33%, and with longest drawdown period spanning almost two years ...
#
# [`Backtest.plot()`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest.plot)
# method provides the same insights in a more visual form.

bt.plot()
