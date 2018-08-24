from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ccxt
import pandas as pd
import numpy as np
import xarray as xr
#import sqlalchemy
from datetime import datetime, timedelta
#from mercurius.strategy import olmar


class CandleReader(object):
    """A wrapper for ccxt.

    1) Convert OHLCV to pandas dataframe
    2) Save dataframe to database (sqlite, sql, mongodb, etc.)
    """

    def __init__(self, symbols, start=(datetime.utcnow()-timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S"), end=(datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"), timeframe='15m', exchange='poloniex'):
        """
        Args:
            symobols: a list of symbol of target assets
            start: (str) UTC start time
            end: (str) UTC end time
            timeframe (str): the interval getting data
            exchange (str): the exchange for downloading data
        """
        super(CandleReader, self).__init__()
        self.exchange = getattr(ccxt, exchange)()
        #print("start: ", start)
        #print("end: ", end)
        if not isinstance(start, int):
            self.start = self.exchange.parse8601(start)
        else:
            self.start = start
        #print("start: ", self.start)
        if not isinstance(end, int):
            self.end = self.exchange.parse8601(end)
        else:
            self.end = end
        self.timeframe = timeframe
        self.symbols = symbols

    def _to_df(self, ohlcv):
        #index = pd.Series(np.arange(start, end, interval), name='openTime')
        # init_chart = pd.DataFrame(np.nan, index=index, columns
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['openTime'] = pd.to_datetime(df.timestamp / 1e3, unit='s')
        df.index = pd.DatetimeIndex(df['openTime']).round('S')
        df = df.drop(columns=['openTime', 'timestamp'])
        return df

    def get_data(self):
        raw_data_dict = {}
        if not isinstance(self.end, int) or not isinstance(self.start, int):
            raise TypeError("Instance Time Type needs to be int")

        if self.end > self.start:
            num_candles = (self.end-self.start) / \
                self.exchange.parse_timeframe(self.timeframe)/1e3
            #print("start time: ", self.start)
            #print("end time: ", self.end)
            #print("num of candles: ", num_candles)
        else:
            raise ValueError('end_date should be larger than start_data')

        #candle_list = []
        for symbol in self.symbols:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, self.timeframe, self.start, num_candles)
            #ohlcv = np.array(ohlcv)

            #features = ['open', 'high', 'low', 'close', 'volume']
            #timestamps = ohlcv[:,0]

            #print(ohlcv)
            #print(ohlcv.shape)
            #exit()
            raw_data_dict[symbol] = xr.DataArray(self._to_df(ohlcv), dims=['openTime', 'feature'])
            #arr = xr.DataArray(ohlcv, dims=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            #arr = self._to_df(ohlcv)
            #candle_list.append((symbol, arr))

        #chart = xr.concat(candle_list, dim=['asset', 'openTime', 'feature'])
        #chart = pd.MultiIndex.from_array(candle_list)
        chart = xr.Dataset(raw_data_dict).to_array(dim='asset')

        #chart = xr.DataArray(pd.Panel(raw_data_dict, dtype=np.float32), dims=[
        #                     "asset", "openTime", "feature"])
        #variables = {k: xr.DataArray()}
        #print(chart)
        #print(type(chart))
        return chart.transpose("feature", "asset", "openTime")

    def get_close(self, normalize=True, norm_rel=True):
        da = self.get_data()
        close_price = da.sel(feature='close')
        re = close_price.data.T
        btc_data = np.ones((re.shape[0],1))
        re = np.hstack((btc_data, re))

        if norm_rel:
            re = re[1:] / re[:-1]

        if normalize:
            re = re / re[0]

        return re

    def save_to_sql(self, df):
        df.to_csv('')
        pass
        #engine = create_engine("mysql+mysqldb://...")
        #df.to_sql('candlestick', engine, if_exists='replace', index=False, chunksize=10000)

