import os
import time
from datetime import datetime
import numba
import numpy as np
import pandas as pd
import functools
from concurrent.futures import ThreadPoolExecutor

from src.common.chrono.chrono import DateRangeIterator, convert_dates_to_timestamps, drop_hours_from_datetime_object
from src.common.queries.funding_queries import funding_values
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.equinox_api_call.api_calls import DatalinkCreateBands, format_band_values, FundingOptionsBandCreation
from src.common.constants.constants import client1
from src.common.utils.utils import Util, itertools_flatten, spread_entry_func, spread_exit_func


class PriceResult:
    def __init__(self, prices, t0, t1):
        self.prices = prices
        if len(self.prices) == 0:
            self.int_timestamps = []
        else:
            self.int_timestamps = np.array((self.prices.index.view(np.int64) + 1) / 1000000).astype(np.int64)
            self.prices['time'] = self.prices.index
            self.prices['timems'] = np.array((self.prices.index.view(np.int64) + 1) / 1000000).astype(np.int64)
        self.t0 = t0
        self.t1 = t1

    def has_data(self, t0, t1):
        if len(self) == 0:
            return False
        t0_ix = self.int_timestamps.searchsorted(t0, side='left')
        if t0_ix == len(self.int_timestamps):
            return False
        return self.int_timestamps[t0_ix] <= t1 and self.int_timestamps[t0_ix] >= t0

    def get_data(self, t0, t1):
        if not self.has_data(t0, t1):
            return None

        index_update_start = np.searchsorted(self.int_timestamps, t0, side='right')
        index_update_end = np.searchsorted(self.int_timestamps, t1, side='right')
        return self.prices.iloc[index_update_start:index_update_end]

    def __len__(self):
        return len(self.prices)


class DataQueryBase:
    def __init__(self, client):
        self.client = client
        self.influx_connection = InfluxConnection.getInstance()
        self.query = None

    def query_data(self, t0, t1):
        if self.query is None:
            raise Exception("No query set")
        result = self.influx_connection.query_influx(self.client, self.query(t0, t1))
        return Util.create_array_from_query(result)

    @staticmethod
    def fill_nans(array):
        mask = np.isnan(array)
        idx = np.where(~mask, np.arange(mask.shape[0]), -mask.shape[0] + 1)
        idx = np.maximum.accumulate(idx, axis=0)
        out = array[idx]
        return out


class Prices(DataQueryBase):
    def __init__(self, client, exchange, symbol, side=None):
        super().__init__(client)
        self.exchange = exchange
        self.symbol = symbol
        where_clause = f'''exchange = '{self.exchange}' AND symbol = '{self.symbol}' '''
        if side is not None:
            where_clause += f'''AND side = '{side}' '''
        self.query = lambda t0, t1: f"SELECT price, side FROM price " \
                                    f"WHERE {where_clause}" \
                                    f"AND time >= {t0}ms and time <= {t1}ms"
        self.threadpool_executor = ThreadPoolExecutor(max_workers=2)

    def query_data(self, t0, t1):
        if self.query is None:
            raise Exception("No query set")
        if os.getenv("USE_LOCAL_INFLUXDB_CACHE") != "true":
            result = self.influx_connection.query_influx(self.client, self.query(t0, t1), epoch="ns")
            if len(result) == 0:
                return PriceResult([], t0, t1)
            return PriceResult(result['price'], t0, t1)

        result = self.influx_connection.query_influx(self.influx_connection.local_client_spotswap_dataframe,
                                                     self.query(t0, t1), epoch="ns")
        if len(result) == 0 or t1 - int(result['price'].index.view(np.int64)[-1] / 1000000) > 1000 * 60:
            result = self.influx_connection.query_influx(self.client, self.query(t0, t1), epoch="ns")
            if len(result) == 0:
                return PriceResult([], t0, t1)
            self.threadpool_executor.submit(self.write_to_influx, result)
        return PriceResult(result['price'], t0, t1)

    def write_to_influx(self, result):
        for j in range(0, len(result['price']), 10000):
            points = Util.influx_points_from_dataframe(result['price'][j:  j + 10000], 'price',
                                                       tag_columns=["side"],
                                                       field_columns=["price"],
                                                       time_precision="n")
            self.influx_connection.local_client_spotswap.write_points(points, time_precision='n')


class Takers(DataQueryBase):
    def __init__(self, client, exchanges, symbols):
        super().__init__(client)
        self.exchanges = exchanges
        self.symbols = symbols
        where_query = " OR ".join(
            [f"(exchange = '{exchanges[j]}' AND symbol = '{symbols[j]}')" for j in range(len(exchanges))])
        self.query = lambda t0, t1: f"SELECT price, size, side, exchange, symbol FROM trade WHERE {where_query} AND time >= {t0}ms and time <= {t1}ms"
        self.median = None
        self.threadpool_executor = ThreadPoolExecutor(max_workers=2)


@functools.lru_cache(maxsize=2)
def query_data(self, t0, t1):
    if self.query is None:
        raise Exception("No query set")
    if os.getenv("USE_LOCAL_INFLUXDB_CACHE") != "true":
        result = self.influx_connection.query_influx(self.client, self.query(t0, t1), epoch="ns")
        if len(result) == 0:
            return TakersResult([], t0, t1)
        return TakersResult(result['trade'], t0, t1)

    result = self.influx_connection.query_influx(self.influx_connection.local_client_spotswap_dataframe,
                                                 self.query(t0, t1), epoch="ns")
    if len(result) == 0 or t1 - int(result['trade'].index.view(np.int64)[-1] / 1000000) > 1000 * 60:
        result = self.influx_connection.query_influx(self.client, self.query(t0, t1), epoch="ns")
        if len(result) == 0:
            return TakersResult([], t0, t1)
        self.threadpool_executor.submit(self.write_to_influx, result)
    return TakersResult(result['trade'], t0, t1)


def query_median_traded(self, t0, t1):
    if t1 - t0 < 1000 * 60 * 60 * 24 * 30:
        t0 = t1 - 1000 * 60 * 60 * 24 * 30
    where_query = " OR ".join([f"(exchange = '{self.exchanges[j]}' AND symbol = '{self.symbols[j]}')" for j in
                               range(len(self.exchanges))])
    query = f'''select median(x) as median from (SELECT sum("size") as x FROM "trade" WHERE {where_query} AND time >= {t0}ms and time <= {t1}ms GROUP BY time(1d) fill(null))'''
    result = self.influx_connection.query_influx(self.client, query, epoch="ns")
    self.median = int(result['trade']['median'])
    return int(result['trade']['median'])


def write_to_influx(self, result):
    for j in range(0, len(result['trade']), 10000):
        points = Util.influx_points_from_dataframe(result['trade'][j:  j + 10000], 'trade',
                                                   tag_columns=["exchange", "symbol", "side"],
                                                   field_columns=["price", "size"],
                                                   time_precision="n")
        self.influx_connection.local_client_spotswap.write_points(points, time_precision='n')


class TakersResult:
    def __init__(self, takers, t0, t1):
        self.takers = takers
        self.t0 = t0
        self.t1 = t1
        if len(self.takers) == 0:
            self.int_timestamps = []
        else:
            self.takers['time'] = self.takers.index
            self.takers['timems'] = np.array((self.takers.index.view(np.int64) + 1) / 1000000).astype(np.int64)
            self.int_timestamps = np.array((self.takers.index.view(np.int64) + 1) / 1000000).astype(np.int64)

    def has_data(self, t0, t1):
        if len(self) == 0:
            return False
        t0_ix = self.int_timestamps.searchsorted(t0, side='left')
        return self.int_timestamps[t0_ix] <= t1 and self.int_timestamps[t0_ix] >= t0

    def get_data(self, t0, t1):
        if not self.has_data(t0, t1):
            return None

        index_update_start = np.searchsorted(self.int_timestamps, t0, side='right')
        index_update_end = np.searchsorted(self.int_timestamps, t1, side='right')
        return self.takers.iloc[index_update_start:index_update_end]

    def __len__(self):
        return len(self.takers)


class Funding(DataQueryBase):
    def __init__(self, client, exchange, symbol):
        super().__init__(client)
        self.exchange = exchange
        self.symbol = symbol
        self.query = lambda t0, t1: f"SELECT funding FROM funding WHERE exchange = '{self.exchange}' AND symbol = '{self.symbol}' AND time >= {t0}ms and time <= {t1}ms"

        self.threadpool_executor = ThreadPoolExecutor(max_workers=2)


@functools.lru_cache(maxsize=2)
def query_data(self, t0, t1):
    if self.query is None:
        raise Exception("No query set")
    result = self.influx_connection.query_influx(self.client, self.query(t0, t1), epoch="ns")
    if len(result) == 0:
        return FundingResult([], t0, t1)
    return FundingResult(result['funding'], t0, t1)


class FundingResult:
    def __init__(self, funding, t0, t1):
        self.funding = funding
        self.t0 = t0
        self.t1 = t1
        if len(self.funding) == 0:
            self.int_timestamps = []
        else:
            self.funding['time'] = self.funding.index
            self.funding['timems'] = np.array((self.funding.index.view(np.int64) + 1) / 1000000).astype(np.int64)
            self.int_timestamps = np.array((self.funding.index.view(np.int64) + 1) / 1000000).astype(np.int64)

    def has_data(self, t0, t1):
        if len(self) == 0:
            return False
        t0_ix = self.int_timestamps.searchsorted(t0, side='left')
        return self.int_timestamps[t0_ix] <= t1 and self.int_timestamps[t0_ix] >= t0

    def get_data(self, t0, t1):
        if not self.has_data(t0, t1):
            return None

        index_update_start = np.searchsorted(self.int_timestamps, t0, side='right')
        index_update_end = np.searchsorted(self.int_timestamps, t1, side='right')
        return self.funding.iloc[index_update_start:index_update_end]

    def __len__(self):
        return len(self.funding)


@numba.jit(nopython=True)
def df_numba(df_mat):
    for idx in range(1, len(df_mat) - 1):
        if df_mat[idx, 0] >= df_mat[idx, 2]:
            df_mat[idx, 5] = df_mat[idx - 1, 5] + abs(df_mat[idx, 0] - df_mat[idx, 2]) * df_mat[idx, 4]

        if df_mat[idx, 1] <= df_mat[idx, 3]:
            df_mat[idx, 6] = df_mat[idx - 1, 6] + abs(df_mat[idx, 1] - df_mat[idx, 3]) * df_mat[idx, 4]
    return df_mat


def get_strategy_pos_value(t0, t1, strategy, **kwargs):
    """
    A function inorder to find the current value of a strategy
    Inputs:
    t0 :  starting time
    t1 : ending time
    strategy : strategy name

    Outputs:
    df : a dataframe containing the current value of a strategy

    Args:
        **kwargs:

    """
    connection = InfluxConnection.getInstance()
    result = connection.prod_client_spotswap_dataframe.query(
        f'''SELECT "position" FROM "position_values" WHERE ("strategy" = '{strategy}' AND "source"='REST' AND "instrument"='ETH-PERPETUAL'
                                                           AND "application" = 'Audit') AND time>={t0}ms AND time<={t1}ms''',
        epoch='ns')
    if len(result["position_values"]) > 0:
        return result["position_values"]
    else:
        return print('empty dataframe')


def get_inout_pos(t0, t1, strategy, environment):
    """
    A function to determine the time stamps that we are in position
    Inputs:
    t0 :  starting time
    t1 : ending time
    strategy : strategy name

    Outputs:
    df : a dataframe that contains the time when we entered the position and when we exited the position
    """
    if environment == 'server':
        result = client1.query(
            f'''SELECT "position" FROM "position_values" WHERE ("strategy" = '{strategy}' AND "source"='REST'
            AND("instrument"='ETH-PERPETUAL' OR "instrument"='BTC-PERPETUAL')
                                                               AND "application" = 'Audit') AND time>={t0}ms AND time<={t1}ms''',
            epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "position" FROM "position_values" WHERE ("strategy" = '{strategy}' AND "source"='REST'
                    AND("instrument"='ETH-PERPETUAL' OR "instrument"='BTC-PERPETUAL')
                                                                       AND "application" = 'Audit') AND time>={t0}ms AND time<={t1}ms''',
            epoch='ns')

    if len(result["position_values"]) > 0:
        df = result["position_values"]
        df.reset_index(inplace=True)
        df.rename({"index": "Time"}, axis=1, inplace=True)
        # df.to_csv(f'/home/enea/Data/InPos_{strategies[i]}.csv')
        idx = df[df["position"] < -1].index
        numpy_idx = np.array(idx)
        differences = np.diff(numpy_idx)
        indices_jump_end = np.where(differences != 1)[0]
        indices_jump_start = np.where(differences != 1)[0] + 1
        indices_position_end = idx[indices_jump_end].union([idx[-1]])
        indices_position_start = idx[indices_jump_start].union([idx[0]])
        timestamps_position_start = df.iloc[indices_position_start]['Time']
        timestamps_position_end = df.iloc[indices_position_end]['Time']
        if len(timestamps_position_end) != len(timestamps_position_start):
            return

        return pd.DataFrame({'Start': timestamps_position_start.to_list(), 'End': timestamps_position_end.to_list()})
    else:
        return pd.DataFrame({'Start': pd.NaT, 'End': pd.NaT})


def get_quanto_profit(t0, t1, strategy):
    """
    A function to return the Qunato profit of an ETHUSD strategy

    Inputs:
    t0 : starting time in ms
    t1 : ending time in ms
    strategy :  strategy name

    Outputs:
    df =  dataframe containing the Quanto Profit and the Quanto Profit in Dollars

    """

    query_string = f'''SELECT "normalized_profit" AS "Quanto profit per ETH",
    "normalized_profit" * "position" AS "Dollar profit"
    FROM "mark_to_market_quanto_profit" WHERE ("strategy" = '{strategy}') AND time > {t0}ms AND time < {t1}ms '''
    connection = InfluxConnection.getInstance()
    result = connection.prod_client_spotswap_dataframe.query(query_string, epoch="ns")
    if len(result["mark_to_market_quanto_profit"]) > 0:
        return result["mark_to_market_quanto_profit"]
    else:
        return print('empty dataframe')


def get_price(t_start,
              t_end,
              exchange='BitMEX',
              symbol='XBTUSD',
              side='Ask',
              environment='production',
              split_data=True,
              use_side=True):
    """
    A function to retrieve the current price of the future
    Inputs:
    t0 : starting time
    t1: ending time
    exchange :  the exchange (default : BitMEX)
    symbol :  the symbol of the contract (default : XBTUSD)

    Outputs:
    df : a dataframe containing the price of the future
    """

    day_in_millis = 1000 * 60 * 60 * 6
    if t_end - t_start >= day_in_millis and split_data:
        t_start = t_start
        t_end = t_start + day_in_millis
        while t_end <= t_end:
            if not t_start == t_start:
                df_previous = df
            if t_end - day_in_millis <= t_end <= t_end:
                t_end = t_end
            if use_side:
                query_string = f'''SELECT "price" FROM "price" WHERE ("exchange" = '{exchange}' and symbol = '{symbol}' AND side='{side}') AND time > {t_start}ms AND time < {t_end}ms '''
            else:
                query_string = f'''SELECT "price" , "side" FROM "price" WHERE ("exchange" = '{exchange}' and symbol = '{symbol}') AND time > {t_start}ms AND time < {t_end}ms '''
            if environment == 'server':
                result = client1.query(query_string, epoch="ns")
            elif environment == 'production':
                connection = InfluxConnection.getInstance()
                result = connection.prod_client_spotswap_dataframe.query(query_string, epoch="ns")
            elif environment == 'staging':
                connection = InfluxConnection.getInstance()
                result = connection.staging_client_spotswap_dataframe.query(query_string, epoch="ns")
                attempts = 0
                while attempts < 5:
                    if len(result) > 0:
                        break
                    else:
                        time.sleep(1)
                        result = connection.staging_client_spotswap_dataframe.query(query_string, epoch="ns")
                        attempts += 1
                if attempts != 0:
                    print(f"attempt= {attempts}")
                    print(f"t_start = {pd.to_datetime(t_start, unit='ms')}, t_end = {pd.to_datetime(t_end, unit='ms')}")
            if t_start == t_start:
                try:
                    df = result["price"]
                except:
                    pass
            else:
                try:
                    df = result["price"]
                    df = pd.concat([df_previous, df])
                except:
                    print("no data in the requested period")
            t_start = t_start + day_in_millis
            t_end = t_end + day_in_millis
            time.sleep(1)
    else:

        if use_side:
            query_string = f'''SELECT "price" FROM "price" WHERE ("exchange" = '{exchange}' and symbol = '{symbol}' AND side='{side}') AND time > {t_start}ms AND time < {t_end}ms '''
        else:
            query_string = f'''SELECT "price", "side" FROM "price" WHERE ("exchange" = '{exchange}' and symbol = '{symbol}') AND time > {t_start}ms AND time < {t_end}ms '''
        # print(query_string)
        if environment == 'server':
            result = client1.query(query_string, epoch="ns")
        elif environment == 'production':
            connection = InfluxConnection.getInstance()
            result = connection.prod_client_spotswap_dataframe.query(query_string, epoch="ns")
        elif environment == 'staging':
            connection = InfluxConnection.getInstance()
            result = connection.staging_client_spotswap_dataframe.query(query_string, epoch="ns")
        df = result["price"]

    if len(result) > 0:
        return df
    else:
        return print('empty dataframe')


def get_symbol(t0, t1, exchange, environment):
    query_string = f"""SHOW TAG VALUES   FROM "price"  WITH KEY  IN ( "symbol" ,"exchange") where
    ("exchange" = '{exchange}') AND time > {t0}ms AND time < {t1}ms """
    if environment == 'server':
        result = client1.query(query_string, epoch="ns")
    elif environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap.query(query_string, epoch="ns")
    elif environment == 'staging':
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap.query(query_string, epoch="ns")
    else:
        return None

    rr = np.array([x for x in result])
    list_n = []
    for ix in range(1, np.shape(rr)[1]):
        list_n.append(rr[0, ix]['value'])

    return pd.Series(list_n, name="symbol")


def get_exhange_names(t0, t1, environment):
    query_string = f"""SHOW TAG VALUES   FROM "price"  WITH KEY="exchange" WHERE  time > {t0}ms AND time < {t1}ms  """
    if environment == 'server':
        result = client1.query(query_string, epoch="ns")
    elif environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap.query(query_string, epoch="ns")
    elif environment == 'staging':
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap.query(query_string, epoch="ns")
    else:
        return None

    rr = np.array([x for x in result])
    list_n = []
    for ix in range(0, np.shape(rr)[1]):
        list_n.append(rr[0, ix]['value'])

    return pd.Series(list_n, name="exhange")


def get_strategy_influx(environment='production'):
    query_string = f"""SHOW TAG VALUES   FROM "band"  WITH KEY="strategy"  """
    if environment == 'server':
        result = client1.query(query_string, epoch="ns")
    elif environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap.query(query_string, epoch="ns")
    elif environment == 'staging':
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap.query(query_string, epoch="ns")

    rr = np.array([x for x in result])
    list_n = []
    for ix in range(0, np.shape(rr)[1]):
        list_n.append(rr[0, ix]['value'])

    return pd.Series(list_n, name="strategy")


def get_orderbook_price(t0, t1, exchange='Deribit', spotInstrument='hybrid_BTC-PERPETUAL',
                        swapInstrument='hybrid_BTC-USD',
                        type=0, environment='production'):
    if environment == 'server':
        result = client1.query(f'''SELECT "price_spot",
            "price_swap" FROM "orderbook_update" WHERE ("exchangeName" ='{exchange}'
            AND "spotInstrument" = '{spotInstrument}'
            AND "swapInstrument" ='{swapInstrument}' AND "type" = '{type}')
            AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "price_spot",
        "price_swap" FROM "orderbook_update" WHERE ("exchangeName" ='{exchange}'
        AND "spotInstrument" = '{spotInstrument}'
        AND "swapInstrument" ='{swapInstrument}' AND "type" = '{type}')
        AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')

    return result["orderbook_update"]


def get_entry_opportunity_points(t0, t1, exchange='Deribit', spot='hybrid_ETH-PERPETUAL', swap='hybrid_ETHUSD',
                                 environment='production'):
    """
        A function in order to obtain the entry opportunity points
        Inputs:
        t0 :  starting time in ms
        t1 :  ending time in ms
        exchange : the exchange (default : 'Deribit')
        spot : the spot product name (default: 'hybrid_ETH-PERPETUAL')
        swap : the swap product name (default : 'hybrid_ETHUSD')

        Outputs:
        df= a dataframe containing the 'Entry Opportunity', Entry Opportunity_takers' and 'Entry Opportunity_takers_lat'
        columns. If the last two columns do not exist then it will return only the 'Entry Opportunity'
        """
    if environment == 'server':
        result = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='0')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                               epoch='ns')
        result1 = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='entry_with_takers')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                epoch='ns')
        result2 = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='entry_with_takers_latency_200')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                           WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='0')
                                                                           AND time >= {t0}ms and time <= {t1}ms''',
                                                                 epoch='ns')
        result1 = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                           WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='entry_with_takers')
                                                                           AND time >= {t0}ms and time <= {t1}ms''',
                                                                  epoch='ns')
        result2 = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                           WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='entry_with_takers_latency_200')
                                                                           AND time >= {t0}ms and time <= {t1}ms''',
                                                                  epoch='ns')
    try:
        df = result["trading_opportunities"]
    except:
        df1 = result1["trading_opportunities"]
        df1['Time'] = df1.index
        df1.rename(columns={'opportunity': 'Entry_Opportunity_takers'}, inplace=True)
        return df1
    if len(df) > 0:
        df['Time'] = df.index
        df.rename(columns={'opportunity': 'Entry Opportunity'}, inplace=True)

        df1 = result1["trading_opportunities"]
        df1['Time'] = df1.index
        df1.rename(columns={'opportunity': 'Entry Opportunity_takers'}, inplace=True)

        df2 = result2["trading_opportunities"]
        df2['Time'] = df2.index
        df2.rename(columns={'opportunity': 'Entry Opportunity_takers_lat'}, inplace=True)

        df4 = pd.merge_ordered(df, df1, on='Time')
        df4 = pd.merge_ordered(df4, df2, on='Time')
        return df4[['Time', 'Entry Opportunity', 'Entry Opportunity_takers', 'Entry Opportunity_takers_lat']]
    else:
        return print('empty dataframe')


def get_exit_opportunity_points(t0, t1, exchange='Deribit', spot='hybrid_ETH-PERPETUAL', swap='hybrid_ETHUSD',
                                environment='production'):
    """
    A function in order to obtain the exit opportunity points
    Inputs:
    t0 :  starting time in ms
    t1 :  ending time in ms
    exchange : the exchange (default : 'Deribit')
    spot : the spot product name (default: 'hybrid_ETH-PERPETUAL')
    swap : the swap product name (default : 'hybrid_ETHUSD')

    Outputs:
    df= a dataframe containing the 'Exit Opportunity', Exit Opportunity_takers' and 'Exit Opportunity_takers_lat'
    columns. If the last two columns do not exist then it will return only the 'Exit Opportunity'
    """
    if environment == 'server':
        result = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='1')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                               epoch='ns')
        result1 = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='exit_with_takers')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                epoch='ns')
        result2 = client1.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='exit_with_takers_latency_200')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='1')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                                                 epoch='ns')
        result1 = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='exit_with_takers')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                                                  epoch='ns')
        result2 = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}' AND "type"='exit_with_takers_latency_200')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                                                  epoch='ns')

    try:
        df = result["trading_opportunities"]
    except:
        df1 = result1["trading_opportunities"]
        df1['Time'] = df1.index
        df1.rename(columns={'opportunity': 'Exit_Opportunity_takers'}, inplace=True)
        return df1
    if len(df) > 0:
        df['Time'] = df.index
        df.rename(columns={'opportunity': 'Exit Opportunity'}, inplace=True)

        if len(result1["trading_opportunities"]) > 0:
            df1 = result1["trading_opportunities"]
            df1['Time'] = df1.index
            df1.rename(columns={'opportunity': 'Exit Opportunity_takers'}, inplace=True)

        if len(result2["trading_opportunities"]) > 0:
            df2 = result2["trading_opportunities"]
            df2['Time'] = df2.index
            df2.rename(columns={'opportunity': 'Exit Opportunity_takers_lat'}, inplace=True)

        if (len(result1["trading_opportunities"]) > 0) & (len(result2["trading_opportunities"]) > 0):
            df4 = pd.merge_ordered(df, df1, on='Time')
            df4 = pd.merge_ordered(df4, df2, on='Time')
            return df4[['Time', 'Exit Opportunity', 'Exit Opportunity_takers', 'Exit Opportunity_takers_lat']]
        else:
            return df

    else:
        return print('empty dataframe')


def get_strat_curr_value(t0, t1, strategy, environment='production'):
    """
    A function in order to get the current value of a given strategy.
    Inputs:
    t0 : starting time in ms
    t1 : ending time in ms
    strategy :  the given strategy name

    Outputs:
    df :  a dataframe containing the time stamp 'Time' and the current value of a strategy 'alias'.
    """

    if environment == 'server':
        result = client1.query(f'''SELECT "current_value" AS "strategy_value" ,
                                                                    "spot_funds","swap_funds"
                                                                    FROM "mark_to_market_changes"
                                                                    WHERE ("strategy" = '{strategy}')
                                                                    AND time >= {t0}ms and time <= {t1}ms''',
                               epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "current_value" AS "strategy_value" ,
                                                                            "spot_funds","swap_funds"
                                                                            FROM "mark_to_market_changes"
                                                                            WHERE ("strategy" = '{strategy}')
                                                                            AND time >= {t0}ms and time <= {t1}ms''',
                                                                 epoch='ns')
    df = result["mark_to_market_changes"]
    df['Time'] = df.index
    df = df[['Time', 'strategy_value', 'spot_funds', 'swap_funds']]
    return df


def get_entry_exit_bands(t0, t1, strategy, entry_delta_spread, exit_delta_spread, btype='central_band',
                         environment='production'):
    """
    A function in order to get the central band, the entry band and the exit band of a given strategy
    Inputs:
     t0 : starting time in ms
     t1 : ending time in ms
     strategy :  strategy name
     entry_delta_spread :  entry delta spread for the given strategy
     exit_delta_spread :  exit delta spread for the given strategy

     Outputs:
      df : dataframe with central band, entry band, exit band
    """

    if environment == 'server':
        result = client1.query(f'''SELECT ("exit_window_avg" + "entry_window_avg")/2 AS "Band"
        FROM bogdan_bins_{strategy} WHERE time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    elif environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(
            f'''SELECT ("exit_window_avg" + "entry_window_avg")/2 AS "Band" FROM "bogdan_bins_{strategy}" WHERE time >= {t0}ms and time <= {t1}ms''',
            epoch='ns')
    elif environment == 'staging':
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap_dataframe.query(
            f'''SELECT ("exit_window_avg" + "entry_window_avg")/2 AS "Band" FROM "bogdan_bins_{strategy}" WHERE time >= {t0}ms and time <= {t1}ms''',
            epoch='ns')

    df = result[f'bogdan_bins_{strategy}']
    if len(df) > 0:
        df['Time'] = df.index
        df['Entry Band'] = df['Band'] + entry_delta_spread
        df['Exit Band'] = df['Band'] - exit_delta_spread
        if btype == 'Central Band':
            return df[['Time', 'Band']]
        else:
            return df[['Time', 'Band', 'Entry Band', 'Exit Band']]
    else:
        print('empty dataframe')


def get_executions(t0, t1, strategy: str = None, environment: str = "production", group: bool = False,
                   group_name: str = 'XBTUSD', get_type: bool = False):
    """
    A function in order to get the executions and the volume of the executions in a given time period
    Inputs:
    t0 : starting time in ms
    t1 : ending time in ms
    strategy :  strategy name, exit

    Outputs:
    df : a dataframe containing the 'Time', 'entry_executions', 'exit_executions','entry_volume','exit_volume'
    """

    if environment == "production":
        connection = InfluxConnection.getInstance()
        if group:
            result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "spread", type, volume_executed_spot, 
            strategy  FROM "executed_spread" WHERE ("strategy" =~ /{group_name}*/) AND time >= {t0}ms and time <= {t1}ms ''',
                                                                     epoch='ns')
        else:
            result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "spread", type, volume_executed_spot  FROM "executed_spread"
                        WHERE ("strategy" = '{strategy}') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
    elif environment == 'server':
        result = client1.query(f'''SELECT "spread", type, volume_executed_spot  FROM "executed_spread"
                WHERE ("strategy" = '{strategy}') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
    else:
        return

    if len(result) == 0:
        return print('empty data frame')

    else:
        if group:
            return result["executed_spread"]
        if get_type:
            return result["executed_spread"]
        df = result["executed_spread"]
        df['Time'] = df.index
        df['entry_executions'] = df[df.type == 'entry']['spread']
        df['exit_executions'] = df[df.type == 'exit']['spread']
        df['entry_volume'] = df[df.type == 'entry']['volume_executed_spot']
        df['exit_volume'] = df[df.type == 'exit']['volume_executed_spot']
        df.drop(columns=['spread', 'type', 'volume_executed_spot'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def get_strategy_names(t0, environment='production'):
    """
    function to get the strategy names
    environment : server strategies are stored(default : production)
    """

    if environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SHOW TAG VALUES FROM "band"
        WITH KEY = "strategy" where time > {t0}ms''', epoch='ns')
    elif environment == 'server':
        result = client1.query(f'''SHOW TAG VALUES FROM "band"
               WITH KEY = "strategy" where time > {t0}ms''', epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.aws_client_spotswap_dataframe.query(f'''SHOW TAG VALUES FROM "band"
               WITH KEY = "strategy" where time > {t0}ms''', epoch='ns')

    strategies = result["band"]
    df = pd.DataFrame.from_records(list(strategies), columns=['key', 'value'])
    df.drop(columns=['key'], inplace=True)
    df.rename(columns={'value': 'strategy'}, inplace=True)
    return df


def average_difference(t0, t1, strategy: str, entry_delta_spread: float, exit_delta_spread: float):
    """
    a function to find the weighted average difference of entry/exit executions

    Inputs:
    t0 : starting time
    t1 : ending time
    strategy : strategy name
    entry_delta_spread : entry delta spread of strategy
    exit_delta_spread :  exit delta spread of strategy

    Outputs:
    entry_avg_diff :  entry executions average difference weighted over volume
    exit_avg_diff :  exit executions average difference weighted over volume
    """
    df = get_entry_exit_bands(t0, t1, strategy, entry_delta_spread, exit_delta_spread)
    df1 = get_executions(t0, t1, strategy)
    dff = pd.merge_ordered(df, df1, on='Time')

    # entry weighted average over volume
    dff['diff'] = dff['entry_executions'] - dff['Entry Band'].ffill()
    dff['w_diff'] = dff['diff'] * dff['entry_volume']
    entry_avg_diff = dff['w_diff'].sum() / dff['entry_volume'].sum()

    # exit weighted average over volume
    dff['diff'] = dff['Exit Band'].ffill() - dff['exit_executions']
    dff['w_diff'] = dff['diff'] * dff['exit_volume']
    exit_avg_diff = dff['w_diff'].sum() / dff['exit_volume'].sum()

    return entry_avg_diff, exit_avg_diff


def get_strategy_families(t0, environment='production'):
    """
    A function to get all strategies in production and place them in families.
    Inputs:
    t0 : a starting time
    environment : server strategies are stored(default : production)

    Outputs:
    strategy_families : a dictionary with all strategies in families.
                        dictionary keys : 'deribit_eth','deribit_xbtusd','huobi_xbtusd','huobi_eth'
    """
    strategies = get_strategy_names(t0, environment=environment)
    deribit_eth = itertools_flatten(strategies.
                                    loc[strategies['strategy'].str.
                                    contains('ETHUSD')].
                                    values.tolist())
    deribit_xbtusd = itertools_flatten(strategies.
                                       loc[strategies['strategy'].
                                       str.contains('deribit_XBTUSD_maker_perpetual')].
                                       values.tolist())
    deribit_perp_huobi = strategies.loc[strategies['strategy'].str.contains('deribit_perp_huobi_perp')]
    huobi_xbtusd = itertools_flatten(deribit_perp_huobi.
                                     drop(deribit_perp_huobi.loc[deribit_perp_huobi['strategy'].
                                          str.contains('ETH')].index).values.tolist())
    huobi_eth = itertools_flatten(strategies.loc[strategies['strategy'].
                                  str.contains('deribit_perp_huobi_perp_ETH')].values.tolist())
    if environment == 'production':
        strategy_families = {'deribit_eth': deribit_eth, 'deribit_xbtusd': deribit_xbtusd,
                             'huobi_xbtusd': huobi_xbtusd, 'huobi_eth': huobi_eth}
    elif environment == 'server':
        strategy_families = {'deribit_eth': deribit_eth, 'deribit_xbtusd': deribit_xbtusd,
                             'huobi_xbtusd': huobi_xbtusd, 'huobi_eth': huobi_eth}
    else:
        strategy_families = {'huobi_xbtusd': huobi_xbtusd, 'huobi_eth': huobi_eth}
    return strategy_families


def get_microparams_strategy(t0, t1, strategy, environment='production'):
    """
    a function to get the entry and exit delta spread and the window size for every strategy in production.
    Inputs:
    t0 : starting time
    t1 : ending time
    strategy :  strategy name

    Outputs
    df = results dataframe with delta_spread, window_size and type( entry or exit)
    """
    connection = InfluxConnection.getInstance()
    if environment == "production":
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "window_size","delta_spread" ,"type" FROM
        "execution_quality" WHERE ("strategy" ='{strategy}') AND time >={t0}ms and time <=
        {t1}ms''', epoch='ns')
    elif environment == 'server':
        result = client1.query(f'''SELECT "window_size","delta_spread" ,"type" FROM
               "execution_quality" WHERE ("strategy" ='{strategy}') AND time >={t0}ms and time <=
               {t1}ms''', epoch='ns')
    else:
        return

    if len(result) == 0:
        return print('empty dataframe on entry or exit')

    df = result['execution_quality']
    df['entry_delta_spread'] = df.loc[df.type == 'entry', 'delta_spread']
    df['exit_delta_spread'] = df.loc[df.type == 'exit', 'delta_spread']
    df.drop(columns=['type', 'delta_spread'], inplace=True)

    return df


def get_band_values(t0, t1, strategy, environment='production', typeb='bogdan_bands'):
    connection = InfluxConnection.getInstance()
    if environment == 'production':
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "value","side" FROM "band"
            WHERE ("strategy" ='{strategy}' AND "type" = '{typeb}')
            AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    elif environment == 'server':
        result = client1.query(f'''SELECT "value","side" FROM "band"
                    WHERE ("strategy" ='{strategy}' AND "type" = '{typeb}')
                    AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    elif environment == 'staging':
        result = connection.staging_client_spotswap_dataframe.query(f'''SELECT "value","side" FROM "band"
                    WHERE ("strategy" ='{strategy}' AND "type" = '{typeb}')
                    AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    else:
        return

    if len(result) == 0:
        return

    df = result["band"]
    df['Time'] = df.index
    df['Entry Band'] = df.loc[df['side'] == 'entry', 'value']
    df['Exit Band'] = df.loc[df['side'] == 'exit', 'value']
    df.drop(columns=['side', 'value'], inplace=True)
    return result["band"]


def get_execution_quality(t0, t1, strategy, environment='production'):
    """
    function to get the difference between an execution and the band values
    Inputs:
    t0 : start time
    t1 : end time
    strategy :  strategy name
    environment :  the server that contains the values of

    Output
    df :  dataframe with the entry execution quality and exit execution quality
    """
    connection = InfluxConnection.getInstance()
    if environment == 'production':
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "diff_band","volume","type" FROM "execution_quality"
               WHERE( "strategy" ='{strategy}')AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    elif environment == 'server':
        result = client1.query(f'''SELECT "diff_band","volume","type" FROM "execution_quality"
                       WHERE( "strategy" ='{strategy}')AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    else:
        return

    if len(result) == 0:
        return

    df = result["execution_quality"]
    df['Time'] = df.index
    df['entry_exec_q'] = df.loc[df['type'] == 'entry', 'diff_band']
    df['exit_exec_q'] = df.loc[df['type'] == 'exit', 'diff_band']
    df['entry_exec_v'] = df.loc[df['type'] == 'entry', 'volume']
    df['exit_exec_v'] = df.loc[df['type'] == 'exit', 'volume']
    df.drop(columns=['diff_band', 'volume', 'type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_http_latency(t0, t1, strategy, environment='production', path='huobi/swap_cancel'):
    if environment == 'server':
        result = client1.query(f'''SELECT "latency" FROM "http_latency"
        WHERE ("strategy" = '{strategy}' AND "path" = '{path}')
        AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "latency" FROM "http_latency"
                WHERE ("strategy" = '{strategy}' AND "path" = '{path}')
                AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')

    df = result["http_latency"]
    df['Time'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df[['Time', 'latency']]


def get_percentage_band_values(t0, t1, lookback, recomputation_time, target_percentage_exit,
                               target_percentage_entry, entry_opportunity_source, exit_opportunity_source,
                               spot_name, spot_instrument, swap_instrument, environment='production'):
    connection = InfluxConnection.getInstance()
    q = f'''SELECT "value" as entry_band
               FROM "percentage_band" WHERE ("exchangeName" = '{spot_name}' AND "spotInstrument" = '{spot_instrument}'
               AND "swapInstrument" = '{swap_instrument}' AND "lookback" = '{lookback}'
               AND "recomputation_time" = '{recomputation_time}' AND "target_percentage" = '{target_percentage_entry:g}'
               AND "side"='entry' AND "entryOpportunitySource" = '{entry_opportunity_source}'
               AND "exitOpportunitySource" = '{exit_opportunity_source}')
               AND time >= {t0}ms and time <= {t1}ms'''

    q1 = f'''SELECT "value" as exit_band
                    FROM "percentage_band" WHERE ("exchangeName" = '{spot_name}' AND "spotInstrument" = '{spot_instrument}'
                    AND "swapInstrument" = '{swap_instrument}' AND "lookback" = '{lookback}'
                    AND "recomputation_time" = '{recomputation_time}' AND "target_percentage" = '{target_percentage_exit:g}'
                    AND "side"='exit' AND "entryOpportunitySource" = '{entry_opportunity_source}'
                    AND "exitOpportunitySource" = '{exit_opportunity_source}')
                    AND time >= {t0}ms and time <= {t1}ms'''

    if environment == 'server':
        result = client1.query(q, epoch='ns')

        result2 = client1.query(q1, epoch='ns')
    else:
        result = connection.prod_client_spotswap_dataframe.query(q, epoch='ns')

        result2 = connection.prod_client_spotswap_dataframe.query(q1, epoch='ns')

    df1 = result["percentage_band"]
    df1['Time'] = df1.index
    df2 = result2["percentage_band"]
    df2['Time'] = df2.index
    df = pd.merge_ordered(df1, df2, on='Time')
    df.reset_index(drop=True, inplace=True)
    return df[['Time', 'entry_band', 'exit_band']]


def get_quanto_profit_without_zeros(t0, t1, strategy, environment='production'):
    if environment == 'server':
        result = client1.query(f'''SELECT mean("normalized_profit") as "Quanto profit per ETH"
        FROM "mark_to_market_quanto_profit" WHERE("strategy" = '{strategy}' AND
        "normalized_profit" > 0) AND time >= {t0}ms and time <= {t1}ms GROUP BY time(30s) fill(0)''', epoch='ns')
    else:
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(
            f'''SELECT mean("normalized_profit") as "Quanto profit per ETH"
            FROM "mark_to_market_quanto_profit" WHERE("strategy" = '{strategy}' AND
            "normalized_profit" > 0) AND time >= {t0}ms and time <= {t1}ms GROUP BY time(30s) fill(0)''',
            epoch='ns')

    df = result["mark_to_market_quanto_profit"]
    df['Time'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df[['Time', 'Quanto profit per ETH']]


def get_opportunity_points_all(t0, t1, exchange='Deribit', spot='hybrid_ETH-PERPETUAL', swap='hybrid_ETHUSD',
                               environment='production'):
    """
        A function in order to obtain the exit opportunity points
        Inputs:
        t0 :  starting time in ms
        t1 :  ending time in ms
        exchange : the exchange (default : 'Deribit')
        spot : the spot product name (default: 'hybrid_ETH-PERPETUAL')
        swap : the swap product name (default : 'hybrid_ETHUSD')

        Outputs:
        df= a dataframe containing the 'Entry Opportunity', Entry Opportunity_takers' and 'Entry Opportunity_takers_lat'
        columns. If the last two columns do not exist then it will return only the 'Entry Opportunity'
        """
    if environment == 'server':
        result = client1.query(f'''SELECT "opportunity" , "type" FROM "trading_opportunities"
                                                                        WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}')
                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                               epoch='ns')

    elif environment == 'production':
        connection = InfluxConnection.getInstance()
        result = connection.prod_client_spotswap_dataframe.query(f'''SELECT "opportunity", "type" FROM "trading_opportunities"
                                                                           WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}')
                                                                           AND time >= {t0}ms and time <= {t1}ms''',
                                                                 epoch='ns')
    elif environment == 'staging':
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap_dataframe.query(f'''SELECT "opportunity", "type" FROM "trading_opportunities"
                                                                                   WHERE ("exchangeName" = '{exchange}' AND "spotInstrument" = '{spot}' AND "swapInstrument" = '{swap}')
                                                                                   AND time >= {t0}ms and time <= {t1}ms''',
                                                                    epoch='ns')
    else:
        return

    return result["trading_opportunities"]


def get_avg_fixed_spreads(t0, t1, family="deribit_xbtusd"):
    connection = InfluxConnection.getInstance()
    if family == "deribit_xbtusd":
        base_query = f'''SELECT last(sum_vol_entry) as vol_entry, last(sum_vol_exit) as vol_exit, last(f_spread) as fixed_spread , min(sum_vol)*last(f_spread) as min_vol_mult_fixed_spread
                        From (SELECT sum(v)/sum(x) -  sum(v_x)/sum(x_x) as f_spread   FROM 
                        ( SELECT "spread" * "volume_executed_spot" as v, "volume_executed_spot" as x FROM "executed_spread" WHERE  "strategy" =~ /XBTUSD/  AND "type"='entry'),
                        ( SELECT "spread" * "volume_executed_spot" as v_x, "volume_executed_spot" as x_x FROM "executed_spread" WHERE "strategy" =~ /XBTUSD/  AND "type"='exit' ) ),
                        (SELECT mean("spread") as m_spread_entry , sum(volume_executed_spot) as sum_vol_entry FROM "executed_spread" WHERE  type='entry' ),
                        (SELECT sum(volume_executed_spot) as sum_vol_exit FROM "executed_spread" WHERE  type='exit' ),
                        (SELECT sum(volume_executed_spot) as sum_vol FROM "executed_spread"  GROUP BY  "type")
                        WHERE "strategy" =~ /XBTUSD/  AND time >= {t0}ms and time <= {t1}ms GROUP BY "strategy" '''
        result = connection.prod_client_spotswap_dataframe.query(base_query, epoch='ns')
    elif family == "deribit_eth":
        base_query = f'''SELECT last(sum_vol_entry) as vol_entry, last(sum_vol_exit) as vol_exit, last(f_spread) as fixed_spread , min(sum_vol)*last(f_spread) as min_vol_mult_fixed_spread
                        From (SELECT sum(v)/sum(x) -  sum(v_x)/sum(x_x) as f_spread   FROM 
                        ( SELECT "spread" * "volume_executed_spot" as v, "volume_executed_spot" as x FROM "executed_spread" WHERE  "strategy" =~ /ETHUSD/  AND "type"='entry'),
                        ( SELECT "spread" * "volume_executed_spot" as v_x, "volume_executed_spot" as x_x FROM "executed_spread" WHERE "strategy" =~ /ETHUSD/  AND "type"='exit' ) ),
                        (SELECT mean("spread") as m_spread_entry , sum(volume_executed_spot) as sum_vol_entry FROM "executed_spread" WHERE  type='entry'),
                        (SELECT sum(volume_executed_spot) as sum_vol_exit FROM "executed_spread" WHERE  type='exit' ),
                        (SELECT sum(volume_executed_spot) as sum_vol FROM "executed_spread"  GROUP BY  "type")
                        WHERE "strategy" =~ /ETHUSD/  AND time >= {t0}ms and time <= {t1}ms GROUP BY "strategy" '''
        result = connection.prod_client_spotswap_dataframe.query(base_query, epoch='ns')
    else:
        result = None

    if result != None:
        df = pd.concat(result, axis=0)
        idx_list = []
        for idx in range(len(result)):
            idx_list.append(str(df.index[idx]).split("'")[5])
        df.reset_index(drop=True, inplace=True)
        df['strategy'] = idx_list
        return df


def get_funding_for_symbol(t0, t1, exchange, symbol):
    connection = InfluxConnection.getInstance()
    base_query = f'''SELECT "funding" FROM "funding" WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}') AND time >= {t0 - 1000 * 60 * 60 * 8}ms and time <= {t1 + 1000 * 60 * 60 * 16}ms'''
    result = connection.staging_client_spotswap_dataframe.query(base_query, epoch='ns')
    return result['funding']


def get_predicted_funding_for_symbol(t0, t1, exchange, symbol):
    connection = InfluxConnection.getInstance()
    base_query = f'''SELECT "funding" FROM "predicted_funding" WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}') AND time >= {t0 - 1000 * 60 * 60 * 8}ms and time <= {t1 + 1000 * 60 * 60 * 16}ms'''
    result = connection.staging_client_spotswap_dataframe.query(base_query, epoch='ns')
    return result['predicted_funding']


def get_estimated_next_funding_for_symbol(t0, t1, exchange, symbol):
    connection = InfluxConnection.getInstance()
    base_query = f'''SELECT "funding" FROM "estimated_next_funding" WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}') AND time >= {t0 - 1000 * 60 * 60 * 8}ms and time <= {t1 + 1000 * 60 * 60 * 16}ms'''
    result = connection.staging_client_spotswap_dataframe.query(base_query, epoch='ns')
    return result['estimated_next_funding']


def get_funding(t0, t1):
    connection = InfluxConnection.getInstance()
    base_query = f'''SELECT sum(pnl) as funding FROM "mark_to_market_funding"
    WHERE  (time >= {t0}ms AND time <= {t1}ms )
    GROUP BY "environment" '''
    result = connection.prod_client_spotswap_dataframe.query(base_query, epoch='ns')

    if result != None:
        df = pd.concat(result, axis=0)
        idx_list = []
        for idx in range(len(result)):
            idx_list.append(str(df.index[idx]).split("'")[5])
        df.reset_index(drop=True, inplace=True)
        df['account_name'] = idx_list
        return df


def get_deribit_implied_volatility(t0, t1):
    connection = InfluxConnection.getInstance()
    base_query = f"""SELECT "high" FROM "dvol" WHERE time >= {t0}ms and time <= {t1}ms  GROUP BY "coin" """
    result = connection.staging_client_spotswap_dataframe.query(base_query, epoch='ns')
    if len(result) != 0:
        df1 = result[('dvol', (('coin', 'BTC'),))]
        df1['Time'] = df1.index
        df1.rename(columns={"high": "dvol_btc"}, inplace=True)
        df2 = result[('dvol', (('coin', 'ETH'),))]
        df2['Time'] = df2.index
        df2.rename(columns={"high": "dvol_eth"}, inplace=True)
        df = pd.merge_ordered(df1, df2, on='Time')
        df['timestamp'] = df['Time'].view(int) // 10 ** 9
        return df[['timestamp', 'Time', 'dvol_btc', 'dvol_eth']]
    else:
        return pd.DataFrame(columns=['timestamp', 'Time', 'dvol_btc', 'dvol_eth'])


def get_realtime_funding_values(t0: int = 1677978000000, t1: int = 1680048000000, exchange: str = 'Deribit',
                                symbol: str = 'BTC-PERPETUAL', moving_average: int = 90):
    connection = InfluxConnection.getInstance()
    base_query = f""" SELECT moving_average(mean("funding"), {moving_average})  AS funding FROM real_time_funding WHERE exchange = '{exchange}' AND symbol = '{symbol}' AND time >= {t0 - 1000 * 60 * 60 * 8}ms and time <= {t1 + 1000 * 60 * 60 * 2}ms  GROUP BY time(10s) fill(null) """
    result = connection.staging_client_spotswap_dataframe.query(base_query, epoch='ns')
    return result['real_time_funding']


def funding_values_mark_to_market(t0: int = 0, t1: int = 0, exchange: str = None, symbol: str = None,
                                  environment: str = None):
    connection = InfluxConnection.getInstance()
    if symbol == 'ETH-PERPETUAL':
        query = f''' SELECT "funding_rate" as "funding" FROM "mark_to_market_funding"
         WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}' and "environment" = '[Deribit] EquinoxAIBV_10_ETH') AND 
         time >={t0}ms and time <= {t1}ms'''
    elif symbol == 'BTC-PERPETUAL':
        query = f''' SELECT "funding_rate" as "funding" FROM "mark_to_market_funding"
            WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}'and "environment" = '[Deribit] EquinoxAIBV_11_BTC') AND 
            time >={t0}ms and time <= {t1}ms'''
    else:
        query = f''' SELECT mean("funding_rate") as "funding" FROM "mark_to_market_funding"
                 WHERE ("exchange" = '{exchange}' AND "symbol" = '{symbol}') AND 
                 time >={t0}ms and time <= {t1}ms GROUP BY time(10s)'''

    if environment == 'production':
        result = connection.prod_client_spotswap_dataframe.query(query, epoch='ns')
    elif environment == 'staging':
        result = connection.staging_client_spotswap_dataframe.query(query, epoch='ns')
    else:
        result = None
    # print(query)
    return result["mark_to_market_funding"]

def create_price_dataframe_local_folder(t_start: int = 0,
                                        t_end: int = 0,
                                        spot_exchange: str = 'Deribit',
                                        spot_symbol: str = 'ETH-PERPETUAL',
                                        swap_exchange: str = 'BitMEX',
                                        swap_symbol: str = 'ETHUSD',
                                        side: str = 'both'):
    """
    Creates a pandas DataFrame by aggregating price data from files located in a specified local folder.

    This function scans a specified local directory for files containing price data, reads each of these files
    into a pandas DataFrame, and then combines these DataFrames into a single DataFrame.

    Parameters:
    - t_start (int): Start timestamp in milliseconds.
    - t_end (int): End timestamp in milliseconds.
    - spot_exchange (str): Exchange name for spot data.
    - spot_symbol (str): Symbol for spot data.
    - swap_exchange (str): Exchange name for swap data.
    - swap_symbol (str): Symbol for swap data.
    - side (str): Market side ('Bid', 'Ask', or 'both').

    Returns:
    - Tuple of DataFrames (price_bid, price_ask) if side is 'both'.
    - DataFrames (df1, df2) filtered by side otherwise.
    """

    # Locate Base Directory
    base_dir = f"/home/equinoxai/data"
    if not os.path.isdir(base_dir):
        base_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "../../", "simulations", "simulations_management", "data")
        )
    base_dir_spot = os.path.normpath(f"{base_dir}/prices/{spot_exchange}/{spot_symbol}")
    base_dir_swap = os.path.normpath(f"{base_dir}/prices/{swap_exchange}/{swap_symbol}")

    # Convert start and end timestamps to datetime
    start_date = pd.to_datetime(t_start, unit='ms', utc=True)
    end_date = pd.to_datetime(t_end, unit='ms', utc=True)

    data_list_spot = []
    data_list_swap = []

    iterator = DateRangeIterator(start_date, end_date)

    while iterator.has_more_ranges():
        current_start_date, current_end_date = iterator.get_next_range()
        data_list_spot, data_list_swap = sub_function(base_dir_spot=base_dir_spot,
                                                      base_dir_swap=base_dir_swap,
                                                      t_start=current_start_date,
                                                      t_end=current_end_date,
                                                      spot_exchange=spot_exchange,
                                                      spot_symbol=spot_symbol,
                                                      swap_exchange=swap_exchange,
                                                      swap_symbol=swap_symbol,
                                                      data_list_spot=data_list_spot,
                                                      data_list_swap=data_list_swap)

    df1 = pd.DataFrame(data_list_spot, columns=['Time', 'price', 'side']).astype(dtype={'Time': 'datetime64[ns, UTC]',
                                                                                        'price': float, 'side': str})
    df1['Time'] = pd.to_datetime(df1.Time, utc=True)

    df2 = pd.DataFrame(data_list_swap, columns=['Time', 'price', 'side']).astype(dtype={'Time': 'datetime64[ns, UTC]',
                                                                                        'price': float, 'side': str})
    df2['Time'] = pd.to_datetime(df2.Time, utc=True)

    timems_spot = df1.Time.astype(np.int64) // 10 ** 6
    timems_swap = df2.Time.astype(np.int64) // 10 ** 6

    df1['timems'] = timems_spot
    df2['timems'] = timems_swap

    # Keep only prices that are after t_start and before t_end
    df1 = df1[(df1['timems'] <= t_end) & (df1['timems'] >= t_start)]
    df2 = df2[(df2['timems'] <= t_end) & (df2['timems'] >= t_start)]

    df1.drop_duplicates(subset=['timems', 'side'], keep='last', inplace=True)
    df2.drop_duplicates(subset=['timems', 'side'], keep='last', inplace=True)

    ### Bid/Ask Entry/Exit
    if side == 'both':
        price_bid = pd.merge_ordered(df1[df1.side == 'Bid'].drop(columns=['side']),
                                     df2[df2.side == 'Bid'].drop(columns=['side']),
                                     on='Time',
                                     suffixes=['_spot_bid', '_swap_bid'])
        price_ask = pd.merge_ordered(df1[df1.side == 'Ask'].drop(columns=['side']),
                                     df2[df2.side == 'Ask'].drop(columns=['side']),
                                     on='Time',
                                     suffixes=['_spot_ask', '_swap_ask'])

        return price_bid, price_ask
    else:
        return df1[df1.side == side], df2[df2.side == side]


def sub_function(base_dir_spot: str = None,
                 base_dir_swap: str = None,
                 t_start: datetime = None,
                 t_end: datetime = None,
                 spot_exchange: str = None,
                 spot_symbol: str = None,
                 swap_exchange: str = None,
                 swap_symbol: str = None,
                 data_list_spot: list = None,
                 data_list_swap: list = None):
    """
    Fetch data from local storage or external source and append to existing data lists.

    Parameters:
    - base_dir_spot (str): Base directory for spot data.
    - base_dir_swap (str): Base directory for swap data.
    - t_start (datetime): Start date.
    - t_end (datetime): End date.
    - spot_exchange (str): Exchange name for spot data.
    - spot_symbol (str): Symbol for spot data.
    - swap_exchange (str): Exchange name for swap data.
    - swap_symbol (str): Symbol for swap data.
    - data_list_spot (list): List to append spot data.
    - data_list_swap (list): List to append swap data.

    Returns:
    - Updated data lists for spot and swap data.
    """
    temp_start, temp_end = convert_dates_to_timestamps(t_start, t_end)
    clipped_start = drop_hours_from_datetime_object(t_start)

    # Fetch spot data
    data_list_spot = fetch_data(f"{base_dir_spot}/{spot_exchange}_{spot_symbol}_{clipped_start}.parquet.br",
                                temp_start, temp_end, spot_exchange, spot_symbol, data_list_spot)

    # Fetch swap data
    data_list_swap = fetch_data(f"{base_dir_swap}/{swap_exchange}_{swap_symbol}_{clipped_start}.parquet.br",
                                temp_start, temp_end, swap_exchange, swap_symbol, data_list_swap)

    return data_list_spot, data_list_swap


def fetch_data(local_dir, t_start, t_end, exchange, symbol, data_list):
    """
    Helper function to read parquet files or query data from an external source and append to existing data list.

    Parameters:
    - local_dir (str): Local directory path for the data file.
    - t_start (int): Start timestamp in milliseconds.
    - t_end (int): End timestamp in milliseconds.
    - exchange (str): Exchange name.
    - symbol (str): Symbol name.
    - data_list (list): List to append the data.

    Returns:
    - Updated data list.
    """
    if os.path.exists(local_dir):
        try:
            df = pd.read_parquet(local_dir)
        except Exception as e:
            print(f"Error reading {local_dir}: {e}")
            return data_list
        data_list.extend(df.values.tolist())
    else:
        print(f"File {local_dir} did not exist. Data queried from InfluxDB.")
        if t_start != t_end:
            df = get_price(t_start=t_start,
                           t_end=t_end,
                           exchange=exchange,
                           symbol=symbol,
                           side='Ask', environment='staging',
                           split_data=True, use_side=False)
            df['timestamp'] = df.index.astype(str)
            df['price'] = df['price'].astype(str)
            df.reset_index(drop=True, inplace=True)
            data_list.extend(df[['timestamp', 'price', 'side']].values.tolist())
    return data_list





def create_price_dataframe_from_influxdb(t_start: int = 0, t_end: int = 0,
                                         spot_exchange: str = 'Deribit',
                                         spot_symbol: str = 'ETH-PERPETUAL',
                                         swap_exchange: str = 'BitMEX',
                                         swap_symbol: str = 'ETHUSD',
                                         environment: str = 'staging'):
    print(f"t_start= {t_start}, t_end= {t_end}")
    price1 = get_price(t_start=t_start, t_end=t_end, exchange=spot_exchange, symbol=spot_symbol, side='Ask',
                       environment=environment)

    price2 = get_price(t_start=t_start, t_end=t_end, exchange=swap_exchange, symbol=swap_symbol, side='Ask',
                       environment=environment)

    price1['Time'] = price1.index
    price2['Time'] = price2.index

    timems_spot = price1.Time.view(np.int64) // 10 ** 6
    timems_swap = price2.Time.view(np.int64) // 10 ** 6
    indices_no_data_spot = np.where(np.diff(timems_spot, prepend=0) > 60000)[0]
    indices_no_data_swap = np.where(np.diff(timems_swap, prepend=0) > 60000)[0]
    price1["has_prices_spot"] = True
    price1["has_prices_spot"].iloc[indices_no_data_spot] = False
    price1["has_prices_spot"] = price1["has_prices_spot"].shift(-1)
    price2["has_prices_swap"] = True
    price2["has_prices_swap"].iloc[indices_no_data_swap] = False
    price2["has_prices_swap"] = price2["has_prices_swap"].shift(-1)
    # merge the price dataframes
    price_ask = pd.merge_ordered(price1, price2, on='Time', suffixes=['_spot_entry', '_swap_entry'])

    price3 = get_price(t_start=t_start,
                       t_end=t_end,
                       exchange=spot_exchange,
                       symbol=spot_symbol,
                       side='Bid',
                       environment=environment)

    price4 = get_price(t_start=t_start,
                       t_end=t_end,
                       exchange=swap_exchange,
                       symbol=swap_symbol,
                       side='Bid',
                       environment=environment)

    price3['Time'] = price3.index
    price4['Time'] = price4.index

    price_bid = pd.merge_ordered(price3, price4, on='Time', suffixes=['_spot_exit', '_swap_exit'])

    return price_bid, price_ask


def get_data_for_trader(t_start, t_end, exchange_spot, spot_instrument, exchange_swap, swap_instrument, swap_fee,
                        spot_fee, strategy, area_spread_threshold, environment, band_type,
                        window_size=None, exit_delta_spread=None, entry_delta_spread=None,
                        band_funding_system=None,
                        hoursBeforeSwapList: list = None,
                        slowWeightSwapList: list = None,
                        fastWeightSwapList: list = None,
                        hoursBeforeSpotList: list = None,
                        slowWeightSpotList: list = None,
                        fastWeightSpotList: list = None,
                        generate_percentage_bands=False,
                        lookback=None, recomputation_time=None, target_percentage_exit=None,
                        target_percentage_entry=None, entry_opportunity_source=None, exit_opportunity_source=None,
                        minimum_target=None, use_aggregated_opportunity_points=None, ending=None,
                        force_band_creation=True, move_bogdan_band='No', use_bp=False,
                        window_size2=None, exit_delta_spread2=None, entry_delta_spread2=None,
                        band_funding_system2=None, funding_window=90,
                        funding_periods_lookback=0,
                        slow_funding_window=0,
                        funding_options=None,
                        use_stored_bands=False):
    """
    Variables in the function get_data_for_trader:
    t_start  = start timestamp of the period in millis
    t_end = end timestamp of the period in milliseconds
    exchange_spot = spot exchange name
    exchange_swap  = swap exchange name
    spot_instrument = coin ticker in spot market
    swap_instrument =  coin ticker in swap market
    strategy =  the name of the strategy if we want to use the band values of a known strategy
    area_spread_threshold =  the area spread parameter
    environment = the environment form the data ar queried, staging or production
    band_type = the type of the band, bogdan_bands, percentage_bands, ...
    window_size = the window size parameter
    entry_delta_spread  =  the entry delta spread parameter
    exit_delta_spread =  the exit delta spread parameter
    band_funding_system = funding system used in the creation of the band in the API, funding_adjusted_exit_band,
    funding_adjusted_exit_band_with_drop
    generate_percentage_bands =  boolean. If true percentage_bands are generated in the API, default=False
    lookback = percentage band parameter
    recomputation_time = percentage band parameter
    target_percentage_entry = percentage band parameter
    target_percentage_exit = percentage band parameter
    entry_opportunity_source = percentage band parameter
    exit_opportunity_source = percentage band parameter
    use_aggregated_opportunity_points =
    ending =
    force_band_creation =  boolean, If True the bands are created in the API and values are writen in influxdb
    move_bogdan_band = move the bands based on the Deribit funding
    """

    price_bid, price_ask = create_price_dataframe_local_folder(t_start=t_start,
                                                               t_end=t_end,
                                                               spot_exchange=exchange_spot,
                                                               spot_symbol=spot_instrument,
                                                               swap_exchange=exchange_swap,
                                                               swap_symbol=swap_instrument,
                                                               )

    df_price = pd.merge_ordered(price_ask, price_bid, on='Time')
    df_price['price_spot_mid'] = (df_price['price_spot_ask'] + df_price['price_spot_bid']) / 2
    print("Got prices")
    # get the band values of the selected strategy
    if band_type == 'percentage_band':
        band_values = get_percentage_band_values(t0=t_start, t1=t_end,
                                                 lookback=lookback,
                                                 recomputation_time=recomputation_time,
                                                 target_percentage_exit=target_percentage_exit,
                                                 target_percentage_entry=target_percentage_entry,
                                                 entry_opportunity_source=entry_opportunity_source,
                                                 exit_opportunity_source=exit_opportunity_source,
                                                 spot_name=exchange_spot,
                                                 spot_instrument=f"hybrid_{spot_instrument}",
                                                 swap_instrument=f"hybrid_{swap_instrument}",
                                                 environment=environment)
        band_values.rename(columns={'entry_band': 'Entry Band', 'exit_band': 'Exit Band'}, inplace=True)
    elif band_type == 'bogdan_bands':
        start = time.time()
        if (force_band_creation or strategy == '') and funding_options is None:
            datalink = DatalinkCreateBands(t_start=t_start,
                                           t_end=t_end,
                                           swap_exchange=exchange_swap,
                                           swap_symbol=swap_instrument,
                                           spot_exchange=exchange_spot,
                                           spot_symbol=spot_instrument,
                                           window_size=window_size,
                                           entry_delta_spread=entry_delta_spread,
                                           exit_delta_spread=exit_delta_spread,
                                           swap_fee=swap_fee,
                                           spot_fee=spot_fee,
                                           generate_percentage_bands=generate_percentage_bands,
                                           funding_system=band_funding_system,
                                           funding_window=funding_window,
                                           funding_periods_lookback=funding_periods_lookback,
                                           slow_funding_window=slow_funding_window,
                                           environment=environment,
                                           recomputation_time=recomputation_time,
                                           entry_opportunity_source=entry_opportunity_source,
                                           exit_opportunity_source=exit_opportunity_source,
                                           target_percentage_entry=target_percentage_entry,
                                           target_percentage_exit=target_percentage_exit,
                                           lookback=lookback,
                                           minimum_target=minimum_target,
                                           use_aggregated_opportunity_points=use_aggregated_opportunity_points,
                                           hoursBeforeSwapList=hoursBeforeSwapList,
                                           slowWeightSwapList=slowWeightSwapList,
                                           fastWeightSwapList=fastWeightSwapList,
                                           hoursBeforeSpotList=hoursBeforeSpotList,
                                           slowWeightSpotList=slowWeightSpotList,
                                           fastWeightSpotList=fastWeightSpotList,
                                           ending=ending,
                                           use_bps=use_bp)
            if not use_stored_bands:
                band_list = datalink.generate_bogdan_bands()
                band_values = format_band_values(band_list)
            else:
                print("Using local bands")
                band_values = datalink.load_bands_from_disk()

            band_values = band_values.astype({"Entry Band": np.float32, "Exit Band": np.float32, "timems": np.int64})
            print(f"It took {time.time() - start}s to query the bands")
        elif (force_band_creation or strategy == '') and funding_options is not None:
            band_creation = FundingOptionsBandCreation(t_start=t_start, t_end=t_end,
                                                       swap_exchange=exchange_swap, swap_symbol=swap_instrument,
                                                       spot_exchange=exchange_spot, spot_symbol=spot_instrument,
                                                       swap_fee=swap_fee, spot_fee=spot_fee,
                                                       window_size_net=window_size,
                                                       entry_delta_spread_net_entry=entry_delta_spread,
                                                       exit_delta_spread_net_exit=exit_delta_spread,
                                                       band_funding_system_net=band_funding_system,
                                                       funding_window=funding_window,
                                                       funding_periods_lookback=funding_periods_lookback,
                                                       slow_funding_window=slow_funding_window,
                                                       window_size_zero=window_size2,
                                                       entry_delta_spread_entry_zero=entry_delta_spread2,
                                                       exit_delta_spread_exit_zero=exit_delta_spread2,
                                                       band_funding_system_zero=band_funding_system2,
                                                       band_funding_option=funding_options,
                                                       hoursBeforeSwapList=hoursBeforeSwapList,
                                                       slowWeightSwapList=slowWeightSwapList,
                                                       fastWeightSwapList=fastWeightSwapList,
                                                       hoursBeforeSpotList=hoursBeforeSpotList,
                                                       slowWeightSpotList=slowWeightSpotList,
                                                       fastWeightSpotList=fastWeightSpotList,
                                                       environment=environment
                                                       )
            band_values = band_creation.create_band_from_api()
        else:
            band_values = get_band_values(t0=t_start, t1=t_end, typeb=band_type,
                                          strategy=strategy, environment=environment)

    elif band_type == 'custom_multi' or band_type == 'custom_multi_symmetrical' or band_type == 'custom_multi_custom':

        datalink = DatalinkCreateBands(t_start=t_start, t_end=t_end, swap_exchange=exchange_swap,
                                       swap_symbol=swap_instrument, spot_exchange=exchange_spot,
                                       spot_symbol=spot_instrument, window_size=window_size,
                                       entry_delta_spread=entry_delta_spread, exit_delta_spread=exit_delta_spread,
                                       swap_fee=swap_fee, spot_fee=spot_fee,
                                       generate_percentage_bands=generate_percentage_bands,
                                       funding_system=band_funding_system, funding_window=funding_window,
                                       environment=environment, recomputation_time=recomputation_time,
                                       entry_opportunity_source=entry_opportunity_source,
                                       exit_opportunity_source=exit_opportunity_source,
                                       target_percentage_entry=target_percentage_entry,
                                       target_percentage_exit=target_percentage_exit, lookback=lookback,
                                       minimum_target=minimum_target,
                                       use_aggregated_opportunity_points=use_aggregated_opportunity_points,
                                       ending=ending)
        if force_band_creation:
            band_list = datalink.generate_bogdan_bands()
            band_values = format_band_values(band_list)
            time.sleep(5)
        else:
            band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands',
                                          strategy=datalink.strategy_name, environment=environment)
        print(f'band length: {len(band_values)}')

        try:
            isinstance(band_values, type(None))
            datalink.generate_bogdan_bands()
            time.sleep(5)
        except:
            if band_values.empty:
                band_list = datalink.generate_bogdan_bands()
                band_values = format_band_values(band_list)
                time.sleep(5)
            elif band_values.iloc[:100, :].dropna().empty or band_values.iloc[100:, :].dropna().empty:
                band_list = datalink.generate_bogdan_bands()
                band_values = format_band_values(band_list)
                time.sleep(5)

    # add the deribit funding to the band
    if move_bogdan_band != 'No' and exchange_spot == 'Deribit' and (spot_instrument == 'ETH-PERPETUAL' or
                                                                    spot_instrument == 'BTC-PERPETUAL'):
        deribit_funding = funding_values(t0=t_start, t1=t_end, exchange=exchange_spot,
                                         symbol=spot_instrument, environment=environment)
        deribit_funding['funding'] = deribit_funding['funding'] / (8 * 3600)
        deribit_funding['rolling_funding'] = deribit_funding['funding'].rolling('8h').mean()
        deribit_funding['rolling_funding'].fillna(0, inplace=True)
        # deribit_funding['percentual_change'] = (deribit_funding['rolling_funding'] - deribit_funding['funding']) / deribit_funding['funding'] * 100
        deribit_funding['Time'] = deribit_funding.index
        deribit_funding.reset_index(drop=True, inplace=True)
        band_values = pd.merge_ordered(band_values, deribit_funding, on='Time')
        band_values['Entry Band'].ffill(inplace=True)
        band_values['Exit Band'].ffill(inplace=True)
        if funding_options is not None:
            band_values['Entry Band Enter to Zero'].ffill(inplace=True)
            band_values['Exit Band Exit to Zero'].ffill(inplace=True)
        band_values['funding'].ffill(inplace=True)
        band_values['rolling_funding'].fillna(0, inplace=True)
        # take the mid price for deribit
        if move_bogdan_band == 'move_entry':
            band_values['Entry Band Old'] = band_values['Entry Band']
            band_values['Entry Band'] = band_values['Entry Band Old'] + band_values['Entry Band Old'] * \
                                        band_values['rolling_funding']

        elif move_bogdan_band == 'move_both':
            band_values['Entry Band Old'] = band_values['Entry Band']
            band_values['Exit Band Old'] = band_values['Exit Band']
            band_values['Entry Band'] = band_values['Entry Band Old'] + band_values['Entry Band Old'] * \
                                        band_values['rolling_funding']
            band_values['Exit Band'] = band_values['Exit Band Old'] + band_values['Exit Band Old'] * \
                                       band_values['rolling_funding']
    # merge all dataframes in to a new one containing all values.
    df = pd.merge_ordered(band_values, df_price, on='Time')
    df.ffill(inplace=True)
    df['timems'] = df.Time.view(np.int64) // 10 ** 6
    df.set_index('timems', drop=False, inplace=True)
    print(f'dataframe length: {len(df)}')

    df.drop_duplicates(subset=['timems'], keep='last', inplace=True)

    df['entry_area_spread'] = 0
    df['exit_area_spread'] = 0

    df['spread_bid_bid'] = spread_entry_func(df['price_swap_bid'], df['price_spot_bid'],
                                           swap_fee=swap_fee, spot_fee=spot_fee)
    df['spread_ask_ask'] = spread_exit_func(df['price_swap_ask'], df['price_spot_ask'],
                                         swap_fee=swap_fee, spot_fee=spot_fee)
    if area_spread_threshold != 0:
        df['multiplier'] = df['timems'].diff() // 100

        df_mat = df.loc[:, ['spread_bid_bid',
                            'spread_ask_ask',
                            'Entry Band',
                            'Exit Band',
                            'multiplier',
                            'entry_area_spread',
                            'exit_area_spread']].to_numpy()

        df_mat = df_numba(df_mat)

        df['entry_area_spread'] = df_mat[:, 5]
        df['exit_area_spread'] = df_mat[:, 6]
    print(f'dataframe length: {len(df)}')
    if band_type == 'custom_multi' or band_type == 'custom_multi_symmetrical' or band_type == 'custom_multi_custom' or \
            strategy == '':
        try:
            str_name = datalink.strategy_name
        except:
            str_name = f"generic_{exchange_swap}_{swap_instrument}_{exchange_spot}_{spot_instrument}_ws_{window_size}_ens_{entry_delta_spread}_exs_{exit_delta_spread}"

        return df, str_name
    else:
        return df, None
