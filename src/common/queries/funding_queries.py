import os
from dataclasses import dataclass
from datetime import time
from typing import Dict

import numba
import numpy as np
import pandas as pd

from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.constants.constants import one_hour, one_second, five_minutes


@dataclass
class FundingRatiosParams():
    ratio_to_zero_entry: int = None
    ratio_to_zero_exit: int = None
    ratio_entry: int = None
    ratio_exit: int = None

    def __bool__(self):
        if self.ratio_to_zero_entry is not None:
            return True
        return False

    def __eq__(self, other):
        return (self.ratio_to_zero_entry == other.ratio_to_zero_entry) and \
            (self.ratio_to_zero_exit == other.ratio_to_zero_exit) and \
            (self.ratio_entry == other.ratio_entry) and \
            (self.ratio_exit == other.ratio_exit)

    def __neg__(self):
        return FundingRatiosParams(ratio_to_zero_entry=-self.ratio_to_zero_entry,
                                   ratio_to_zero_exit=-self.ratio_to_zero_exit,
                                   ratio_entry=-self.ratio_entry,
                                   ratio_exit=-self.ratio_exit)


class FundingBase():
    ms_column = 0
    value_column = 1
    idx = 0

    def __init__(self, fundings):
        self.fundings = fundings
        self.current_funding = 0

    def update(self, timestamp):
        self.idx = np.searchsorted(self.fundings[:, self.ms_column], timestamp, side='right')
        self.current_funding = self.fundings[self.idx, self.value_column]

    def get_next_funding(self):
        return self.current_funding

    def get_next_funding_entry(self):
        return self.get_next_funding()

    def get_next_funding_to_zero_entry(self):
        return self.get_next_funding()

    def get_next_funding_exit(self):
        return self.get_next_funding()

    def get_next_funding_to_zero_exit(self):
        return self.get_next_funding()

    def get_predicted_funding(self):
        return


class FundingBaseWithRatios(FundingBase):
    ms_column = 0
    value_column = 1
    idx = 0

    def __init__(self, fundings, funding_ratios: FundingRatiosParams):
        super(FundingBaseWithRatios, self).__init__(fundings)
        self.funding_ratios = funding_ratios

    def update(self, timestamp):
        self.idx = np.searchsorted(self.fundings[:, self.ms_column], timestamp, side='right')
        if self.idx >= len(self.fundings):
            self.current_funding = 0
            return
        self.current_funding = self.fundings[self.idx, self.value_column]

    def get_next_funding(self):
        return self.current_funding

    def get_next_funding_entry(self):
        return self.get_next_funding() * self.funding_ratios.ratio_entry

    def get_next_funding_to_zero_entry(self):
        return self.get_next_funding() * self.funding_ratios.ratio_to_zero_entry

    def get_next_funding_exit(self):
        return self.get_next_funding() * self.funding_ratios.ratio_exit

    def get_next_funding_to_zero_exit(self):
        return self.get_next_funding() * self.funding_ratios.ratio_to_zero_exit

    def get_predicted_funding(self):
        return


class FundingBinanceDiscounted(FundingBase):
    funding_interval = 8 * one_hour
    # the n in the formula of the TWA. From Binance
    max_weight = 5760

    def __init__(self, fundings):
        super(FundingBinanceDiscounted, self).__init__(fundings)
        self.weight = 1

    def update(self, timestamp):
        super(FundingBinanceDiscounted, self).update(timestamp)
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight

    def get_next_funding(self):
        return self.current_funding * self.weight


class FundingBinanceDiscountedWithRatios(FundingBaseWithRatios):
    funding_interval = 8 * one_hour
    # the n in the formula of the TWA. From Binance
    max_weight = 5760

    def __init__(self, fundings, funding_ratios: FundingRatiosParams):
        super(FundingBinanceDiscountedWithRatios, self).__init__(fundings, funding_ratios)
        self.weight = 1

    def update(self, timestamp):
        super(FundingBinanceDiscountedWithRatios, self).update(timestamp)
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight

    def get_next_funding(self):
        return super(FundingBinanceDiscountedWithRatios, self).get_next_funding() * self.weight


class FundingBinanceDiscountedWithRatiosModel(FundingBaseWithRatios):
    funding_interval = 8 * one_hour
    # the n in the formula of the TWA. From Binance
    max_weight = 5760

    def __init__(self, fundings, funding_ratios: FundingRatiosParams, prediction_emitter):
        super(FundingBinanceDiscountedWithRatiosModel, self).__init__(fundings, funding_ratios)
        self.weight = 1
        self.prediction_emitter = prediction_emitter

    def update(self, timestamp):
        # super(FundingBinanceDiscountedWithRatiosModel, self).update(timestamp)
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight
        self.current_funding = self.prediction_emitter.predict(timestamp)

    def get_next_funding(self):
        return super(FundingBinanceDiscountedWithRatiosModel, self).get_next_funding()  # * self.weight


class FundingSystemEmpty():
    name = ""

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBase):
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        pass

    def band_adjustments(self):
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def entry_band_adjustment(self):
        return 0

    def exit_band_adjustment(self):
        return 0

    def entry_band_adjustment_to_zero(self):
        return 0

    def exit_band_adjustment_to_zero(self):
        return 0


class FundingSystemDeribitBitMEXWithRatios():
    name = "deribit_bitmex_with_ratios"

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBaseWithRatios, funding_swap: FundingBaseWithRatios):
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        self.funding_spot.update(timestamp)
        self.funding_swap.update(timestamp)
        self.timestamp_last_update = timestamp

    def band_adjustments(self):
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def band_adjustments_to_zero(self):
        return self.entry_band_adjustment_to_zero(), self.exit_band_adjustment_to_zero()

    def entry_band_adjustment(self):
        return 10000 * (self.funding_spot.get_next_funding_entry() + self.funding_swap.get_next_funding_entry())

    def exit_band_adjustment(self):
        return 10000 * (self.funding_spot.get_next_funding_exit() + self.funding_swap.get_next_funding_exit())

    def entry_band_adjustment_to_zero(self):
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_entry() + self.funding_swap.get_next_funding_to_zero_entry())

    def exit_band_adjustment_to_zero(self):
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_exit() + self.funding_swap.get_next_funding_to_zero_exit())


class FundingSystemOkxBinanceDiscounted():
    name = "okx_binance_discounted"

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBase):
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        self.funding_spot.update(timestamp)
        self.funding_swap.update(timestamp)
        self.timestamp_last_update = timestamp

    def band_adjustments(self):
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def entry_band_adjustment(self):
        if self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() >= 0:
            return abs(self.funding_spot.get_next_funding()) * 10000
        elif self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() < 0:
            return (abs(self.funding_spot.get_next_funding()) + abs(self.funding_swap.get_next_funding())) * 10000
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() >= 0:
            return 0
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() < 0:
            return abs(self.funding_swap.get_next_funding()) * 10000

    def exit_band_adjustment(self):
        if self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() >= 0:
            # Move band down by swap funding
            return abs(self.funding_swap.get_next_funding()) * 10000
        elif self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() < 0:
            return 0
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() >= 0:
            return (abs(self.funding_spot.get_next_funding()) + abs(self.funding_swap.get_next_funding())) * 10000
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() < 0:
            return abs(self.funding_spot.get_next_funding()) * 10000

    def entry_band_adjustment_to_zero(self):
        return 0

    def exit_band_adjustment_to_zero(self):
        return 0


class FundingSystemOkxBinanceDiscoutedWithRatios(FundingSystemOkxBinanceDiscounted):
    name = "okx_binance_discounted_with_ratios"

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBinanceDiscountedWithRatios):
        super(FundingSystemOkxBinanceDiscoutedWithRatios, self).__init__(funding_spot, funding_swap)

    def band_adjustments(self):
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def band_adjustments_to_zero(self):
        return self.entry_band_adjustment_to_zero(), self.exit_band_adjustment_to_zero()

    def entry_band_adjustment(self):
        return 10000 * (self.funding_spot.get_next_funding_entry() + self.funding_swap.get_next_funding_entry())

    def exit_band_adjustment(self):
        return 10000 * (self.funding_spot.get_next_funding_exit() + self.funding_swap.get_next_funding_exit())

    def entry_band_adjustment_to_zero(self):
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_entry() + self.funding_swap.get_next_funding_to_zero_entry())

    def exit_band_adjustment_to_zero(self):
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_exit() + self.funding_swap.get_next_funding_to_zero_exit())


class FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff(FundingSystemOkxBinanceDiscoutedWithRatios):
    name = "okx_binance_discounted_with_ratios_on_diff"

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBaseWithRatios, funding_swap: FundingBinanceDiscountedWithRatios):
        super(FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff, self).__init__(funding_spot, funding_swap)
        if funding_spot.funding_ratios != -funding_swap.funding_ratios:
            raise ("This funding system accepts only opposite values for the spot and swap ratios!")


class FundingSystemBinanceBinanceDiscoutedWithRatiosOnDiff(FundingSystemOkxBinanceDiscoutedWithRatios):
    name = "binance_binance_discounted_with_ratios_on_diff"


class FundingSystemOkxBinanceDiscoutedWithRatiosModel(FundingSystemOkxBinanceDiscoutedWithRatios):
    name = "okx_binance_discounted_with_ratios_model"

    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBinanceDiscountedWithRatiosModel):
        super(FundingSystemOkxBinanceDiscoutedWithRatiosModel, self).__init__(funding_spot, funding_swap)

    # What is needed? move each band up or down. There are 4 bands instead of 2 (entry/exit in position and
    # entry/exit not in position). There are 8 parameters, 2 per band, as each band can move up or down.
    # There are two additional cases


funding_systems = {"okx_binance_discounted": FundingSystemOkxBinanceDiscounted,
                   "": FundingSystemEmpty,
                   "okx_binance_discounted_with_ratios": FundingSystemOkxBinanceDiscoutedWithRatios,
                   "okx_binance_discounted_with_ratios_model": FundingSystemOkxBinanceDiscoutedWithRatiosModel,
                   "okx_binance_discounted_with_ratios_on_diff": FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff,
                   "deribit_bitmex_with_ratios": FundingSystemDeribitBitMEXWithRatios,
                   "binance_binance_discounted_with_ratios_on_diff": FundingSystemBinanceBinanceDiscoutedWithRatiosOnDiff}

funding_classes = {"Binance": FundingBinanceDiscountedWithRatios,
                   "Okex": FundingBaseWithRatios,
                   "BitMEX": FundingBaseWithRatios,
                   "Deribit": FundingBaseWithRatios}


def funding_values(t0, t1, exchange, symbol, environment):
    connection = InfluxConnection.getInstance()
    denormalized_factor = 8 * 60 * 60
    temp = []
    if exchange == 'Deribit':
        query = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                    FROM "real_time_funding"
                    WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t0}ms and time <= {t1}ms) GROUP BY time(1s)'''
    else:
        query = f''' SELECT "funding"
                        FROM "funding"
                        WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t0}ms and time <= {t1}ms) '''
    if environment == 'production':
        result = connection.prod_client_spotswap_dataframe.query(query, epoch='ns')
    elif environment == 'staging':
        if exchange == 'Deribit':
            t_start = t0
            t_end = t_start + 24 * 60 * 60 * 1000
            while t_end <= t1:
                query1 = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                                FROM "real_time_funding"
                                WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t_start}ms and time <= {t_end}ms) GROUP BY time(1s)'''
                result = connection.staging_client_spotswap_dataframe.query(query1, epoch='ns')
                temp.append(result['real_time_funding'])
                t_start = t_end
                t_end = t_start + 24 * 60 * 60 * 1000
                time.sleep(1)
            if 0 < t1 - t_start < 24 * 60 * 60 * 1000:
                query1 = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                                FROM "real_time_funding"
                                WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t_start}ms and time <= {t1}ms) GROUP BY time(1s)'''
                result = connection.staging_client_spotswap_dataframe.query(query1, epoch='ns')
                if len(result) > 0:
                    temp.append(result['real_time_funding'])
            return pd.concat(temp)
        else:
            result = connection.staging_client_spotswap_dataframe.query(query, epoch='ns')
    else:
        result = None

    if exchange == 'Deribit':
        return result["real_time_funding"]
    else:
        return result["funding"]


def get_real_time_funding_local(t0: int = 0, t1: int = 0, market: str = 'Deribit', symbol: str = 'BTC-PERPETUAL'):
    try:
        day_in_millis = 1000 * 60 * 60 * 24
        dfs = []
        if t1 - t0 >= day_in_millis:
            t_start = t0
            t_end = t0 + day_in_millis

            while t_end <= t1:
                if t1 - day_in_millis <= t_end <= t1:
                    t_end = t1

                base_dir = f"/home/equinoxai/data"
                if not os.path.isdir(base_dir):
                    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../",
                                            "simulations_management", "data")
                local_dir_swap = f"{base_dir}/real_time_funding/{market}/{symbol}/{market}_{symbol}_{pd.to_datetime(t_start, unit='ms', utc=True).date()}.parquet.br"
                if os.path.exists(local_dir_swap):
                    # print(f"Loading real time funding from local file {local_dir_swap}")
                    try:
                        df = pd.read_parquet(local_dir_swap, engine="pyarrow")
                    except:
                        df = pd.read_parquet(local_dir_swap)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
                        df = df.set_index("timestamp")
                    elif 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], unit='ns', utc=True)
                        df = df.set_index("time")
                    else:
                        df.index = pd.to_datetime(df.index, unit='ns', utc=True)
                    df['funding'] = df['funding'].astype(np.float64)
                    dfs.append(df)
                    # print(df.head())
                    df = None
                    # time.sleep(1)
                else:
                    print(f"Loading real time funding from influx. Couldn't find {local_dir_swap}")
                    # print(f"t_start: {pd.to_datetime(t_start, unit='ms', utc=True).date()}, t_end: {pd.to_datetime(t_end, unit='ms', utc=True).date()}")
                    dfs.append(
                        funding_values(t0=t_start, t1=t_end, exchange=market, symbol=symbol, environment='staging'))
                    time.sleep(1)
                t_start = t_start + day_in_millis + 1000
                t_end = t_start + day_in_millis
            # print('end while loop')
            if 0 < t1 - t_start < day_in_millis:
                # print('entering last if')
                dfs.append(funding_values(t0=t_start, t1=t1, exchange=market, symbol=symbol, environment='staging'))
        else:
            df = funding_values(t0=t0, t1=t1, exchange=market, symbol=symbol, environment='staging')
            dfs.append(df)
            time.sleep(1)
        # print(f"dfs: {pd.concat(dfs)}")
        return pd.concat(dfs)
    except KeyError:
        print(f"keyError: {KeyError} in get_real_time_funding_local")
        return pd.DataFrame(columns=['funding'])


def funding_implementation(t0: int = 0,
                           t1: int = 0,
                           swap_exchange: str = None,
                           swap_symbol: str = None,
                           spot_exchange: str = None,
                           spot_symbol: str = None,
                           position_df: pd.DataFrame = None,
                           environment: str = None):
    if len(position_df) == 0:
        return 0, 0, 0, 0, 0

    try:
        if swap_exchange == 'Deribit':
            swap_funding = get_real_time_funding_local(t0=t0, t1=t1, market=swap_exchange, symbol=swap_symbol)
        else:
            swap_funding = funding_values(t0=t0, t1=t1, exchange=swap_exchange, symbol=swap_symbol,
                                          environment=environment)
        swap_funding['timems'] = swap_funding.index.view(np.int64) // 10 ** 6
        swap_funding.reset_index(drop=True, inplace=True)
        if swap_exchange == 'Deribit':
            swap_index = swap_funding['timems'].searchsorted(position_df['timems'].to_list(), side='left')
        else:
            swap_index = 0
    except:
        swap_funding = pd.DataFrame()
        swap_index = None

    try:
        if spot_exchange == 'Deribit':
            spot_funding = get_real_time_funding_local(t0=t0, t1=t1, market=spot_exchange, symbol=spot_symbol)
        else:
            spot_funding = funding_values(t0=t0, t1=t1, exchange=spot_exchange, symbol=spot_symbol,
                                          environment=environment)
        spot_funding['timems'] = spot_funding.index.view(np.int64) // 10 ** 6
        spot_funding.reset_index(drop=True, inplace=True)
        if spot_exchange == 'Deribit':
            spot_index = spot_funding['timems'].searchsorted(position_df['timems'].to_list(), side='left')
        else:
            spot_index = 0
    except:
        spot_funding = pd.DataFrame()
        spot_index = None

    if len(swap_funding.index) != 0:
        swap_funding['value'] = 0.0
        if swap_exchange == 'Deribit':
            # swap_funding = continuous_funding_fun(t1=t1, funding_df=swap_funding, index_list=swap_index,
            #                                       position_df=position_df, spot=False)
            swap_funding.reset_index(drop=True, inplace=True)
            swap_funding_array = swap_funding.fillna(0).to_numpy()
            position_df_array = position_df[['timems', 'traded_volume']].fillna(0).to_numpy()
            position_df_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            position_df_columns_map['timems'] = 0
            position_df_columns_map['traded_volume'] = 1
            swap_funding_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            for ix in range(len(swap_funding.columns)):
                swap_funding_columns_map[swap_funding.columns[ix]] = ix
            swap_funding_array = continuous_funding_fun_numba(funding_array=swap_funding_array, index_list=swap_index,
                                                              position_array=position_df_array, spot=False,
                                                              position_columns_map=position_df_columns_map,
                                                              funding_columns_map=swap_funding_columns_map)
            swap_funding = pd.DataFrame(swap_funding_array, columns=swap_funding.columns)

        else:
            swap_funding = periodical_funding_fun(t0=t0, t1=t1, swap_exchange=swap_exchange, swap_symbol=swap_symbol,
                                                  position_df=position_df, environment=environment, spot=False)
        total_funding_swap = swap_funding['value'].sum()
    else:
        total_funding_swap = 0

    if len(spot_funding.index) != 0:
        spot_funding['value'] = 0.0
        if spot_exchange == 'Deribit':
            # spot_funding = continuous_funding_fun(t1=t1, funding_df=spot_funding, index_list=spot_index,
            #                                       position_df=position_df)
            spot_funding_array = spot_funding.fillna(0).to_numpy()
            position_df_array = position_df[['timems', 'traded_volume']].fillna(0).to_numpy()
            position_df_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            position_df_columns_map['timems'] = 0
            position_df_columns_map['traded_volume'] = 1
            spot_funding_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            for ix in range(len(spot_funding.columns)):
                spot_funding_columns_map[spot_funding.columns[ix]] = ix
            spot_funding_array = continuous_funding_fun_numba(funding_array=spot_funding_array, index_list=spot_index,
                                                              position_array=position_df_array, spot=True,
                                                              position_columns_map=position_df_columns_map,
                                                              funding_columns_map=spot_funding_columns_map)
            spot_funding = pd.DataFrame(spot_funding_array, columns=spot_funding.columns)


        else:
            spot_funding = periodical_funding_fun(t0=t0, t1=t1, swap_exchange=spot_exchange, swap_symbol=spot_symbol,
                                                  position_df=position_df, environment=environment, spot=True)

        total_funding_spot = spot_funding['value'].sum()
    else:
        total_funding_spot = 0

    return total_funding_spot, total_funding_swap, total_funding_swap + total_funding_spot, spot_funding, swap_funding


def periodical_funding_fun(t0: int = 0, t1: int = 0, swap_exchange: str = None, swap_symbol: str = None,
                           position_df: pd.DataFrame = None, environment: str = None, spot=True):
    funding_df = funding_values(t0=t0, t1=t1, exchange=swap_exchange, symbol=swap_symbol, environment=environment)
    if len(funding_df) == 0 or funding_df.empty:
        print('No funding data available')
        return pd.DataFrame(columns=['funding', 'timems', 'value'])
    funding_df['timems'] = funding_df.index.view(np.int64) // 10 ** 6
    funding_df.fillna(0, inplace=True)
    df = pd.merge_ordered(position_df, funding_df, on='timems')
    df['traded_volume'].ffill(inplace=True)
    df['value'] = np.nan
    for ix in df[~df['funding'].isna()].index:
        if spot:
            if df.loc[ix, 'traded_volume'] > 0 and df.loc[ix, 'funding'] > 0:
                df.loc[ix, 'value'] = -abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] > 0 >= df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] <= 0 < df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            else:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
        else:
            if df.loc[ix, 'traded_volume'] > 0 and df.loc[ix, 'funding'] > 0:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] > 0 >= df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] <= 0 < df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            else:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
    temp = df.loc[~df['value'].isna(), 'value']
    funding_df['value'] = 0.0
    if len(temp) <= len(funding_df):
        val = len(funding_df) - len(temp)
        funding_df.iloc[val:, -1] = temp.values
        return funding_df
    else:
        funding_df['value'] = temp.iloc[:len(funding_df)].values
        return funding_df


@numba.jit(nopython=True)
def continuous_funding_fun_numba(funding_array: np.array = None, index_list: list = None,
                                 position_array: np.array = None, spot: bool = True,
                                 position_columns_map=None, funding_columns_map=None):
    timems = position_columns_map['timems']
    traded_volume = position_columns_map['traded_volume']
    funding = funding_columns_map['funding']
    value = funding_columns_map['value']
    abs_funding_array = np.abs(funding_array[:, :])
    abs_position_array = np.abs(position_array[:, :])
    for idx3, idx4 in zip(index_list, range(len(position_array))):
        if idx4 >= len(position_array) - 1:
            break
        if idx3 >= len(funding_array) - 1:
            break

        pos_dur = int((position_array[idx4 + 1, timems] - position_array[idx4, timems]) / 1000)

        idx_local = idx3
        counter = pos_dur
        while counter >= 0 and idx_local < len(funding_array) - 1:

            funding_deribit = abs_funding_array[idx_local, funding] * abs_position_array[idx4, traded_volume]

            if spot:
                if position_array[idx4, traded_volume] > 0 and funding_array[idx_local, funding] > 0:
                    funding_value = - funding_deribit
                elif position_array[idx4, traded_volume] > 0 >= funding_array[idx_local, funding]:
                    funding_value = funding_deribit
                elif position_array[idx4, traded_volume] <= 0 < funding_array[idx_local, funding]:
                    funding_value = funding_deribit
                else:
                    funding_value = - funding_deribit
            else:
                if position_array[idx4, traded_volume] > 0 and funding_array[idx_local, funding] > 0:
                    funding_value = funding_deribit
                elif position_array[idx4, traded_volume] > 0 >= funding_array[idx_local, funding]:
                    funding_value = - funding_deribit
                elif position_array[idx4, traded_volume] <= 0 < funding_array[idx_local, funding]:
                    funding_value = - funding_deribit
                else:
                    funding_value = funding_deribit

            idx_local += 1
            counter -= 1

            funding_array[idx_local, value] = funding_value
    return funding_array
