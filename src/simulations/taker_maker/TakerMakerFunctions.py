import numba
import pandas as pd
import time
from src.common.utils.quanto_utils import quanto_pnl_func
from src.common.utils.utils import bp_to_dollars
from src.common.connections.DatabaseConnections import InfluxConnection
import numpy as np
from src.common.queries.queries import get_price, get_band_values, get_percentage_band_values, Takers, \
    funding_values_mark_to_market

from pytictoc import TicToc

import warnings

from src.common.equinox_api_call import DatalinkCreateBands

warnings.filterwarnings("ignore")
t = TicToc()
t.tic()


def get_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


def get_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


class LimitOrder:
    def __init__(self, timestamp_posted, price, is_executed, side, is_spot, targeted_spread, volume_executed,
                 max_targeted_depth, cancelled_to_post_deeper=False, timestamp_cancelled=None,
                 price_other_exchange=None, was_trying_to_cancel=False, id_=1, source_at_execution=None,
                 dest_at_execution=None, amount=0):
        self.timestamp_posted = timestamp_posted
        self.timestamp_executed = None
        self.price = price
        self.is_executed = is_executed
        self.side = side
        self.volume_executed = volume_executed
        self.max_targeted_depth = max_targeted_depth
        self.price_other_exchange = price_other_exchange
        self.is_spot = is_spot
        self.targeted_spread = targeted_spread
        self.amount = amount
        self.timestamp_cancelled = timestamp_cancelled
        self.cancelled_to_post_deeper = cancelled_to_post_deeper
        self.was_trying_to_cancel = was_trying_to_cancel
        self.id = id_
        self.source_at_execution = source_at_execution
        self.dest_at_execution = dest_at_execution


class TakerExecution:
    def __init__(self, timestamp_posted, targeted_price, executed_price, side, is_spot, volume_executed,
                 source_at_execution=None, dest_at_execution=None, amount=0):
        self.timestamp_posted = timestamp_posted
        self.timestamp_executed = None
        self.targeted_price = targeted_price
        self.executed_price = executed_price
        self.volume_executed = volume_executed
        self.side = side
        self.is_spot = is_spot
        self.amount = amount
        # Defined for convenience
        self.was_trying_to_cancel = False
        self.source_at_execution = source_at_execution
        self.dest_at_execution = dest_at_execution


class TakerMakerFunctions(object):
    def set_order_depth(self, new_depth):
        # @TODO decide the unit of the order depth
        self.order_depth = new_depth

    def reset_depth(self, event):
        self.predicted_depth = None

    def reset_boolean_depth(self, event):
        self.need_to_post_deeper__ = False

    def is_order_too_deep(self, event):
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()

        entry_swap = self.df.loc[idx_lat_swap, 'price_swap_entry']
        exit_swap = self.df.loc[idx_lat_swap, 'price_swap_exit']
        order = self.limit_orders_swap[0]
        self.df.iloc[self.df.index.get_loc(self.timestamp) - 4: self.df.index.get_loc(self.timestamp) + 20]

        if self.side == 'entry':
            targeted_depth_usd = bp_to_dollars(order.max_targeted_depth, entry_swap)
            if targeted_depth_usd < order.price - entry_swap:
                if self.verbose:
                    print(
                        f"Time: {self.df.Time.loc[self.timestamp]}\t Order too deep. Side: {self.limit_orders_swap[0].side}\t Order id {self.limit_orders_swap[0].id}\t Order spread {self.limit_orders_swap[0].targeted_spread}\t Order price {order.price}\t Best ask {entry_swap}\t Max depth {targeted_depth_usd}$")

                return True

        elif self.side == 'exit':
            targeted_depth_usd = bp_to_dollars(self.limit_orders_swap[0].max_targeted_depth, exit_swap)
            if targeted_depth_usd < exit_swap - order.price:
                if self.verbose:
                    print(
                        f"Time: {self.df.Time.loc[self.timestamp]}\t Order too deep. Side: {self.limit_orders_swap[0].side}\t Order id {self.limit_orders_swap[0].id}\t Order spread {self.limit_orders_swap[0].targeted_spread}\t Order price {order.price}\t Best ask {exit_swap}\t Max depth {targeted_depth_usd}$")
                return True

    @property
    def need_to_post_deeper(self):
        self.predicted_depth = self.depth_posting_predictor.where_to_post(self.timestamp)
        # self.depth_posting_predictor.need_to_post_deeper(*args)
        # predicted depth is larger than current depth -> re-post
        if len(self.limit_orders_swap) == 0:
            return True
        current_depth = self.limit_orders_swap[0].max_targeted_depth
        if self.predicted_depth > current_depth:
            self.need_to_post_deeper__ = True
            return True
        return False

    def compute_volume_size(self):
        traded_volume = self.max_trade_volume
        return traded_volume

    def find_known_values_index(self):
        '''
        Find the index (timestamp) of the known spot and swap prices.
        Known values: values we know at any time moment and include a latency concern the time we receive the
        information of a new price.
        '''

        df = self.df
        if self.timestamp - self.latency_spot >= df.index[0]:
            # position_current_timestamp = df.index.get_loc(self.timestamp)
            position_current_timestamp = df.index.searchsorted(self.timestamp)
            idx_lat_spot = df.index[
                get_index_left(self.timestamps, position_current_timestamp, latency=self.latency_spot)]
            idx_lat_swap = df.index[
                get_index_left(self.timestamps, position_current_timestamp, latency=self.latency_swap)]
            # idx_lat_spot = df.index[df.index.searchsorted(self.timestamp - self.latency_spot) - 1]
            # idx_lat_swap = df.index[df.index.searchsorted(self.timestamp - self.latency_swap) - 1]

            if idx_lat_spot is None:
                idx_lat_spot = self.timestamp
            if idx_lat_swap is None:
                idx_lat_swap = self.timestamp
        else:
            idx_lat_spot = self.timestamp
            idx_lat_swap = self.timestamp

        return idx_lat_spot, idx_lat_swap

    def entry_band_fn(self, event):
        return self.df['Entry Band'].iloc[self.idx] + self.entry_band_adjustment

    def exit_band_fn(self, event):
        return self.df['Exit Band'].iloc[self.idx] - self.exit_band_adjustment

    def add_temp_order_to_orders(self, event):
        if self.temporary_order_swap is not None:
            if self.verbose:
                band = self.df.loc[self.timestamp]["Entry Band"] if self.side == 'entry' else \
                self.df.loc[self.timestamp]["Exit Band"]
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Posted swap. Side: {self.side}\t Order id {self.temporary_order_swap.id}\t Order spread {self.temporary_order_swap.targeted_spread}\t Order price {self.temporary_order_swap.price}\tAdded at {self.temporary_order_swap.timestamp_posted}\t Index {self.df.index.get_loc(self.temporary_order_swap.timestamp_posted)}\t Band {band}")
            self.limit_orders_swap.append(self.temporary_order_swap)
            self.temporary_order_swap = None

    def spread_available(self):
        '''
        Find the spread of the known values
        '''

        df = self.df
        # print(df)
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()

        entry_swap = df.loc[idx_lat_swap, 'price_swap_entry']
        exit_swap = df.loc[idx_lat_swap, 'price_swap_exit']
        if self.predicted_depth is None:
            self.predicted_depth = self.depth_posting_predictor.where_to_post(self.timestamp)
        predicted_depth_usd = bp_to_dollars(self.predicted_depth, entry_swap)
        predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)

        entry_spot = df.loc[idx_lat_spot, 'price_spot_entry']
        exit_spot = df.loc[idx_lat_spot, 'price_spot_exit']

        # lat is a timestamp in milliseconds.
        idx = max(idx_lat_spot, idx_lat_swap)
        self.idx = df.index.get_loc(idx)

        self.entry_band = self.entry_band_fn(event=None)
        self.exit_band = self.exit_band_fn(event=None)

        self.spread_entry = get_spread_entry(entry_swap + predicted_depth_usd, entry_spot, swap_fee=self.swap_fee,
                                             spot_fee=self.spot_fee)
        self.spread_exit = get_spread_exit(exit_swap - predicted_depth_usd, exit_spot, swap_fee=self.swap_fee,
                                           spot_fee=self.spot_fee)

        if self.spread_entry >= self.entry_band:
            self.side = 'entry'
        elif self.spread_exit <= self.exit_band:
            self.side = 'exit'
        else:
            self.side = ''

        spread_cond = ((self.spread_entry >= self.entry_band) & (self.side == 'entry')) | \
                      ((self.spread_exit <= self.exit_band) & (self.side == 'exit'))
        # additional check for quanto profit
        if not self.funding_system == 'Quanto_profit':
            return spread_cond
        else:
            if self.minimum_value < self.quanto_loss < self.max_quanto_profit - self.trailing_value:
                self.quanto_profit_triggered = True
            return self.quanto_profit_triggered or spread_cond

    def area_spread(self):
        if (self.side == 'entry') & (self.df.loc[self.timestamp, 'entry_area_spread'] >= self.area_spread_threshold):
            return True
        elif (self.side == 'exit') & (self.df.loc[self.timestamp, 'exit_area_spread'] >= self.area_spread_threshold):
            return True
        else:
            return False

    def rate_limit(self):
        counter = len(self.list_trying_post_counter)
        # if counter >= 10:
        #     print('a')
        if counter <= 1:
            return True
        df_counter = pd.DataFrame(self.list_trying_post_counter, columns=['timestamp', 'side'])
        df_counter['diff_cumsum'] = df_counter.loc[::-1, 'timestamp'].diff(-1).cumsum()[::-1]
        time_diff = df_counter.iloc[-1, 0] - df_counter.iloc[0, 0]

        if self.timestamp - df_counter.iloc[-1, 0] >= 60000:
            self.list_trying_post_counter = [[self.timestamp, self.side]]
            return True

        if time_diff > 60000:
            if df_counter['diff_cumsum'].iloc[-1] <= 60000:
                accounting_for_nan = 1
                del self.list_trying_post_counter[:len(df_counter.loc[df_counter['diff_cumsum'] >= 1000 * 60].index) +
                                                   accounting_for_nan]
            else:
                self.list_trying_post_counter = []
                return True

            df_counter.drop(index=df_counter.loc[df_counter['diff_cumsum'] > 1000 * 60].index, inplace=True)
            df_counter.reset_index(drop=True, inplace=True)

        if len(df_counter) < 30:
            if len(df_counter.loc[df_counter.loc[::-1, 'diff_cumsum'] < 1000]) < 3:
                return True
            else:
                return False
        else:
            return False

    @property
    def try_post_condition(self):
        if not self.spread_available():
            return False
        if self.area_spread_threshold != 0:
            if self.area_spread() & self.rate_limit() & self.trading_condition():
                return True
            else:
                return False
        else:
            if self.rate_limit() & self.trading_condition():
                return True
            else:
                return False

    def update_time_latency_try_post(self, event):
        df = self.df
        self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                          self.latency_try_post) - 1
        self.timestamp = df.index[self.position_current_timestamp]
        # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_try_post) - 1]
        # self.position_current_timestamp = df.index.get_loc[self.timestamp]

    @property
    def try_to_post(self):
        '''
        Condition to exit the trying_to-post state
        This conditions performs the following:
        * Adds a latency to the values.
        * Checks if in this time interval there is a price change in the swap prices
        * returns the side of the execution that will be used later on

        Before this function is called, a signal needs to set the depth of the order
        '''
        df = self.df
        idx_last_spot, idx_last_swap = self.find_known_values_index()
        if self.predicted_depth is None:
            self.predicted_depth = self.depth_posting_predictor.where_to_post(self.timestamp)
        if self.side == 'entry':
            last_known_price_swap = df.at[idx_last_swap, 'price_swap_entry']
            predicted_depth_usd = bp_to_dollars(self.predicted_depth, last_known_price_swap)
            predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)
            posting_price_swap = last_known_price_swap + predicted_depth_usd
            last_known_price_spot = df.at[idx_last_spot, 'price_spot_entry']
            targeted_spread = get_spread_entry(posting_price_swap, last_known_price_spot, self.swap_fee, self.spot_fee)
        elif self.side == 'exit':
            last_known_price_swap = df.at[idx_last_swap, 'price_swap_exit']
            last_known_price_spot = df.at[idx_last_spot, 'price_spot_exit']
            predicted_depth_usd = bp_to_dollars(self.predicted_depth, last_known_price_swap)
            predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)
            posting_price_swap = last_known_price_swap - predicted_depth_usd
            targeted_spread = get_spread_exit(posting_price_swap, last_known_price_spot, self.swap_fee, self.spot_fee)

        a = self.position_current_timestamp
        try:
            b_idx = get_index_right(self.timestamps, self.position_current_timestamp, self.latency_try_post)
        except:
            b_idx = 0
        b = b_idx - 1
        idx_list = df.index[a:b]
        if len(idx_list) > 1:
            idx = 1
            while idx < len(idx_list):
                if (df.loc[idx_list[idx - 1], 'price_swap_entry'] < df.loc[idx_list[idx], 'price_swap_entry']) \
                        & (self.side == 'entry'):
                    return False
                elif (df.loc[idx_list[idx - 1], 'price_swap_exit'] > df.loc[idx_list[idx], 'price_swap_exit']) \
                        & (self.side == 'exit'):
                    return False
                else:
                    idx += 1

            if idx == len(idx_list):
                self.temporary_order_swap = LimitOrder(self.timestamp, price=posting_price_swap, is_executed=False,
                                                       side=self.side, is_spot=False,
                                                       volume_executed=self.max_trade_volume,
                                                       max_targeted_depth=predicted_depth_usd,
                                                       targeted_spread=targeted_spread,
                                                       price_other_exchange=last_known_price_spot,
                                                       id_=np.random.randint(1, 100))

                return True

        else:
            self.temporary_order_swap = LimitOrder(self.timestamp, price=posting_price_swap, is_executed=False,
                                                   side=self.side, is_spot=False,
                                                   volume_executed=self.max_trade_volume,
                                                   max_targeted_depth=predicted_depth_usd,
                                                   targeted_spread=targeted_spread,
                                                   price_other_exchange=last_known_price_spot,
                                                   id_=np.random.randint(1, 100))
            return True

    @property
    def spread_unavailable_post(self):
        '''
        Define if the spread is unavailable or the swap price moves in the wrong direction.
        Return True if spread is unavailable or swap price moved in wrong direction.
        * in entry side wrong direction is previous value larger than current value
        * in exit side wrong direction is previous value smaller than current value
        The spread is calculated on the known values and not on the real values
        '''
        # @TODO need to call the signal generating function first

        df = self.df
        # print(df)
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()

        # idx_c=current value, idx_p=previous value
        idx_c = df.index.get_loc(idx_lat_swap)
        idx_p = idx_c - 1

        entry_swap = df.loc[idx_lat_swap, 'price_swap_entry']
        exit_swap = df.loc[idx_lat_swap, 'price_swap_exit']

        entry_spot = df.loc[idx_lat_spot, 'price_spot_entry']
        exit_spot = df.loc[idx_lat_spot, 'price_spot_exit']

        idx = max(idx_lat_spot, idx_lat_swap)
        self.idx = df.index.get_loc(idx)
        entry_band_posted = self.entry_band_fn(event=None)
        exit_band_posted = self.exit_band_fn(event=None)

        spread_entry = get_spread_entry(self.limit_orders_swap[0].price, entry_spot, swap_fee=self.swap_fee,
                                        spot_fee=self.spot_fee)
        spread_exit = get_spread_exit(self.limit_orders_swap[0].price, exit_spot, swap_fee=self.swap_fee,
                                      spot_fee=self.spot_fee)
        self.limit_orders_swap[0]

        if self.side == 'entry':
            if (spread_entry < entry_band_posted):
                if self.verbose:
                    print(
                        f"Time: {self.df.Time.loc[self.timestamp]}\t Spread no longer available. Side: {self.limit_orders_swap[0].side}\t Order id {self.limit_orders_swap[0].id}\t Is target spot: {False}\t Order spread {self.limit_orders_swap[0].targeted_spread}\t Current spread {spread_entry}\t Current band {entry_band_posted}")

                return True
            else:
                return False
        else:
            if (spread_exit > exit_band_posted):
                if self.verbose:
                    print(
                        f"Time: {self.df.Time.loc[self.timestamp]}\t Spread no longer available. Side: {self.limit_orders_swap[0].side}\t Order id {self.limit_orders_swap[0].id}\t Is target spot: {False}\t Order spread {self.limit_orders_swap[0].targeted_spread}\t Current spread {spread_exit}\t Current band {exit_band_posted}")

                return True
            else:
                return False

    def keep_time_post(self, event):
        if event.transition.dest == 'posted':
            self.time_post = self.timestamp
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]

    def cancel_open_swap(self, event):
        if len(self.limit_orders_swap) > 0:
            if self.verbose:
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Removing order from bookkeeping. Side: {self.limit_orders_swap[0].side}\t Order id {self.limit_orders_swap[0].id}\t Is target spot: {False}\t Order spread {self.limit_orders_swap[0].targeted_spread}")
            order = self.limit_orders_swap.pop()
            order.timestamp_cancelled = self.timestamp
            if not order.is_executed:
                if self.need_to_post_deeper__:
                    order.cancelled_to_post_deeper = True
                self.cancelled_orders_swap.append(order)

    @property
    def swap_price_no_movement(self):
        '''
        When there is no movement in the swap price this function will return True
        In order to archive that we compare the current known value with the previous one
        '''
        return not (self.spread_unavailable_post or self.swap_price_movement_correct)

    @property
    def swap_price_movement_correct(self):
        '''
        Condition to check if we have a movement of the known swap price towards the correct side
        (price increase on entry and price decrease on exit)
        '''
        df = self.df
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        idx_c = df.index.get_loc(idx_lat_swap)
        idx_p = idx_c - 1
        if self.time_post == 0:
            return False

        takers_volume = self.taker_volume_df.loc[
            (self.taker_volume_df['timems'] >= self.time_post) & (self.taker_volume_df['timems'] <= self.timestamp)]

        if (len(takers_volume[takers_volume.side == 'Ask'].index) == 0 and self.side == 'entry') or (
                len(takers_volume[takers_volume.side == 'Bid'].index) == 0 and self.side == 'exit'):
            return False
        order = self.limit_orders_swap[0]

        if self.side == 'entry':
            # price moving up
            if order.price < df['price_swap_entry'].iloc[idx_c]:
                # @TODO removed volume condition
                if len(takers_volume[takers_volume.side == 'Ask'].index) > 0:
                    if len(takers_volume.loc[(takers_volume.side == 'Ask') & (takers_volume.price >= order.price)]) > 0:
                        return True
                    else:
                        return False
            else:
                return False
        else:
            # price moving down
            if order.price > df['price_swap_exit'].iloc[idx_c]:
                # @TODO removed volume condition
                if len(takers_volume[takers_volume.side == 'Bid'].index) > 0:
                    if len(takers_volume.loc[(takers_volume.side == 'Bid') & (takers_volume.price <= order.price)]) > 0:
                        return True
                    else:
                        return False

            else:
                return False

    @property
    def activation_function_cancel(self):
        '''
        We get cancelled when the price swap moves against us in the time interval between now and now+latency
        '''
        df = self.df
        a = self.position_current_timestamp  # df.index.searchsorted(self.timestamp)
        b = get_index_right(self.timestamps, self.position_current_timestamp,
                            self.latency_cancel) - 1  # df.index.searchsorted(self.timestamp + self.latency_cancel) - 1
        idx_list = df.index[a:b]
        if self.time_post == 0:
            return False

        takers_volume = self.taker_volume_df.loc[(self.taker_volume_df['timems'] >= self.time_post) &
                                                 (self.taker_volume_df['timems'] <= df.index[b])]

        if (len(takers_volume[takers_volume.side == 'Ask'].index) == 0 and self.side == 'exit') or \
                (len(takers_volume[takers_volume.side == 'Bid'].index) == 0 and self.side == 'entry'):
            return False

        if len(idx_list) > 1:
            idx = 1
            while idx < len(idx_list):

                if (df.loc[idx_list[idx - 1], 'price_swap_entry'] < df.loc[idx_list[idx], 'price_swap_entry']) \
                        & (self.side == 'entry'):

                    if len(takers_volume[takers_volume.side == 'Bid'].index) > 0:
                        if len(takers_volume.loc[(takers_volume.side == 'Bid') &
                                                 (takers_volume.price >= df.loc[
                                                     self.time_post, 'price_swap_exit'])]) > 0:
                            return True
                        else:
                            return False
                    else:
                        return False


                elif (df.loc[idx_list[idx - 1], 'price_swap_exit'] > df.loc[idx_list[idx], 'price_swap_exit']) \
                        & (self.side == 'exit'):

                    if len(takers_volume[takers_volume.side == 'Ask'].index) > 0:
                        if len(takers_volume.loc[(takers_volume.side == 'Ask') &
                                                 (takers_volume.price <= df.loc[
                                                     self.time_post, 'price_swap_exit'])]) > 0:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    idx += 1

            if idx == len(idx_list):
                return False
        else:
            return False

    # function to compute the swap value that will be used for the computation of the spread

    def move_time_forward(self, event):
        '''
        move time forward from one transition to another when no latency is Included.
        When condition is met compute also the quanto loss
        '''
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]
            # self.timestamp = self.df.index[self.df.index.searchsorted(self.timestamp) + 1]

    def swap_value(self, event):
        '''
        Find the swap price on the exit of the trying to post state in order to use it later in order to define the
        executing spread.
        '''
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        if self.side == 'entry':
            self.swap_price = self.df.loc[idx_lat_swap, 'price_swap_entry']
        else:
            self.swap_price = self.df.loc[idx_lat_swap, 'price_swap_exit']

    def trying_to_post_counter(self, event):
        self.list_trying_post_counter.append([self.timestamp, self.side])

    # On entry and on exit functions of the states
    def posting_f(self, event):
        # print('currently posting')
        df = self.df

        if self.timestamp + self.latency_try_post in df.index:
            self.timestamp = self.timestamp + self.latency_try_post
            self.position_current_timestamp = df.index.get_loc(self.timestamp)
        else:
            # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_try_post) - 1]
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_try_post) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def executing_f(self, event):
        # print('currently executing')
        df = self.df

        if event.transition.source == 'try_cancel':
            latency = self.latency_cancel
        else:
            latency = 0

        if self.timestamp + latency in df.index:
            self.timestamp = self.timestamp + latency
            self.position_current_timestamp = df.index.get_loc(self.timestamp)

        else:
            # self.timestamp = df.index[df.index.searchsorted(self.timestamp + latency) - 1]
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              latency) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        self.source = event.transition.source

        return self.timestamp

    def cancelled_f(self, event):
        '''
        Add latency when entering the cancel state
        '''
        df = self.df
        # print('currently cancelled')
        if self.timestamp + self.latency_cancel in df.index:
            self.timestamp = self.timestamp + self.latency_cancel
            self.position_current_timestamp = df.index.get_loc(self.timestamp)
        else:
            # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_cancel) - 1]
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_cancel) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def spot_balance_f(self, event):
        '''
        Add a spot_balance latency in order to enter the spot_balance state
        '''
        df = self.df
        # print('currently spot balance')
        if self.timestamp + self.latency_spot_balance in df.index:
            self.timestamp = self.timestamp + self.latency_spot_balance
            self.position_current_timestamp = df.index.get_loc(self.timestamp)

        else:
            # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_spot_balance) - 1]
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_spot_balance) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def send_market_order_spot(self, event):
        df = self.df
        idx_last_spot, idx_last_swap = self.find_known_values_index()
        index_timestamp = df.index.get_loc(self.timestamp)
        index_balanced = get_index_right(self.timestamps, index_timestamp, self.latency_spot_balance)

        if self.side == 'entry':
            if self.verbose:
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Spot executed as taker. Side: {self.side}\t Last known price {df['price_spot_entry'].loc[idx_last_spot]}\t Current exchange price {df['price_spot_entry'].iloc[index_timestamp]}\t Balancing price {df['price_spot_entry'].iloc[index_balanced]}\t Index executed {index_balanced}")
            self.taker_execution_spot = TakerExecution(self.timestamps, df.price_spot_entry[idx_last_spot],
                                                       df.iloc[index_balanced].price_spot_entry, self.side, True,
                                                       self.max_trade_volume, event.transition.source,
                                                       event.transition.dest)
            self.taker_execution_spot.timestamp_executed = df.index[index_timestamp]
        elif self.side == 'exit':
            if self.verbose:
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Spot executed as taker. Side: {self.side}\t Last known price {df['price_spot_exit'].loc[idx_last_spot]}\t Current exchange price {df['price_spot_exit'].iloc[index_timestamp]}\t Balancing price {df['price_spot_exit'].iloc[index_balanced]}\t Index executed {index_balanced}")
            self.taker_execution_spot = TakerExecution(self.timestamp, df.price_spot_exit[idx_last_spot],
                                                       df.iloc[index_balanced].price_spot_exit, self.side, True,
                                                       self.max_trade_volume, event.transition.source,
                                                       event.transition.dest)
            self.taker_execution_spot.timestamp_executed = df.index[index_timestamp]

    def compute_final_spread(self, event):
        '''
        Define the final spread:
        The computation uses the Swap price stored in the trying_to_post state and the real value of the Spot price
        '''
        df = self.df

        if self.side == 'entry':
            self.final_spread = get_spread_entry(self.swap_price, df.loc[self.timestamp, 'price_spot_entry'],
                                                 self.swap_fee, self.spot_fee)
        else:
            self.final_spread = get_spread_exit(self.swap_price, df.loc[self.timestamp, 'price_spot_exit'],
                                                self.swap_fee, self.spot_fee)

    def volume_traded(self, event):
        '''
        function to compute the cumulative volume.
        if side = entry we add the traded volume, if the side = exit we subtract the volume.
        '''
        self.traded_volume = self.compute_volume_size()

        if self.side == 'entry':
            self.cum_volume = self.cum_volume + self.traded_volume
        elif self.side == 'exit':
            self.cum_volume = self.cum_volume - self.traded_volume

        self.total_volume_traded = self.total_volume_traded + self.traded_volume

    def quanto_loss_w_avg(self, event):
        '''
        function to compute the average weighted price for Quanto profit-loss.
        The average price is computed when cum_volume > 0  on the entry side,
        and when cum_volume < 0  on the exit side
        '''

        if self.funding_system == 'Quanto_loss' or self.funding_system == 'Quanto_profit':
            self.quanto_loss_func(event)
            price_btc_t = self.price_btc.loc[self.btc_idx, 'price_ask']
            price_eth_t = self.price_eth.loc[self.eth_idx, 'price_ask']
            if self.side == 'entry':
                self.coin_volume = self.coin_volume + self.traded_volume / price_eth_t
            elif self.side == 'exit':
                self.coin_volume = self.coin_volume - self.traded_volume / price_eth_t

            if self.cum_volume > 0 and self.side == 'entry':
                self.w_avg_price_btc = abs((self.w_avg_price_btc * (self.cum_volume - self.traded_volume) +
                                            self.traded_volume * price_btc_t) / self.cum_volume)
                self.w_avg_price_eth = abs((self.w_avg_price_eth * (self.cum_volume - self.traded_volume) +
                                            self.traded_volume * price_eth_t) / self.cum_volume)
            elif self.cum_volume < 0 and self.side == 'exit':
                self.w_avg_price_btc = abs(
                    (self.w_avg_price_btc * (self.cum_volume + self.traded_volume) -
                     self.traded_volume * price_btc_t) / self.cum_volume)
                self.w_avg_price_eth = abs(
                    (self.w_avg_price_eth * (self.cum_volume + self.traded_volume) -
                     self.traded_volume * price_eth_t) / self.cum_volume)

    def quanto_loss_func(self, event):
        '''
        function to compute the quanto profit or quanto loss whenever conditions are met.
        In order to reduce the computation we have set a condition: for the quanto profit to be computed the previous
        computation must have happened some seconds in the past
        '''
        if self.timestamp - self.previous_timestamp >= 5000:
            self.btc_idx = self.price_btc.loc[self.btc_idx:, 'timestamp'].searchsorted(self.timestamp, side='left') + \
                           self.btc_idx
            self.eth_idx = self.price_eth.loc[self.eth_idx:, 'timestamp'].searchsorted(self.timestamp, side='left') + \
                           self.eth_idx
            if self.btc_idx > self.price_btc.index[-1]:
                self.btc_idx = self.price_btc.index[-1]
            if self.eth_idx > self.price_eth.index[-1]:
                self.eth_idx = self.price_eth.index[-1]
            price_btc_t = self.price_btc.loc[self.btc_idx, 'price_ask']
            price_eth_t = self.price_eth.loc[self.eth_idx, 'price_ask']

            self.quanto_loss = quanto_pnl_func(price_eth=price_eth_t, avg_price_eth=self.w_avg_price_eth,
                                               price_btc=price_btc_t, avg_price_btc=self.w_avg_price_btc,
                                               coin_volume=-self.coin_volume)

            if self.cum_volume == 0:
                self.quanto_loss = 0
                self.coin_volume = 0
                self.w_avg_price_btc = 0
                self.w_avg_price_eth = 0
            # set the previous_timestamp to current for the next computation to happen
            # if event.transition.source == 'clear':
            #
            self.previous_timestamp = self.timestamp

        # move entry - exit band when there is quanto loss
        self.idx = self.df.index[self.idx:].searchsorted(self.timestamp, side='left') + self.idx
        if self.quanto_loss < 0 and self.funding_system == 'Quanto_loss':
            self.exit_band_adjustment = self.quanto_loss
            self.df['Exit Band with Quanto loss'].iloc[self.idx] = self.exit_band_fn(event=None)
            if self.df['Entry Band'].iloc[self.idx] - self.df['Exit Band with Quanto loss'].iloc[
                self.idx] <= self.minimum_distance:
                self.entry_band_adjustment = self.exit_band_fn(event=None) - self.df['Entry Band'].iloc[
                    self.idx] + self.minimum_distance
                self.df['Entry Band with Quanto loss'].iloc[self.idx] = self.entry_band_fn(event=None)

        # move entry band when there is quanto profit above the fixed spread
        elif self.quanto_loss - (self.df['Entry Band'].iloc[self.idx] - self.df['Exit Band'].iloc[self.idx]) > 0 \
                and self.funding_system == 'Quanto_profit' and self.trailing_value != 0:
            if self.quanto_loss >= self.disable_when_below:
                self.df['Entry Band with Quanto loss'].iloc[self.idx] = self.df['Entry Band'].iloc[self.idx] + \
                                                                        self.quanto_loss - (self.df['Entry Band'].iloc[
                                                                                                self.idx] -
                                                                                            self.df['Exit Band'].iloc[
                                                                                                self.idx])

    def update_order_after_executed(self, event):
        df = self.df
        idx_last_spot, idx_last_swap = self.find_known_values_index()
        idx_current = df.index.get_loc(idx_last_swap)
        self.limit_orders_swap[0].is_executed = True
        self.limit_orders_swap[0].source_at_execution = event.transition.source
        self.limit_orders_swap[0].dest_at_execution = event.transition.dest
        self.limit_orders_swap[0].timestamp_executed = self.timestamp  # df.index[idx_current]
        if self.verbose:
            print(
                f"Time: {self.df.Time.loc[self.timestamp]}\t Swap executed. Side: {self.side}\t Order id {self.limit_orders_swap[0].id}\t Order price {self.limit_orders_swap[0].price}\t Current price {df['price_swap_entry'].iloc[idx_current]}\t Index executed {idx_current}")
        if event.transition.source == 'try_to_cancel':
            self.limit_orders_swap[0].was_trying_to_cancel = True
            self.executed_while_cancelling_orders_swap.append(self.limit_orders_swap[0])
            if self.verbose:
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Swap executed\t Order id {self.limit_orders_swap[0].id}\t Executed while cancelling")

    def update_executions(self, event):
        # We are assuming that when balancing we won't have two maker orders in our system
        # need to know if we were makers on both sides or not. In case we were takers need to know where we were takers
        df = self.df
        lat = self.timestamp  # Fix this
        entry_band = df.at[lat, "Entry Band"]
        exit_band = df.at[lat, "Exit Band"]
        # print(f"Time: {self.df.Time.loc[self.timestamp]}\tCentral band: {self.df.loc[self.timestamp, 'Central Band']}\tPrice spot ask: {self.df.price_swap_ask.loc[self.timestamp]}\tPrice spot ask:{self.df.price_spot_exit.loc[self.timestamp]}\tPrice swap bid:{self.df.price_swap_bid.loc[self.timestamp]}\tPrice spot bid:{self.df.price_spot_entry.loc[self.timestamp]}")

        assert self.limit_orders_swap[0].timestamp_executed > self.limit_orders_swap[
            0].timestamp_posted, f"Something is wrong. Found order executed before posted. Timestamp posted {self.limit_orders_swap[0].timestamp_posted}"
        execution_swap = self.limit_orders_swap[0]
        targeted_spread = self.limit_orders_swap[0].targeted_spread
        if self.side == 'entry':
            executed_spread = get_spread_entry(self.limit_orders_swap[0].price,
                                               self.taker_execution_spot.executed_price, self.swap_fee, self.spot_fee)
        elif self.side == 'exit':
            executed_spread = get_spread_exit(self.limit_orders_swap[0].price, self.taker_execution_spot.executed_price,
                                              self.swap_fee, self.spot_fee)
        self.executions.append({
            "timems": self.limit_orders_swap[0].timestamp_executed,
            "timestamp_swap_executed": self.limit_orders_swap[0].timestamp_executed,
            "executed_spread": executed_spread,
            "targeted_spread": targeted_spread,
            "order_depth": execution_swap.max_targeted_depth,
            "volume_executed": execution_swap.volume_executed,
            "entry_band": entry_band,
            "exit_band": exit_band,
            "price": execution_swap.price,
            "was_trying_to_cancel_swap": execution_swap.was_trying_to_cancel,
            "source_at_execution_swap": execution_swap.source_at_execution,
            "dest_at_execution_swap": execution_swap.dest_at_execution,
            "side": self.side,
        })
        if self.verbose:
            print(
                f"Time: {self.df.Time.loc[self.timestamp]}\t Execution updated. Number of executions {len(self.executions)}")

        return

    def quanto_trailing_func(self):
        if self.quanto_loss > self.max_quanto_profit and self.quanto_loss >= self.disable_when_below:
            self.max_quanto_profit = self.quanto_loss
        elif self.quanto_loss < self.disable_when_below:
            self.max_quanto_profit = 0

    def trading_condition(self):
        if self.swap_instrument == 'ETHUSD' and self.exchange_swap == 'BitMEX':
            if self.funding_system == 'No' or self.funding_system == 'Quanto_profit':
                if (self.side == 'entry') & (self.cum_volume >= 0):
                    return False
                elif self.side == 'entry':
                    return True
                if (self.side == 'exit') & (self.cum_volume <= - self.max_position):
                    return False
                elif self.side == 'exit':
                    return True
            elif self.funding_system == 'Quanto_loss':
                if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                    return False
                elif self.side == 'entry':
                    return True
                if (self.side == 'exit') & (self.cum_volume <= 0):
                    return False
                elif self.side == 'exit':
                    return True
        else:
            if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                return False
            elif self.side == 'entry':
                return True
            if (self.side == 'exit') & (self.cum_volume <= - self.max_position):
                return False
            elif self.side == 'exit':
                return True


def get_taker_trades(t0, t1, swapMarket, swapSymbol):
    influx_connection = InfluxConnection.getInstance()

    if swapMarket == 'BitMEX' or swapMarket == 'Deribit' or swapMarket == 'FTX' or swapMarket == 'Binance':
        swap_takers_querier = Takers(influx_connection.archival_client_spotswap_dataframe,
                                     [swapMarket], [swapSymbol])
    elif swapMarket == 'HuobiDMSwap':
        swap_takers_querier = Takers(influx_connection.archival_client_spotswap_dataframe,
                                     [swapMarket], [swapSymbol])
    elif swapMarket == 'Okex':
        swap_takers_querier = Takers(influx_connection.staging_client_spotswap_dataframe,
                                     [swapMarket], [swapSymbol])
    else:
        try:
            swap_takers_querier = Takers(influx_connection.archival_client_spotswap_dataframe,
                                         [swapMarket], [swapSymbol])
        except:
            swap_takers_querier = Takers(influx_connection.staging_client_spotswap_dataframe,
                                         [swapMarket], [swapSymbol])
    try:
        return swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
    except KeyError:
        return pd.DataFrame(columns=['side'])


# @st.experimental_singleton
def get_data_for_trader(t_start, t_end, exchange_spot, spot_instrument, exchange_swap, swap_instrument, swap_fee,
                        spot_fee, strategy, area_spread_threshold, environment, band_type,
                        window_size=None, exit_delta_spread=None, entry_delta_spread=None,
                        band_funding_system=None, funding_window=90, generate_percentage_bands=False,
                        lookback=None, recomputation_time=None, target_percentage_exit=None,
                        target_percentage_entry=None, entry_opportunity_source=None, exit_opportunity_source=None,
                        minimum_target=None, use_aggregated_opportunity_points=None, ending=None,
                        force_band_creation=False, move_bogdan_band='No'):
    price1 = get_price(t_start=t_start, t_end=t_end, exchange=exchange_spot, symbol=spot_instrument, side='Ask',
                       environment=environment)

    price2 = get_price(t_start=t_start, t_end=t_end, exchange=exchange_swap, symbol=swap_instrument, side='Ask',
                       environment=environment)

    price1['Time'] = price1.index

    price2['Time'] = price2.index

    # merge the price dataframes
    price_ask = pd.merge_ordered(price1, price2, on='Time', suffixes=['_spot_entry', '_swap_entry'])

    price3 = get_price(t_start=t_start, t_end=t_end, exchange=exchange_spot, symbol=spot_instrument, side='Bid',
                       environment=environment)

    price4 = get_price(t_start=t_start, t_end=t_end, exchange=exchange_swap, symbol=swap_instrument, side='Bid',
                       environment=environment)

    price3['Time'] = price3.index
    price4['Time'] = price4.index
    price_bid = pd.merge_ordered(price3, price4, on='Time', suffixes=['_spot_exit', '_swap_exit'])

    df_price = pd.merge_ordered(price_ask, price_bid, on='Time')
    df_price['price_spot_mid'] = (df_price['price_spot_entry'] + df_price['price_spot_exit']) / 2

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
        if force_band_creation or strategy == '':
            datalink = DatalinkCreateBands(t_start=t_start - 1000 * 60 * (window_size + 10), t_end=t_end,
                                           swap_exchange=exchange_swap, swap_symbol=swap_instrument,
                                           spot_exchange=exchange_spot, spot_symbol=spot_instrument,
                                           window_size=window_size, entry_delta_spread=entry_delta_spread,
                                           exit_delta_spread=exit_delta_spread, swap_fee=swap_fee, spot_fee=spot_fee,
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
            datalink.generate_bogdan_bands()
            time.sleep(5)
            band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands',
                                          strategy=datalink.strategy_name, environment=environment)
        else:
            band_values = get_band_values(t0=t_start, t1=t_end, typeb=band_type,
                                          strategy=strategy, environment=environment)

    elif band_type == 'custom_multi' or band_type == 'custom_multi_symmetrical' or band_type == 'custom_multi_custom':

        datalink = DatalinkCreateBands(t_start=t_start - 1000 * 60 * (window_size + 10), t_end=t_end,
                                       swap_exchange=exchange_swap, swap_symbol=swap_instrument,
                                       spot_exchange=exchange_spot, spot_symbol=spot_instrument,
                                       window_size=window_size, entry_delta_spread=entry_delta_spread,
                                       exit_delta_spread=exit_delta_spread, swap_fee=swap_fee, spot_fee=spot_fee,
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
            datalink.generate_bogdan_bands()
            time.sleep(5)

        band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands',
                                      strategy=datalink.strategy_name, environment=environment)

        try:
            isinstance(band_values, type(None))
            datalink.generate_bogdan_bands()
            time.sleep(5)
        except:
            if band_values.empty:
                datalink.generate_bogdan_bands()
                time.sleep(5)
                band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands',
                                              strategy=datalink.strategy_name, environment=environment)
            elif band_values.iloc[:100, :].dropna().empty or band_values.iloc[100:, :].dropna().empty:
                datalink.generate_bogdan_bands()
                time.sleep(5)
                band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands',
                                              strategy=datalink.strategy_name, environment=environment)

    # add the deribit funding to the band
    if move_bogdan_band != 'No' and exchange_spot == 'Deribit' and (spot_instrument == 'ETH-PERPETUAL' or
                                                                    spot_instrument == 'BTC-PERPETUAL'):
        deribit_funding = funding_values_mark_to_market(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                        symbol=spot_instrument, environment=environment)
        deribit_funding['funding'] = deribit_funding['funding'] / (8 * 3600)
        deribit_funding['rolling_funding'] = deribit_funding['funding'].rolling('8h').mean()
        # deribit_funding['percentual_change'] = (deribit_funding['rolling_funding'] - deribit_funding['funding']) / deribit_funding['funding'] * 100
        deribit_funding['Time'] = deribit_funding.index
        deribit_funding.reset_index(drop=True, inplace=True)
        band_values = pd.merge_ordered(band_values, deribit_funding, on='Time')
        band_values['Entry Band'].ffill(inplace=True)
        band_values['Exit Band'].ffill(inplace=True)
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
    df['Entry Band'].ffill(inplace=True)
    df['Exit Band'].ffill(inplace=True)
    df.price_spot_entry.ffill(inplace=True)
    df.price_swap_entry.ffill(inplace=True)
    df.price_spot_exit.ffill(inplace=True)
    df.price_swap_exit.ffill(inplace=True)
    df.dropna(inplace=True)
    df['timems'] = df.Time.view(np.int64) // 10 ** 6
    df.set_index('timems', drop=False, inplace=True)

    df.drop_duplicates(subset=['timems'], keep='last', inplace=True)

    df['entry_area_spread'] = 0
    df['exit_area_spread'] = 0

    df['spread_entry'] = get_spread_entry(df['price_swap_entry'], df['price_spot_entry'], swap_fee=swap_fee,
                                          spot_fee=spot_fee)
    df['spread_exit'] = get_spread_exit(df['price_swap_exit'], df['price_spot_exit'], swap_fee=swap_fee,
                                        spot_fee=spot_fee)
    if area_spread_threshold != 0:
        df['multiplier'] = df['timems'].diff() // 100

        df_mat = df.loc[:, ['spread_entry', 'spread_exit', 'Entry Band', 'Exit Band', 'multiplier', 'entry_area_spread',
                            'exit_area_spread']].to_numpy()

        df_mat = df_numba(df_mat)

        df['entry_area_spread'] = df_mat[:, 5]
        df['exit_area_spread'] = df_mat[:, 6]
    if band_type == 'custom_multi' or band_type == 'custom_multi_symmetrical' or band_type == 'custom_multi_custom' or \
            strategy == '':
        return df, datalink.strategy_name
    else:
        return df, None


@numba.jit(nopython=True)
def df_numba(df_mat):
    for idx in range(1, len(df_mat) - 1):
        if df_mat[idx, 0] >= df_mat[idx, 2]:
            df_mat[idx, 5] = df_mat[idx - 1, 5] + abs(df_mat[idx, 0] - df_mat[idx, 2]) * df_mat[idx, 4]

        if df_mat[idx, 1] <= df_mat[idx, 3]:
            df_mat[idx, 6] = df_mat[idx - 1, 6] + abs(df_mat[idx, 1] - df_mat[idx, 3]) * df_mat[idx, 4]
    return df_mat


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    done = True
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            done = False
            return j
    if done:
        return -1
