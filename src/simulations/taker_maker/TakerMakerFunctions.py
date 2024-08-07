import numba
import pandas as pd
from src.common.utils.quanto_utils import quanto_pnl_func
from src.common.utils.utils import bp_to_dollars
from src.common.connections.DatabaseConnections import InfluxConnection
import numpy as np
from src.common.queries.queries import Takers

from pytictoc import TicToc

import warnings

warnings.filterwarnings("ignore")
t = TicToc()
t.tic()


def get_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the spread for entering a trade.

    This function computes the effective spread when entering a trade, considering fees.

    @param entry_swap: The entry price on the swap market.
    @param entry_spot: The entry price on the spot market.
    @param swap_fee: The transaction fee for the swap market.
    @param spot_fee: The transaction fee for the spot market.

    @return: The calculated entry spread.
    """
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


def get_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the spread for exiting a trade.

    This function computes the effective spread when exiting a trade, considering fees.

    @param exit_swap: The exit price on the swap market.
    @param exit_spot: The exit price on the spot market.
    @param swap_fee: The transaction fee for the swap market.
    @param spot_fee: The transaction fee for the spot market.

    @return: The calculated exit spread.
    """
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


class LimitOrder:
    """
    A class representing a limit order.

    This class encapsulates the properties and behaviors of a limit order in a trading system.
    """

    def __init__(self, timestamp_posted, price, is_executed, side, is_spot, targeted_spread, volume_executed,
                 max_targeted_depth, cancelled_to_post_deeper=False, timestamp_cancelled=None,
                 price_other_exchange=None, was_trying_to_cancel=False, id_=1, source_at_execution=None,
                 dest_at_execution=None, amount=0):
        """
        Initializes a new instance of the LimitOrder class.

        @param timestamp_posted: The timestamp when the order was posted.
        @param price: The price of the order.
        @param is_executed: A boolean indicating whether the order has been executed.
        @param side: The side of the order, either 'Bid' or 'Ask'.
        @param is_spot: A boolean indicating whether the order is a spot order.
        @param targeted_spread: The targeted spread for the order.
        @param volume_executed: The volume executed for the order.
        @param max_targeted_depth: The maximum targeted depth for the order.
        @param cancelled_to_post_deeper: A boolean indicating whether the order was cancelled to post deeper.
        @param timestamp_cancelled: The timestamp when the order was cancelled.
        @param price_other_exchange: The price of the order on the other exchange.
        @param was_trying_to_cancel: A boolean indicating whether the order was attempting to be cancelled.
        @param id_: The ID of the order.
        @param source_at_execution: The source at execution for the order.
        @param dest_at_execution: The destination at execution for the order.
        @param amount: The amount of the order.
        """
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
    """
    A class representing a taker execution.

    This class encapsulates the properties and behaviors of a taker execution in a trading system.
    """

    def __init__(self, timestamp_posted, targeted_price, executed_price, side, is_spot, volume_executed,
                 source_at_execution=None, dest_at_execution=None, amount=0):
        """
        Initializes a new instance of the TakerExecution class.

        @param timestamp_posted: The timestamp when the execution was posted.
        @param targeted_price: The targeted price for the execution.
        @param executed_price: The executed price for the execution.
        @param side: The side of the execution, either 'Bid' or 'Ask'.
        @param is_spot: A boolean indicating whether the execution is a spot execution.
        @param volume_executed: The volume executed for the execution.
        @param source_at_execution: The source at execution for the order.
        @param dest_at_execution: The destination at execution for the order.
        @param amount: The amount of the execution.
        """
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
    """
    A class encapsulating functions for taker and maker orders.

    This class provides methods to manage and execute taker and maker orders in a trading strategy.
    """

    def set_order_depth(self, new_depth):
        """
        Sets the order depth for the trading strategy.

        This function updates the order depth to the specified value.

        @param new_depth: The new order depth to be set.
        """
        # @TODO decide the unit of the order depth
        self.order_depth = new_depth

    def reset_depth(self, event):
        """
        Resets the predicted depth.

        This function resets the predicted depth to None when triggered by an event.

        @param event: The event that triggers the depth reset.
        """
        self.predicted_depth = None

    def reset_boolean_depth(self, event):
        """
        Resets the boolean flag for posting deeper.

        This function resets the flag indicating the need to post deeper when triggered by an event.

        @param event: The event that triggers the boolean depth reset.
        """
        self.need_to_post_deeper__ = False

    def is_order_too_deep(self, event):
        """
        Checks if the order is too deep in the order book.

        This function determines if the order is too deep in the order book based on the targeted depth.

        @param event: The event that triggers the check.

        @return: True if the order is too deep, False otherwise.
        """
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
        """
        Determines if there is a need to post orders deeper in the order book.

        This function predicts the depth to post orders at and checks if it is greater than the current depth.

        @return: True if there is a need to post deeper, False otherwise.
        """
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
        """
        Computes the volume size for trades.

        This function calculates the traded volume based on the maximum trade volume.

        @return: The computed traded volume.
        """
        traded_volume = self.max_trade_volume
        return traded_volume

    def find_known_values_index(self):
        """
        Finds the index of known spot and swap prices.

        This function determines the indices of known spot and swap prices, considering latency.

        @return: A tuple containing the indices of known spot and swap prices.
        """
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
        """
        Computes the entry band value.

        This function calculates the entry band value based on the dataframe and entry band adjustment.

        @param event: The event triggering the computation.

        @return: The calculated entry band value.
        """
        return self.df['Entry Band'].iloc[self.idx] + self.entry_band_adjustment

    def exit_band_fn(self, event):
        """
        Computes the exit band value.

        This function calculates the exit band value based on the dataframe and exit band adjustment.

        @param event: The event triggering the computation.

        @return: The calculated exit band value.
        """
        return self.df['Exit Band'].iloc[self.idx] - self.exit_band_adjustment

    def add_temp_order_to_orders(self, event):
        """
        Adds a temporary order to the list of orders.

        This function adds a temporary order to the list of swap limit orders and resets the temporary order.

        @param event: The event triggering the addition.
        """
        if self.temporary_order_swap is not None:
            if self.verbose:
                band = self.df.loc[self.timestamp]["Entry Band"] if self.side == 'entry' else \
                    self.df.loc[self.timestamp]["Exit Band"]
                print(
                    f"Time: {self.df.Time.loc[self.timestamp]}\t Posted swap. Side: {self.side}\t Order id {self.temporary_order_swap.id}\t Order spread {self.temporary_order_swap.targeted_spread}\t Order price {self.temporary_order_swap.price}\tAdded at {self.temporary_order_swap.timestamp_posted}\t Index {self.df.index.get_loc(self.temporary_order_swap.timestamp_posted)}\t Band {band}")
            self.limit_orders_swap.append(self.temporary_order_swap)
            self.temporary_order_swap = None

    def spread_available(self):
        """
        Determines if the spread is available for trading.

        This function checks if the spread meets the criteria for entry or exit, considering various factors.

        @return: True if the spread is available, False otherwise.
        """
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
        """
        Determines if the area spread is above the threshold.

        This function checks if the area spread is above the specified threshold for entry or exit.

        @return: True if the area spread is above the threshold, False otherwise.
        """
        if (self.side == 'entry') & (self.df.loc[self.timestamp, 'entry_area_spread'] >= self.area_spread_threshold):
            return True
        elif (self.side == 'exit') & (self.df.loc[self.timestamp, 'exit_area_spread'] >= self.area_spread_threshold):
            return True
        else:
            return False

    def rate_limit(self):
        """
        Determines if the rate limit has been exceeded.

        This function checks if the rate limit for posting orders has been exceeded based on the time difference.

        @return: True if the rate limit is within bounds, False otherwise.
        """
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
        """
        Determines if the conditions for posting an order are met.

        This function checks if the conditions for posting an order are met, considering spread and rate limits.

        @return: True if the conditions are met, False otherwise.
        """
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
        """
        Updates the timestamp with the latency for trying to post.

        This function updates the timestamp and position with the latency for trying to post an order.

        @param event: The event triggering the update.
        """
        df = self.df
        self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                          self.latency_try_post) - 1
        self.timestamp = df.index[self.position_current_timestamp]
        # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_try_post) - 1]
        # self.position_current_timestamp = df.index.get_loc[self.timestamp]

    @property
    def try_to_post(self):
        """
        Determines if the conditions for trying to post an order are met.

        This function checks if the conditions for trying to post an order are met, considering various factors.

        @return: True if the conditions are met, False otherwise.
        """
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
        """
        Determines if the spread is unavailable for posting.

        This function checks if the spread is unavailable or if the swap price moves in the wrong direction.

        @return: True if the spread is unavailable, False otherwise.
        """
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
        """
        Keeps track of the posting time for an order.

        This function updates the posting time and current timestamp during the 'posted' state.

        @param event: The event triggering the time update.
        """
        if event.transition.dest == 'posted':
            self.time_post = self.timestamp
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]

    def cancel_open_swap(self, event):
        """
        Cancels an open swap order.

        This function removes an open swap order from the list of orders and updates its status.

        @param event: The event triggering the cancellation.
        """
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
        """
        Determines if there is no movement in the swap price.

        This function checks if there is no movement in the swap price based on spread availability and movement.

        @return: True if there is no movement, False otherwise.
        """
        return not (self.spread_unavailable_post or self.swap_price_movement_correct)

    @property
    def swap_price_movement_correct(self):
        """
        Determines if the swap price movement is in the correct direction.

        This function checks if the swap price is moving in the correct direction based on the order side.

        @return: True if the movement is correct, False otherwise.
        """
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
        """
        Determines if a cancellation should be activated.

        This function checks if the conditions for cancellation are met based on swap price movement.

        @return: True if cancellation should be activated, False otherwise.
        """
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

    def move_time_forward(self, event):
        """
        Moves the timestamp forward.

        This function updates the timestamp and current position to move time forward during transitions.

        @param event: The event triggering the time update.
        """
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]
            # self.timestamp = self.df.index[self.df.index.searchsorted(self.timestamp) + 1]

    def swap_value(self, event):
        """
        Computes the swap price value.

        This function finds the swap price for the given side during the trying to post state.

        @param event: The event triggering the computation.
        """
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        if self.side == 'entry':
            self.swap_price = self.df.loc[idx_lat_swap, 'price_swap_entry']
        else:
            self.swap_price = self.df.loc[idx_lat_swap, 'price_swap_exit']

    def trying_to_post_counter(self, event):
        """
        Updates the counter for trying to post.

        This function appends the current timestamp and side to the list of trying to post attempts.

        @param event: The event triggering the update.
        """
        self.list_trying_post_counter.append([self.timestamp, self.side])

    def posting_f(self, event):
        """
        Updates the timestamp during the posting state.

        This function adjusts the timestamp and position during the posting state transition.

        @param event: The event triggering the update.

        @return: The updated timestamp.
        """
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
        """
        Updates the timestamp during the executing state.

        This function adjusts the timestamp and position during the executing state transition.

        @param event: The event triggering the update.

        @return: The updated timestamp.
        """
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
        """
        Updates the timestamp during the cancelled state.

        This function adjusts the timestamp and position during the cancelled state transition.

        @param event: The event triggering the update.

        @return: The updated timestamp.
        """
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
        """
        Updates the timestamp during the spot balance state.

        This function adjusts the timestamp and position during the spot balance state transition.

        @param event: The event triggering the update.

        @return: The updated timestamp.
        """
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
        """
        Sends a market order for the spot market.

        This function executes a market order for the spot market, updating the taker execution details.

        @param event: The event triggering the market order.
        """
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
        """
        Computes the final spread for the trade.

        This function calculates the final spread using the swap price stored during the trying to post state.

        @param event: The event triggering the spread computation.
        """
        df = self.df

        if self.side == 'entry':
            self.final_spread = get_spread_entry(self.swap_price, df.loc[self.timestamp, 'price_spot_entry'],
                                                 self.swap_fee, self.spot_fee)
        else:
            self.final_spread = get_spread_exit(self.swap_price, df.loc[self.timestamp, 'price_spot_exit'],
                                                self.swap_fee, self.spot_fee)

    def volume_traded(self, event):
        """
        Updates the cumulative volume traded.

        This function computes the cumulative volume traded and updates the total volume based on the trade side.

        @param event: The event triggering the volume update.
        """
        self.traded_volume = self.compute_volume_size()

        if self.side == 'entry':
            self.cum_volume = self.cum_volume + self.traded_volume
        elif self.side == 'exit':
            self.cum_volume = self.cum_volume - self.traded_volume

        self.total_volume_traded = self.total_volume_traded + self.traded_volume

    def quanto_loss_w_avg(self, event):
        """
        Computes the average weighted price for Quanto profit-loss.

        This function calculates the average weighted price for Quanto profit-loss based on the trade side and volume.

        @param event: The event triggering the computation.
        """
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
        """
        Computes the Quanto profit or loss.

        This function calculates the Quanto profit or loss based on the conditions and updates relevant values.

        @param event: The event triggering the computation.
        """
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
        """
        Updates the order status after execution.

        This function updates the status and attributes of an executed order in the swap limit orders.

        @param event: The event triggering the update.
        """
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
        """
        Updates the execution details.

        This function records the execution details in the list of executions, updating relevant attributes.

        @param event: The event triggering the update.
        """
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
        """
        Updates the maximum Quanto profit.

        This function adjusts the maximum Quanto profit based on the current loss and conditions.
        """
        if self.quanto_loss > self.max_quanto_profit and self.quanto_loss >= self.disable_when_below:
            self.max_quanto_profit = self.quanto_loss
        elif self.quanto_loss < self.disable_when_below:
            self.max_quanto_profit = 0

    def trading_condition(self):
        """
        Determines if trading conditions are met.

        This function checks if the trading conditions are satisfied based on instrument type and funding system.

        @return: True if trading conditions are met, False otherwise.
        """
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
    """
    Retrieves taker trades from the specified swap market.

    This function queries the specified swap market for taker trades within the given time range.

    @param t0: The start time for the query.
    @param t1: The end time for the query.
    @param swapMarket: The swap market to query trades from.
    @param swapSymbol: The symbol of the asset to query trades for.

    @return: A DataFrame containing the retrieved taker trades.
    """
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


@numba.jit(nopython=True)
def df_numba(df_mat):
    """
    Performs calculations on the input matrix using Numba.

    This function calculates the entry and exit area spread based on the provided matrix.

    @param df_mat: The input matrix containing spread and band values.

    @return: The modified matrix with updated area spread values.
    """
    for idx in range(1, len(df_mat) - 1):
        if df_mat[idx, 0] >= df_mat[idx, 2]:
            df_mat[idx, 5] = df_mat[idx - 1, 5] + abs(df_mat[idx, 0] - df_mat[idx, 2]) * df_mat[idx, 4]

        if df_mat[idx, 1] <= df_mat[idx, 3]:
            df_mat[idx, 6] = df_mat[idx - 1, 6] + abs(df_mat[idx, 1] - df_mat[idx, 3]) * df_mat[idx, 4]
    return df_mat


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    """
    Finds the index of the timestamp to the left within the specified latency.

    This function searches for the index of the timestamp to the left within the given latency.

    @param timestamps: The list of timestamps.
    @param current_index: The current index.
    @param latency: The latency to consider.

    @return: The index of the timestamp to the left.
    """
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    """
    Finds the index of the timestamp to the right within the specified latency.

    This function searches for the index of the timestamp to the right within the given latency.

    @param timestamps: The list of timestamps.
    @param current_index: The current index.
    @param latency: The latency to consider.

    @return: The index of the timestamp to the right, or -1 if not found.
    """
    done = True
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            done = False
            return j
    if done:
        return -1
