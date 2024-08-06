from dataclasses import dataclass

import numpy as np

from src.common.utils.quanto_utils import quanto_pnl_func


class QuantoSystemEmpty:

    w_avg_price_btc = 0
    w_avg_price_eth = 0
    coin_volume = 0
    quanto_loss_pnl = 0
    contracts = 0
    minimum_distance = 0
    quanto_loss = 0
    btc_idx = 0
    eth_idx = 0
    ratio_entry_band_mov = 1.0

    counter = 0

    exp1 = None
    exp2 = None
    exp3 = None

    minimum_value = 0
    trailing_value = 0
    disable_when_below = 0
    max_quanto_profit = 0
    quanto_profit_triggered = False

    def __init__(self, price_btc, price_eth) -> None:
        """
         @brief Initialize the object with the price_btc and price_eth. This is a convenience method for use in tests
         @param price_btc The BTC price of the unit in units of GB. If this is None the unit will be set to 0.
         @param price_eth The Ethernet price of the unit in units of GB. If this is None the unit will be set to 0.
         @return The instance of the class that was initialized with the values given in the parameters. It is possible to get the values without calling this method
        """
        self.price_btc = price_btc
        self.price_eth = price_eth
        self.price_btc_t = 0
        self.price_eth_t = 0

    def update_trade(self, timestamp, cum_volume, execution_side, traded_volume):
        """
         @brief Update trading volume. This is called every time a trade is updated. It takes into account the cumulative volume of the trading volume and the execution side of the trade
         @param timestamp timestamp of the trade in seconds
         @param cum_volume if > 0 the cumulative volume is added
         @param execution_side if > 0 the execution side is added
         @param traded_volume
        """
        self.btc_idx = self.price_btc.loc[self.btc_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.btc_idx
        self.eth_idx = self.price_eth.loc[self.eth_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.eth_idx
        # Set the price index to the next price.
        if self.btc_idx > self.price_btc.index[-1]:
            self.btc_idx = self.price_btc.index[-1]
        # Set the price_eth index to the next price_eth. index 1
        if self.eth_idx > self.price_eth.index[-1]:
            self.eth_idx = self.price_eth.index[-1]

        price_btc_t = self.price_btc.loc[self.btc_idx, 'price']
        price_eth_t = self.price_eth.loc[self.eth_idx, 'price']
        # average price of the volume of the current execution
        if cum_volume > 0 and execution_side == 'entry':
            self.w_avg_price_btc = abs((self.w_avg_price_btc*(cum_volume-traded_volume) +
                                        traded_volume*price_btc_t) / cum_volume)
            self.w_avg_price_eth = abs((self.w_avg_price_eth * (cum_volume - traded_volume) +
                                        traded_volume * price_eth_t) / cum_volume)
        elif cum_volume < 0 and execution_side == 'exit':
            self.w_avg_price_btc = abs(
                (self.w_avg_price_btc * (cum_volume + traded_volume) -
                 traded_volume * price_btc_t) / cum_volume)
            self.w_avg_price_eth = abs(
                (self.w_avg_price_eth * (cum_volume + traded_volume) -
                 traded_volume * price_eth_t) / cum_volume)

        # Calculates the volume of the quanto loss and contracts
        if cum_volume == 0:
            self.quanto_loss = 0
            self.coin_volume = 0
            self.contracts = 0
            self.w_avg_price_btc = 0
            self.w_avg_price_eth = 0
        else:
            self.coin_volume = - cum_volume / self.w_avg_price_eth
            self.contracts = - cum_volume / (self.w_avg_price_eth * self.w_avg_price_btc * 0.000001)


    def update(self, timestamp, position):
        """
         @brief Update price and position. This is called every time a timestamp is updated. The timestamp is used to search for the timestamp in the data and the position is used to update the indices of the btc and eth prices
         @param timestamp The timestamp to search for
         @param position The position of the timestamp in the data ( 0 - based
        """
        # print(f"self.btc_idx: {self.btc_idx}, self.eth_idx: {self.eth_idx}")
        self.btc_idx = self.price_btc.loc[self.btc_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.btc_idx
        self.eth_idx = self.price_eth.loc[self.eth_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.eth_idx
        # Set the price index to the next price.
        if self.btc_idx > self.price_btc.index[-1]:
            self.btc_idx = self.price_btc.index[-1]
        # Set the price_eth index to the next price_eth. index 1
        if self.eth_idx > self.price_eth.index[-1]:
            self.eth_idx = self.price_eth.index[-1]

        self.price_btc_t = self.price_btc.loc[self.btc_idx, 'price']
        self.price_eth_t = self.price_eth.loc[self.eth_idx, 'price']
        # quanto_loss is the loss of the quanto_loss function.
        if self.coin_volume != 0 or self.contracts != 0:
            self.quanto_loss = quanto_pnl_func(price_eth=self.price_eth_t, avg_price_eth=self.w_avg_price_eth,
                                               price_btc=self.price_btc_t, avg_price_btc=self.w_avg_price_btc,
                                               coin_volume=abs(self.contracts / self.coin_volume) * np.sign(self.contracts))
        else:
            self.quanto_loss = 0
        self.quanto_loss_pnl = self.quanto_loss * abs(self.coin_volume)

    def update_exponential_quanto(self, exp1):
        """
         @brief Update the quanto exponent. This is a no - op for non - quadratic quantities.
         @param exp1 The new exponent to update. Must be a float between 0 and 1.
         @return None. If an error occurs None is returned to indicate success. The error can be caused by a variety of reasons : 1. The number of trials is greater than the number of successes. 2. The number of trials is equal
        """
        return None

    def allow_posting(self, side):
        """
         @brief Whether or not this poster can post a message. By default this is False. Subclasses may override this to return True for posting messages that are to be sent to the given side.
         @param side Side to which the message is to be sent.
         @return True if this poster can post a message False otherwise. If side is None it will return False
        """
        return False

    def band_adjustments(self, entry_band, exit_band, move_exit, position):
        """
         @brief Calculate the adjustments to be applied to the entry and exit band. This is a wrapper around : meth : ` ~chainer. Chainer. band_adjustment ` for use in subclasses
         @param entry_band The entry band to adjust.
         @param exit_band The exit band to adjust. Can be None if there is no exit.
         @param move_exit Whether or not to move the exit to the position.
         @param position The position in the tree. Can be None if there is no position.
         @return A tuple of : class : ` ~chainer. adjustments. BandAdjustments ` that are applied
        """
        return self.entry_band_adjustment(entry_band, exit_band, move_exit, position), \
               self.exit_band_adjustment(entry_band, exit_band, move_exit, position)

    def entry_band_adjustment(self, e, ex, me, pos):
        """
         @brief Called by L { entry_band_adjustment } to adjust the band. This is a no - op in this case.
         @param e : class : ` ~compas. datastructures. Entry ` entry of the band being adjusted.
         @param ex : class : ` ~compas. datastructures. Entry ` entry of the band being adjusted.
         @param me : class : ` ~conkit. core. structure. Structure ` structure of the band being adjusted.
         @param pos : class : ` ~conkit. core. structure. Structure. Position ` position of the band in the structure.
         @return an integer indicating the amount of adjustment to be made to get to the entry band at the given position
        """
        return 0

    def exit_band_adjustment(self, e, ex, me, pos):
        """
         @brief Called when an exit band is removed. Subclasses should override this to do any adjustment that needs to be done to the exit band before it is removed.
         @param e The exit band's error. This is a : class : ` excpy. exc. Exposure ` object.
         @param ex The exit band's error. This is a : class : ` excpy. exc. Exposure ` object.
         @param me The exit band's error. This is a : class : ` excpy. exc. MeatizedExit ` object.
         @param pos The position of the exit band in the input.
         @return The number of errors to be added to the exit band. For example if you want to make a decision on the size of the exit band then call this method
        """
        return 0

@dataclass
class PriceBoxParams():
    basis_points: int = None
    upper_threshold: int = None
    lower_threshold: int = None
    aggr_window: int = None
    span: int = None
    entry_movement_ratio: int = 0
    t0 = None
    t1 = None
    upper_threshold_crossed = False
    lower_threshold_crossed = False
    idx_p = 0

    def __bool__(self):
        """
         @brief Returns True if basis points are set. This is useful for debugging. The result of __bool__ should be cached for performance reasons but it's possible to get a False result for some of the tests in this class.
         @return True if basis points are set False otherwise ( default : False ) >>> from sympy. polys
        """
        # Returns true if the basis points are set.
        if self.basis_points is not None:
            return True
        return False


class QuantoLossSystem(QuantoSystemEmpty):
    """
    Input variables:
    price_btc =  dataframe that contains the BTC ask prices from BitMEX
    price_eth = dataframe that contains the ETH ask prices from BitMEX
    ratio_entry_band_mov = the ratio of the band movement
    current_r =  the value of the ratio for the low volatility period
    high_r = the value of the ratio for the high volatility period
    quanto_threshold = value that activates the transition between current and high ratios
    high_to_current = boolean that revert from high to current after 8hours
    """
    # boolean variable to initiate the procedure for stopping the trading
    stop_trading_enabled = False
    # boolean variable to flag the stop trading timestamp
    halt_trading_flag = False

    def __init__(self, price_btc, price_eth, current_r, high_r, quanto_threshold, distance, high_to_current,
                 ratio_entry_band_mov,  window, ratio_entry_band_mov_ind):
        """
        @brief Initialize the object. This is the method that will be called by the class when it is instantiated.
        @param price_btc The price of the btc price.
        @param price_eth The price of the eth price.
        @param current_r The current rate of the exchange.
        @param high_r The high rate of the exchange.
        @param quanto_threshold The quanto threshold for the exchange.
        @param distance The distance between the current and the high rate.
        @param high_to_current The high to current rate.
        @param ratio_entry_band_mov The rolling time window size.
        @param window The rolling time window size. It is an integer.
        @param ratio_entry_band_mov_ind The rolling time window size
        """
        super().__init__(price_btc, price_eth)
        self.current_r = current_r
        self.high_r = high_r
        self.quanto_threshold = quanto_threshold
        self.high_to_current = high_to_current
        self.ratio_entry_band_mov = ratio_entry_band_mov
        self.minimum_distance = distance
        self.rolling_time_window_size = window
        self.ratio_entry_band_mov_ind = ratio_entry_band_mov_ind
        self.price_btc_p = 0
        self.price_eth_p = 0
        self.btc_idx_p = 0
        self.eth_idx_p = 0
        self.exp1 = None
        self.exp2 = None

    def update(self, timestamp, position):
        """
         @brief Update the index of btc and eth. This is called every time a timestamp is added or removed from the data frame
         @param timestamp timestamp of the update to be done
         @param position position of the update in the trading time
        """
        super().update(timestamp, 0)

        self.btc_idx_p = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.btc_idx_p
        self.eth_idx_p = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.eth_idx_p
        # print(f"index now {self.eth_idx}, previous index {self.eth_idx_p}")
        # Set the price of the btc to the next price.
        if self.btc_idx_p > self.price_btc.index[-1]:
            self.btc_idx_p = self.price_btc.index[-1]
        # Set the price of the eth. index to the next price
        if self.eth_idx_p > self.price_eth.index[-1]:
            self.eth_idx_p = self.price_eth.index[-1]

        self.price_btc_p = self.price_btc.loc[self.btc_idx_p, 'price']
        self.price_eth_p = self.price_eth.loc[self.eth_idx_p, 'price']

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        """
         @brief This function adjusts the entry band to the exit band. It is called by the : meth : ` run_thermodynamic ` method
         @param entry_band The band to be adjusted
         @param exit_band The band to be adjusted
         @param move_exit If the exit band should be moved
         @param position The position of the band ( default 0 )
         @return The adjustment : math : ` \ Psi ` that is applied to the entry band : math : ` \ Psi
        """
        exit_adjustment = self.exit_band_adjustment(entry_band, exit_band, move_exit)

        # adjustment of quanto loss in exit_band_adjustment
        if self.ratio_entry_band_mov_ind != 0 or self.rolling_time_window_size != 0:

            volume = 1 / (self.price_btc_p * 0.000001)
            quanto_prof_entry = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                coin_volume=volume)

            # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
            adjustment = self.ratio_entry_band_mov_ind * quanto_prof_entry

            condition1 = ((entry_band + adjustment) - (exit_band - exit_adjustment) <= self.minimum_distance)
            # The adjustment of the entry band
            if condition1:
                return (exit_band - exit_adjustment) - entry_band + self.minimum_distance
            else:
                return adjustment

        condition = (entry_band - (exit_band - exit_adjustment) <= self.minimum_distance)
        # Return the distance between the exit and entry band
        if condition:
            return (exit_band - exit_adjustment) - entry_band + self.minimum_distance
        else:
            return 0

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        """
         @brief Adjust the exit band to account for quanto loss. This is used to adjust the band that is going to be moved and vice versa
         @param entry_band The number of entries in the entry band
         @param exit_band The number of exits in the exit band
         @param move_exit If True the exit band is moved
         @param position The position of the band ( default 0 )
         @return The amount to adjust the exit band to account for quanto loss or 0 if there is no
        """
        # quanto_loss is the ratio of the quanto ratio entry band
        if self.quanto_loss < 0:
            return self.ratio_entry_band_mov * self.quanto_loss
        return 0


# quanto_profit_systems = {"QuantoProfitSystem": QuantoProfitSystem, "QuantoProfitBuildQuanto": QuantoProfitBuildQuanto}

