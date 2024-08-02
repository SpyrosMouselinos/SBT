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
        self.price_btc = price_btc
        self.price_eth = price_eth
        self.price_btc_t = 0
        self.price_eth_t = 0

    def update_trade(self, timestamp, cum_volume, execution_side, traded_volume):
        self.btc_idx = self.price_btc.loc[self.btc_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.btc_idx
        self.eth_idx = self.price_eth.loc[self.eth_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.eth_idx
        if self.btc_idx > self.price_btc.index[-1]:
            self.btc_idx = self.price_btc.index[-1]
        if self.eth_idx > self.price_eth.index[-1]:
            self.eth_idx = self.price_eth.index[-1]

        price_btc_t = self.price_btc.loc[self.btc_idx, 'price']
        price_eth_t = self.price_eth.loc[self.eth_idx, 'price']
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
        # print(f"self.btc_idx: {self.btc_idx}, self.eth_idx: {self.eth_idx}")
        self.btc_idx = self.price_btc.loc[self.btc_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.btc_idx
        self.eth_idx = self.price_eth.loc[self.eth_idx:, 'timestamp'].searchsorted(timestamp, side='left') + \
                       self.eth_idx
        if self.btc_idx > self.price_btc.index[-1]:
            self.btc_idx = self.price_btc.index[-1]
        if self.eth_idx > self.price_eth.index[-1]:
            self.eth_idx = self.price_eth.index[-1]

        self.price_btc_t = self.price_btc.loc[self.btc_idx, 'price']
        self.price_eth_t = self.price_eth.loc[self.eth_idx, 'price']
        if self.coin_volume != 0 or self.contracts != 0:
            self.quanto_loss = quanto_pnl_func(price_eth=self.price_eth_t, avg_price_eth=self.w_avg_price_eth,
                                               price_btc=self.price_btc_t, avg_price_btc=self.w_avg_price_btc,
                                               coin_volume=abs(self.contracts / self.coin_volume) * np.sign(self.contracts))
        else:
            self.quanto_loss = 0
        self.quanto_loss_pnl = self.quanto_loss * abs(self.coin_volume)

    def update_exponential_quanto(self, exp1):
        return None

    def allow_posting(self, side):
        return False

    def band_adjustments(self, entry_band, exit_band, move_exit, position):
        return self.entry_band_adjustment(entry_band, exit_band, move_exit, position), \
               self.exit_band_adjustment(entry_band, exit_band, move_exit, position)

    def entry_band_adjustment(self, e, ex, me, pos):
        return 0

    def exit_band_adjustment(self, e, ex, me, pos):
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
        super().update(timestamp, 0)

        self.btc_idx_p = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.btc_idx_p
        self.eth_idx_p = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.eth_idx_p
        # print(f"index now {self.eth_idx}, previous index {self.eth_idx_p}")
        if self.btc_idx_p > self.price_btc.index[-1]:
            self.btc_idx_p = self.price_btc.index[-1]
        if self.eth_idx_p > self.price_eth.index[-1]:
            self.eth_idx_p = self.price_eth.index[-1]

        self.price_btc_p = self.price_btc.loc[self.btc_idx_p, 'price']
        self.price_eth_p = self.price_eth.loc[self.eth_idx_p, 'price']

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        exit_adjustment = self.exit_band_adjustment(entry_band, exit_band, move_exit)

        if self.ratio_entry_band_mov_ind != 0 or self.rolling_time_window_size != 0:

            volume = 1 / (self.price_btc_p * 0.000001)
            quanto_prof_entry = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                coin_volume=volume)

            # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
            adjustment = self.ratio_entry_band_mov_ind * quanto_prof_entry

            condition1 = ((entry_band + adjustment) - (exit_band - exit_adjustment) <= self.minimum_distance)
            if condition1:
                return (exit_band - exit_adjustment) - entry_band + self.minimum_distance
            else:
                return adjustment

        condition = (entry_band - (exit_band - exit_adjustment) <= self.minimum_distance)
        if condition:
            return (exit_band - exit_adjustment) - entry_band + self.minimum_distance
        else:
            return 0

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        if self.quanto_loss < 0:
            return self.ratio_entry_band_mov * self.quanto_loss
        return 0


# quanto_profit_systems = {"QuantoProfitSystem": QuantoProfitSystem, "QuantoProfitBuildQuanto": QuantoProfitBuildQuanto}

