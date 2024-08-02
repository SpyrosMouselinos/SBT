import numpy as np
import pandas as pd

from src.common.utils.quanto_utils import quanto_pnl_func_exp
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty


class QuantoProfitSystemExponential(QuantoSystemEmpty):
    """
    Input variables:
    price_btc =  dataframe that contains the BTC ask prices from BitMEX
    price_eth = dataframe that contains the ETH ask prices from BitMEX
    distance = the minimum distance allowed between the entry exit band
    perc_entry = the ratio of the entry band movement (ratio_entry_band_mov)
    perc_exit = the ratio of exit band movement
    minimum_value=  the value over the quanto profit is allowed to be taken
    trailing_value = the value of the drop of the quanto profit
    below = value below of which quanto profit mechanism is deactivated.
    window =  the size of the rolling window lookback in the computation of the quanto profit
   """

    def __init__(self, price_btc, price_eth, distance=0, window=0, window2=0, exp1=0, exp2=0, exp3=0,
                 cap_entry_pos=0, cap_entry_neg=0, cap_exit_pos=0, weight1=0, weight2=0):
        super().__init__(price_btc, price_eth)
        self.minimum_distance = distance
        self.rolling_time_window_size = window
        self.rolling_time_window_size2 = window2
        self.price_btc_p = 0
        self.price_eth_p = 0
        self.price_btc_p2 = 0
        self.price_eth_p2 = 0
        self.btc_idx_p = 0
        self.eth_idx_p = 0
        self.btc_idx_p2 = 0
        self.eth_idx_p2 = 0
        self.last_exit_band_adjustment = 0
        self.last_entry_band_adjustment = 0
        self.flag = True
        self.exp_entry_real = exp1
        self.exp_entry_theoretical = exp2
        self.exp_exit_theoretical = exp3

        self.entry_upper_bound = cap_entry_pos
        self.entry_lower_bound = cap_entry_neg
        self.exit_upper_bound = cap_exit_pos

        if self.exit_upper_bound >= self.entry_upper_bound:
            self.exit_upper_bound = 0.95 * self.entry_upper_bound

        self.weight_theoretical_quanto_entry = weight1
        self.weight_real_quanto_entry = weight2

    def update(self, timestamp, position):
        super().update(timestamp, 0)

        self.btc_idx_p = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.btc_idx_p
        self.eth_idx_p = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.eth_idx_p
        self.btc_idx_p2 = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size2),
                                                                                       side='left') + self.btc_idx_p2
        self.eth_idx_p2 = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size2),
                                                                                       side='left') + self.eth_idx_p2
        # print(f"index now {self.eth_idx}, previous index {self.eth_idx_p}")
        if self.btc_idx_p > self.price_btc.index[-1]:
            self.btc_idx_p = self.price_btc.index[-1]
        if self.eth_idx_p > self.price_eth.index[-1]:
            self.eth_idx_p = self.price_eth.index[-1]
        if self.btc_idx_p2 > self.price_btc.index[-1]:
            self.btc_idx_p2 = self.price_btc.index[-1]
        if self.eth_idx_p2 > self.price_eth.index[-1]:
            self.eth_idx_p2 = self.price_eth.index[-1]

        self.price_btc_p = self.price_btc.loc[self.btc_idx_p, 'price']
        self.price_eth_p = self.price_eth.loc[self.eth_idx_p, 'price']
        self.price_btc_p2 = self.price_btc.loc[self.btc_idx_p2, 'price']
        self.price_eth_p2 = self.price_eth.loc[self.eth_idx_p2, 'price']

    def update_exponential_quanto(self, exp1=0):
        if (self.coin_volume != 0 or self.contracts != 0) and (self.exp_entry_real is not None):
            self.quanto_loss_exp = quanto_pnl_func_exp(price_eth=self.price_eth_t, avg_price_eth=self.w_avg_price_eth,
                                                       price_btc=self.price_btc_t, avg_price_btc=self.w_avg_price_btc,
                                                       coin_volume=abs(self.contracts / self.coin_volume) * np.sign(
                                                           self.contracts),
                                                       exp1=self.exp_entry_real, exp2=self.exp_entry_real)
        elif self.exp_entry_real is None:
            self.quanto_loss_exp = None
        else:
            self.quanto_loss_exp = 0

    def band_adjustments(self, entry_band, exit_band, move_exit):
        entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
        exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
        if not self.flag:
            self.flag = True
            entry = exit_band + self.last_exit_band_adjustment - entry_band + self.minimum_distance
            if entry_band + entry >= self.entry_upper_bound + (entry_band + exit_band) / 2:
                entry = self.entry_upper_bound + (entry_band + exit_band) / 2 - entry_band
            if entry_band + entry <= exit_band + exit:
                exit = -(entry_band + entry - exit_band - self.minimum_distance)

        return entry, exit

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):

        volume = 1 / (self.price_btc_p * 0.000001)
        if not(self.exp_entry_theoretical is None and pd.isna(self.exp_entry_theoretical)):
            entry_quanto_profit_theoretical = quanto_pnl_func_exp(avg_price_eth=self.price_eth_p,
                                                                  price_eth=self.price_eth_t,
                                                                  avg_price_btc=self.price_btc_p,
                                                                  price_btc=self.price_btc_t,
                                                                  coin_volume=volume,
                                                                  exp1=self.exp_entry_theoretical,
                                                                  exp2=self.exp_entry_theoretical)
        else:
            entry_quanto_profit_theoretical = 0

        entry_adjustment = self.weight_theoretical_quanto_entry * entry_quanto_profit_theoretical - \
                           self.weight_real_quanto_entry * self.quanto_loss

        # this has to be the final check for the entry band adjustment
        if entry_band + entry_adjustment >= self.entry_upper_bound + (entry_band + exit_band) / 2:
            entry_adjustment = self.entry_upper_bound + (entry_band + exit_band) / 2 - entry_band
        elif entry_band + entry_adjustment <= self.entry_lower_bound:
            entry_adjustment = self.entry_lower_bound + (entry_band + exit_band) / 2 - entry_band

        self.last_entry_band_adjustment = entry_adjustment
        return entry_adjustment

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        volume = 1 / (self.price_btc_p2 * 0.000001)
        if not (self.exp_exit_theoretical is None or pd.isna(self.exp_exit_theoretical)):
            exit_band_adjustment = quanto_pnl_func_exp(avg_price_eth=self.price_eth_p2,
                                                                 price_eth=self.price_eth_t,
                                                                 avg_price_btc=self.price_btc_p2,
                                                                 price_btc=self.price_btc_t,
                                                                 coin_volume=volume,
                                                                 exp1=self.exp_exit_theoretical,
                                                                 exp2=self.exp_exit_theoretical)
        else:
            exit_band_adjustment = 0

        if exit_band + exit_band_adjustment >= self.exit_upper_bound + (entry_band + exit_band) / 2:
            exit_band_adjustment = self.exit_upper_bound + (entry_band + exit_band) / 2 - exit_band

        if entry_band + self.last_entry_band_adjustment <= exit_band + exit_band_adjustment \
                and self.last_entry_band_adjustment < 0:
            exit_band_adjustment = entry_band + self.last_entry_band_adjustment - exit_band - self.minimum_distance
        elif self.last_entry_band_adjustment >= 0:
            self.flag = False
            self.last_exit_band_adjustment = exit_band_adjustment

        # the formula for the exit band new value in the main program is exit_band - exit_band_adjustment,
        # so if we want the exit to move up we have to return a negative number.
        return - exit_band_adjustment