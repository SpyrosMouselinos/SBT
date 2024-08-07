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
        """
        @brief Initialize the Bayesian filter. This is the method to use when the filter is created in order to determine the price of the filter's entry and exit band.
        @param price_btc The price of the first price of the filter.
        @param price_eth The price of the last price of the filter.
        @param distance The minimum distance between the two price bands.
        @param window The rolling time window size. Default is 0.
        @param window2 The rolling time window size. Default is 0.
        @param exp1 The exponent of the first price band. Default is 0.
        @param exp2 The exponent of the second price band. Default is 0.
        @param exp3 The exponent of the third price band. Default is 0.
        @param cap_entry_pos The capacitance of the entry band. Default is 0.
        @param cap_entry_neg The capacitance of the entry band. Default is 0.
        @param cap_exit_pos The capacitance of the exit band. Default is 0.
        @param weight1 The weight of the first price band. Default is 0.
        @param weight2 The weight of the second price band. Default is 0
        """
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

        # If exit_upper_bound is greater than entry_upper_bound then the exit_upper_bound is set to 0. 95 * entry_upper_bound.
        if self.exit_upper_bound >= self.entry_upper_bound:
            self.exit_upper_bound = 0.95 * self.entry_upper_bound

        self.weight_theoretical_quanto_entry = weight1
        self.weight_real_quanto_entry = weight2

    def update(self, timestamp, position):
        """
         @brief Update the index based on a timestamp. This is called by : meth : ` ~sklearn. metrics. BaseHarmonicMetric. update `
         @param timestamp The timestamp to update the index for.
         @param position The position of the price in the price_
        """
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
        # Set the price of the btc to the next price.
        if self.btc_idx_p > self.price_btc.index[-1]:
            self.btc_idx_p = self.price_btc.index[-1]
        # Set the price of the eth. index to the next price
        if self.eth_idx_p > self.price_eth.index[-1]:
            self.eth_idx_p = self.price_eth.index[-1]
        # If the price is greater than the price_btc. index 1 then the price_btc_idx_p2 is set to the highest price_btc index 1.
        if self.btc_idx_p2 > self.price_btc.index[-1]:
            self.btc_idx_p2 = self.price_btc.index[-1]
        # price_eth. index 1.
        if self.eth_idx_p2 > self.price_eth.index[-1]:
            self.eth_idx_p2 = self.price_eth.index[-1]

        self.price_btc_p = self.price_btc.loc[self.btc_idx_p, 'price']
        self.price_eth_p = self.price_eth.loc[self.eth_idx_p, 'price']
        self.price_btc_p2 = self.price_btc.loc[self.btc_idx_p2, 'price']
        self.price_eth_p2 = self.price_eth.loc[self.eth_idx_p2, 'price']

    def update_exponential_quanto(self, exp1=0):
        """
         @brief Update Exponential Quanto Loss. This is a wrapper for : func : ` qutip. functions. quanto_pnl_func_exp ` that takes care of the exp_entry_real and exp_entry_eth and coin_volume as well as the rest of the parameters.
         @param exp1 exponent to use for the quanto
        """
        # quanto_loss_exp Quanto loss_exp price_eth price_eth avg_price_eth avg_price_eth avg_price_eth avg_price_btc exp_entry_real exp_entry_real
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
        """
         @brief Adjust the entry and exit values based on the band adjustments. This is the function that should be called by the user when they want to adjust the entry and exit values.
         @param entry_band The amount of band to adjust the entry value.
         @param exit_band The amount of band to adjust the exit value.
         @param move_exit True if the exit should be moved to the entry.
         @return A tuple of two values : The entry value as a float. The exit value as a float. The entry value is negative if the entry is closer to the exit
        """
        entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
        exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
        # If the exit band is below the minimum distance between the entry and exit band adjustment then the exit band is adjusted to the minimum distance.
        if not self.flag:
            self.flag = True
            entry = exit_band + self.last_exit_band_adjustment - entry_band + self.minimum_distance
            # The entry band of the entry.
            if entry_band + entry >= self.entry_upper_bound + (entry_band + exit_band) / 2:
                entry = self.entry_upper_bound + (entry_band + exit_band) / 2 - entry_band
            # If entry_band entry_band entry exit_band exit
            if entry_band + entry <= exit_band + exit:
                exit = -(entry_band + entry - exit_band - self.minimum_distance)

        return entry, exit

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
        @brief Entry band adjustment for BTC. It is used to adjust the weight of the theoretical band to account for quanto profit
        @param entry_band entry band to be adjusted
        @param exit_band exit band to be adjusted ( default 0 )
        @param move_exit move exits to the higher band ( default False )
        @return adjusted entry band and exit band ( float or None ) depending on move_exit and / or
        """

        volume = 1 / (self.price_btc_p * 0.000001)
        # profit theoretical entry theoretical of the Quanto profit
        if not (self.exp_entry_theoretical is None and pd.isna(self.exp_entry_theoretical)):
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
        # adjusts the entry band to the entry band
        if entry_band + entry_adjustment >= self.entry_upper_bound + (entry_band + exit_band) / 2:
            entry_adjustment = self.entry_upper_bound + (entry_band + exit_band) / 2 - entry_band
        elif entry_band + entry_adjustment <= self.entry_lower_bound:
            entry_adjustment = self.entry_lower_bound + (entry_band + exit_band) / 2 - entry_band

        self.last_entry_band_adjustment = entry_adjustment
        return entry_adjustment

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
         @brief This function adjusts the exit band to account for the change in the band. If there is an exit band the adjustment is based on quanto_pnl_func_exp
         @param entry_band the band that is used to enter the entry price
         @param exit_band the band that is used to exit the exit price
         @param move_exit whether or not to move the exit price
         @return the adjusted exit band
        """
        volume = 1 / (self.price_btc_p2 * 0.000001)
        # The exit band adjustment for the exit theoretical.
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

        # adjust exit band to the end of the exit band
        if exit_band + exit_band_adjustment >= self.exit_upper_bound + (entry_band + exit_band) / 2:
            exit_band_adjustment = self.exit_upper_bound + (entry_band + exit_band) / 2 - exit_band

        # if entry_band entry_band and exit_band_adjustment are less than 0 then the exit_band is adjusted to the minimum distance between entry_band and exit_band
        if entry_band + self.last_entry_band_adjustment <= exit_band + exit_band_adjustment \
                and self.last_entry_band_adjustment < 0:
            exit_band_adjustment = entry_band + self.last_entry_band_adjustment - exit_band - self.minimum_distance
        elif self.last_entry_band_adjustment >= 0:
            self.flag = False
            self.last_exit_band_adjustment = exit_band_adjustment

        # the formula for the exit band new value in the main program is exit_band - exit_band_adjustment,
        # so if we want the exit to move up we have to return a negative number.
        return - exit_band_adjustment
