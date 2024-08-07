from src.common.utils.quanto_utils import quanto_pnl_func
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty


class QuantoBothSystem(QuantoSystemEmpty):
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
                 ratio_entry_band_mov, window, ratio_entry_band_mov_ind):
        """
        @brief Initialize the object. This is the method that will be called by the class when it is instantiated.
        @param price_btc The price of the btc price.
        @param price_eth The price of the eth price.
        @param current_r The current rate of the market.
        @param high_r The high rate of the market.
        @param quanto_threshold The quanto threshold for the market.
        @param distance The distance between the market and the current price.
        @param high_to_current The high to current rate ( mean price ).
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
         @brief Adjust the exit band to account for quanto loss. This is a wrapper around the : meth : ` ratio_entry_band_mov ` method.
         @param entry_band The band to be adjusted for the entry.
         @param exit_band The band to be adjusted for the exit.
         @param move_exit If True the exit will be moved to the bottom of the band.
         @param position The position of the exit. Default is 0.
         @return The amount of adjustment to be applied to the exit band. 0 is returned if there is no quanto loss
        """
        # quanto_loss is the quanto loss of the exit band adjustment
        if self.quanto_loss < 0:
            try:
                return self.ratio_entry_band_mov * self.quanto_loss
            except:
                print(f'quanto loss in exit_band_adjustment: {self.quanto_loss}')
                print(f'ratio entry band mov: {self.ratio_entry_band_mov}')
        return 0
