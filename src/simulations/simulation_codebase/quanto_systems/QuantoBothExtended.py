from src.common.utils.quanto_utils import quanto_pnl_func
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty


class QuantoBothExtendedSystem(QuantoSystemEmpty):
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
                 ratio_entry_band_mov, window, ratio_entry_band_mov_ind, ratio_entry_band_mov_long, window_long,
                 ratio_exit_band_mov_ind_long):
        """
        @brief Initializes the position. This is the constructor for the Position class. It should be called before the position is added to the list
        @param price_btc The price of the BTC position in the price_eth. This is the price of the Ethereum position.
        @param price_eth The price of the Ethereum position in the price_btc. This is the price of the Ethereum position.
        @param current_r The current position in the position.
        @param high_r The quanto threshold in the position.
        @param quanto_threshold The quanto threshold in the position.
        @param distance The distance between the position and the end of the band that is to be moved.
        @param high_to_current The high to current position in the position.
        @param ratio_entry_band_mov The ratio of the entry band movement to the current position.
        @param window The rolling time window size. This is the number of days to move in the rolling time window.
        @param ratio_entry_band_mov_ind The ratio of the entry band movement to the indice of the band that is to be moved.
        @param ratio_entry_band_mov_long The ratio of the entry band movement to the long position.
        @param window_long The rolling time window size. This is the number of days to move in the rollovers.
        @param ratio_exit_band_mov_ind_long The ratio of the exit band movement to the long position
        """

        super().__init__(price_btc, price_eth)

        # parameters for when the position is sort
        self.current_r = current_r
        self.high_r = high_r
        self.quanto_threshold = quanto_threshold
        self.high_to_current = high_to_current
        self.ratio_entry_band_mov = ratio_entry_band_mov
        self.minimum_distance = distance
        self.rolling_time_window_size = window
        self.ratio_entry_band_mov_ind = ratio_entry_band_mov_ind

        # parameters for when the position is long
        self.ratio_entry_band_mov_long = ratio_entry_band_mov_long
        self.rolling_time_window_size_long = window_long
        self.ratio_exit_band_mov_ind_long = ratio_exit_band_mov_ind_long

        self.price_btc_p = 0
        self.price_eth_p = 0
        self.btc_idx_p = 0
        self.eth_idx_p = 0

        self.price_btc_p_long = 0
        self.price_eth_p_long = 0
        self.btc_idx_p_long = 0
        self.eth_idx_p_long = 0

    def update(self, timestamp, position):
        """
         @brief Update the data to account for a new timestamp. This is called every time we have an update in the price_btc and price_eth tables
         @param timestamp timestamp of the update in milliseconds
         @param position position of the update in the market_price
        """
        super().update(timestamp, 0)

        self.btc_idx_p = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.btc_idx_p
        self.eth_idx_p = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                       int(self.rolling_time_window_size),
                                                                                       side='left') + self.eth_idx_p

        self.btc_idx_p_long = self.price_btc.loc[self.btc_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                            int(self.rolling_time_window_size_long),
                                                                                            side='left') + self.btc_idx_p_long
        self.eth_idx_p_long = self.price_eth.loc[self.eth_idx_p:, 'timestamp'].searchsorted(timestamp - 1000 * 60 *
                                                                                            int(self.rolling_time_window_size_long),
                                                                                            side='left') + self.eth_idx_p_long

        # Set the price of the btc to the next price.
        if self.btc_idx_p > self.price_btc.index[-1]:
            self.btc_idx_p = self.price_btc.index[-1]
        # Set the price of the eth. index to the next price
        if self.eth_idx_p > self.price_eth.index[-1]:
            self.eth_idx_p = self.price_eth.index[-1]

        # If the price_btc_idx_p_long is greater than the price_btc. index 1 this method will set the price_btc_idx_p_long to the next price_btc. index 1.
        if self.btc_idx_p_long > self.price_btc.index[-1]:
            self.btc_idx_p_long = self.price_btc.index[-1]
        # Set the price_eth index to the next price_eth. index 1
        if self.eth_idx_p_long > self.price_eth.index[-1]:
            self.eth_idx_p_long = self.price_eth.index[-1]

        self.price_btc_p = self.price_btc.loc[self.btc_idx_p, 'price']
        self.price_eth_p = self.price_eth.loc[self.eth_idx_p, 'price']

        self.price_btc_p_long = self.price_btc.loc[self.btc_idx_p_long, 'price']
        self.price_eth_p_long = self.price_eth.loc[self.eth_idx_p_long, 'price']

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        """
         @brief Entry / Exit band adjustment based on quanto_pnl_func. It is called by self. adjust_btc_and_profit ()
         @param entry_band number of entry band to be adjusted
         @param exit_band number of exit band to be adjusted
         @param move_exit if True the exit band is moved
         @param position position of the entry band ( default 0 )
         @return adjusted value of the exit band or None if no adjustment is needed ( exit_adjustment is a function
        """
        # print(f"entry_band_adjustment position: {position}")
        # position is the position of the entry band in the exit band adjustment.
        if position > 0:
            exit_adjustment = self.exit_band_adjustment(entry_band, exit_band, move_exit, position)
            # adjustment of quanto loss in exit_band_adjustment
            if self.ratio_entry_band_mov_ind != 0 or self.rolling_time_window_size != 0:

                volume = 1 / (self.price_btc_p * 0.000001)
                quanto_prof_entry = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                    avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                    coin_volume=volume)

                # print(f'quanto loss in exit_band_adjustment: {quanto_prof_entry}')
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
        else:
            # quanto_loss is the quanto loss value
            if self.quanto_loss > 0:
                # print(f"entry band adjustment long {- self.ratio_entry_band_mov_long * self.quanto_loss}")
                return - self.ratio_entry_band_mov_long * self.quanto_loss
            return 0

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False, position=0):
        """
         @brief This function adjusts the exit band to account for quanto loss. If position is greater than 0 it is used to determine the position of the band adjustment
         @param entry_band the number of the entry band
         @param exit_band the number of the exit band ( 0 means not used )
         @param move_exit whether or not to move the exit
         @param position the position of the band adjustment
         @return the adjustment in the entry band to account for quanto loss or 0 if there is no qu
        """
        # print(f"exit_band_adjustment position: {position}")
        # quanto loss in exit band adjustment
        if position > 0:
            # quanto_loss is the ratio of the quanto ratio entry band
            if self.quanto_loss < 0:
                return self.ratio_entry_band_mov * self.quanto_loss
            return 0
        else:
            # print("exit band adjustment long")
            entry_adjustment = self.entry_band_adjustment(entry_band, exit_band, move_exit, position)
            # adjustment for exit band adjustment.
            if self.ratio_exit_band_mov_ind_long != 0 or self.rolling_time_window_size_long != 0:

                volume = 1 / (self.price_btc_p_long * 0.000001)
                quanto_prof_exit = quanto_pnl_func(avg_price_eth=self.price_eth_p_long,
                                                   price_eth=self.price_eth_t,
                                                   avg_price_btc=self.price_btc_p_long,
                                                   price_btc=self.price_btc_t,
                                                   coin_volume=volume)

                # print(f'quanto loss in exit_band_adjustment: {quanto_prof_exit}')
                adjustment = - self.ratio_exit_band_mov_ind_long * quanto_prof_exit
                # print(f"exit band adjustment when long based on theoretical quanto: {adjustment}")
                condition1 = ((entry_band + entry_adjustment) - (exit_band - adjustment) <= self.minimum_distance)
                # condition1 activated exit band if condition1 is true
                if condition1:
                    # print(f"condition1 activated exit adjustment = {exit_band - (entry_band + entry_adjustment) + self.minimum_distance}")
                    return exit_band - (entry_band + entry_adjustment) + self.minimum_distance
                else:
                    return adjustment

            condition = ((entry_band + entry_adjustment) - exit_band <= self.minimum_distance)
            # condition1 activated exit band if condition is true
            if condition:
                # print(
                #     f"condition1 activated exit adjustment = {exit_band - (entry_band + entry_adjustment) + self.minimum_distance}")
                return exit_band - (entry_band + entry_adjustment) + self.minimum_distance
            else:
                return 0
