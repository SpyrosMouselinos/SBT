from src.common.utils.quanto_utils import get_price_box_signal, quanto_pnl_func
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty, PriceBoxParams


class QuantoProfitBuildQuanto(QuantoSystemEmpty):
    def __init__(self, price_btc, price_eth, distance, perc_entry, perc_exit, minimum_value, trailing_value, below,
                 window, price_box_params):
        """
        @brief Initialize the class. This is the constructor for the QuantoTPC object. It sets the minimum distance between the btc and eth price and the percent of the band that is used to determine the price of the event
        @param price_btc The price of the btc price
        @param price_eth The price of the eth price ( float )
        @param distance The distance between the btc and eth price ( float )
        @param perc_entry The percentage of the entry band to the price of the event ( float )
        @param perc_exit The percentage of the exit band to the price of the event ( float )
        @param minimum_value The minimum value of the price of the event ( float )
        @param trailing_value The trailing value of the price of the event ( float )
        @param below The boolean indicating if the event is below or above the price
        @param window The rolling time window size ( int )
        @param price_box_params The parameters of the price
        """
        super().__init__(price_btc, price_eth)
        self.minimum_distance = distance
        self.ratio_entry_band_mov = perc_entry
        self.ratio_exit_band_mov = perc_exit
        self.minimum_value = minimum_value
        self.trailing_value = trailing_value
        self.disable_when_below = below
        self.rolling_time_window_size = window
        self.price_btc_p = 0
        self.price_eth_p = 0
        self.btc_idx_p = 0
        self.eth_idx_p = 0
        self.last_exit_band_adjustment = 0
        self.last_entry_band_adjustment = 0
        self.price_box_params: PriceBoxParams = price_box_params
        # computes the quanto tp box signal
        if self.price_box_params:
            print("Computing quanto tp box signal")
            self.quanto_tp_signal = get_price_box_signal(self.price_box_params.t0-1000*60*int(self.rolling_time_window_size),
            self.price_box_params.t1,
            self.price_box_params.basis_points,
            self.price_box_params.aggr_window,
            self.price_box_params.span)
            self.quanto_tp_signal['timems'] = self.quanto_tp_signal['end_time'].view(np.int64) // 10 ** 6
            self.quanto_tp_signal['triggered'] = False

    def update(self, timestamp, position):
        """
         @brief Update data to account for time window. This is called every time a timestamp is added or removed from the price_btc and price_eth tables
         @param timestamp timestamp of the update in milliseconds
         @param position position of the update in millimet
        """
        super(QuantoProfitBuildQuanto, self).update(timestamp, position)
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

        # This method is used to determine if the quanto_tp_signal is within the threshold threshold.
        if self.price_box_params:
            self.price_box_params.idx_p = \
                self.quanto_tp_signal.loc[self.price_box_params.idx_p:, 'timems'].searchsorted(timestamp, side='left') \
                + self.price_box_params.idx_p
            # If the price box signal is greater than the quanto_tp_signal. index 1 this. quanto_tp_signal. index 1
            if self.price_box_params.idx_p >= self.quanto_tp_signal.index[-1]:
                self.price_box_params.idx_p = self.quanto_tp_signal.index[-1]

            # If the quanto_tp_signal. ioc self. quanto_tp_signal. iloc self. quanto_tp_signal. iloc self. quanto_tp_signal. iloc self. price_box_params. idx_p. signal is less than upper threshold then the lower threshold is crossed.
            if self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal > self.price_box_params.upper_threshold:
                self.price_box_params.upper_threshold_crossed = True
            # if lower_threshold_crossed and upper_threshold_crossed are true then the lower threshold crossed is set to false
            if self.price_box_params.upper_threshold_crossed and \
                    self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal < self.price_box_params.lower_threshold:
                self.price_box_params.lower_threshold_crossed = False
                self.price_box_params.upper_threshold_crossed = False
            # If the quanto_tp_signal is too low then trigger the signal.
            if self.price_box_params.lower_threshold_crossed:
                # If the price box has been triggered by the quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc
                if self.price_box_params.idx_p < len(self.quanto_tp_signal):
                    self.quanto_tp_signal.loc[self.price_box_params.idx_p, 'triggered'] = True

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
         @brief Adjust entry / exit band for quanto profit. This is based on the difference between the price of the ethereum and bought cash price.
         @param entry_band The band to adjust for entry.
         @param exit_band The band to adjust for exit.
         @param move_exit If True the exit is moved to the end of the band.
         @return The amount of adjustments to apply to the box in order to achieve the desired change in the entry band
        """
        volume = 1 / (self.price_btc_p * 0.000001)
        quanto_loss_virtual = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                              avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                              coin_volume=volume)
        box_movement = 0
        additional_quanto_profit = max(quanto_loss_virtual - (entry_band - exit_band),0)
        # The box movement ratio of the quanto profit.
        if self.price_box_params and self.price_box_params.upper_threshold_crossed:
            box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
        # + self.ratio_entry_band_mov * additional_quanto_profit
        return box_movement

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
         @brief Adjust the exit band to account for the fact that it is in the same direction as the entry band.
         @param entry_band The amount to adjust the entry band to.
         @param exit_band The amount to adjust the exit band to.
         @param move_exit If True the exit band will be subtracted from the entry band.
         @return The difference between the entry and exit band adjustments or 0 if there is no change in the entry or
        """
        # Move the exit band to the exit band.
        if move_exit:
            entry_adjustment = self.entry_band_adjustment(entry_band, exit_band)
            # The minimum distance between the entry and exit band.
            if abs(entry_adjustment) >= 0.05:
                return - (entry_band + entry_adjustment - exit_band) + self.minimum_distance
            return - entry_adjustment
        else:
            return 0