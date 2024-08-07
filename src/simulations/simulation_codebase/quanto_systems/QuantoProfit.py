import numpy as np

from src.common.utils.quanto_utils import get_price_box_signal, quanto_pnl_func
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty, PriceBoxParams


class QuantoProfitSystem(QuantoSystemEmpty):
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

    def __init__(self, price_btc, price_eth, distance, perc_entry, perc_exit, minimum_value, trailing_value, below,
                 window, price_box_params):
        """
        @brief Initializes the object. This is the method that should be called by the user
        @param price_btc The price of the btc to use. It is in BTC format. If this is None the price will be calculated from the eth price.
        @param price_eth The price of the eth to use. It is in ETH format. If this is None the price will be calculated from the btc price
        @param distance The distance between the two halves of the price
        @param perc_entry The percentage of the entry band that will be adjusted by this method. This is in ETH format. If this is None the price will be calculated from the btc price
        @param perc_exit The percentage of the exit band that will be adjusted by this method. If this is None the price will be calculated from the eth price
        @param minimum_value The minimum value that will be added to the price
        @param trailing_value The trailing value that will be added to the price
        @param below The boolean that indicates if the window is below or above
        @param window The rolling time window size ( in seconds ). This is in ETH format. If this is None the price will be calculated from the eth price
        @param price_box_params The parameters that will be passed to the price box
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
        self.flag = True
        self.exp1 = None
        self.exp2 = None

        # computes the quanto tp box signal
        if self.price_box_params:
            print("Computing quanto tp box signal")
            self.quanto_tp_signal = get_price_box_signal(self.price_box_params.t0,
                                                         self.price_box_params.t1,
                                                         self.price_box_params.basis_points,
                                                         self.price_box_params.aggr_window,
                                                         self.price_box_params.span)
            self.quanto_tp_signal['timems'] = self.quanto_tp_signal['end_time'].view(np.int64) // 10 ** 6
            self.quanto_tp_signal['triggered'] = False

    def update(self, timestamp, position):
        """
         @brief Update the state of the market. This is called every time we are interested in a position ( buy sell etc. )
         @param timestamp timestamp of the trade in milliseconds
         @param position position of the trade in millimet
        """
        super().update(timestamp, 0)
        # Set the max_quanto_trailing_func to the max_quanto_trailing_func if the coin volume or contracts are not zero.
        if self.coin_volume != 0 or self.contracts != 0:
            self.max_quanto_profit = self.quanto_trailing_func()

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

        # This method is used to calculate the price box parameters.
        if self.price_box_params:
            self.price_box_params.idx_p = \
                self.quanto_tp_signal.loc[self.price_box_params.idx_p:, 'timems']. \
                    searchsorted(timestamp, side='left') + self.price_box_params.idx_p

            # If the price box signal is greater than the quanto_tp_signal. index 1 this. quanto_tp_signal. index 1
            if self.price_box_params.idx_p >= self.quanto_tp_signal.index[-1]:
                self.price_box_params.idx_p = self.quanto_tp_signal.index[-1]

            # If the quanto_tp_signal. ioc self. quanto_tp_signal. iloc self. quanto_tp_signal. iloc self. quanto_tp_signal. iloc self. price_box_params. idx_p. signal is less than upper threshold then the lower threshold is crossed.
            if self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal > self.price_box_params.upper_threshold:
                self.price_box_params.upper_threshold_crossed = True
            # If the quanto_tp_signal is less than the lower threshold the lower threshold is crossed.
            if self.price_box_params.upper_threshold_crossed and self.quanto_tp_signal. \
                    iloc[self.price_box_params.idx_p].signal < self.price_box_params.lower_threshold:
                self.price_box_params.lower_threshold_crossed = True
                self.price_box_params.upper_threshold_crossed = False
            # If position is 0 then the price box is considered to be crossed.
            if position == 0:
                self.price_box_params.upper_threshold_crossed = False
                self.price_box_params.lower_threshold_crossed = False
            # If the quanto_tp_signal is too low then trigger the signal.
            if self.price_box_params.lower_threshold_crossed:
                # If the price box has been triggered by the quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc self. quanto_tp_signal. loc
                if self.price_box_params.idx_p < len(self.quanto_tp_signal):
                    self.quanto_tp_signal.loc[self.price_box_params.idx_p, 'triggered'] = True

    def band_adjustments(self, entry_band, exit_band, move_exit):
        """
         @brief Adjust entry and exit bands. This is the method that performs the band adjustments.
         @param entry_band The list of entry band adjustments.
         @param exit_band The list of exit band adjustments.
         @param move_exit Whether or not to move the exit.
         @return A tuple of : class : ` menpo. image. Image `
        """
        # Move exit band and exit band adjustments.
        if move_exit:
            entry_adjustment_before = self.last_entry_band_adjustment
            entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
            exit_adjustment_before = self.last_exit_band_adjustment
            exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
            # Returns entry band adjustment for the entry and exit band.
            if entry_adjustment_before is not self.last_entry_band_adjustment or exit_adjustment_before \
                    is not self.last_exit_band_adjustment:
                entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
                exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
            return entry, exit
        else:
            entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
            exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
            return entry, exit

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
        @brief Adjust the entry and exit band to account for quanto loss. This is called by
        @param entry_band The band at which the entry is located.
        @param exit_band The band at which the exit is located.
        @param move_exit If True the exit band is moved.
        @return The amount to adjust the entry and exit band to
        """

        # adjustment for quanto quanto_loss entry_band exit_band exit_band
        if move_exit:
            adjustment = 0
            # Calculates the quanto loss for the quanto_loss.
            if self.quanto_loss - (entry_band - exit_band) > 0 and self.trailing_value != 0 and \
                    self.quanto_loss >= self.disable_when_below:
                box_movement = 0
                additional_quanto_profit = self.quanto_loss - (entry_band - exit_band)
                # The box movement ratio of the quanto profit.
                if self.price_box_params and self.price_box_params.lower_threshold_crossed:
                    box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
                adjustment = box_movement + self.ratio_entry_band_mov * additional_quanto_profit

            # adjustment is the adjustment between the exit band and the exit band
            if abs(adjustment) < 0.05:
                self.flag = False
                # adjustment of the exit band to the last exit band
                if exit_band - self.exit_band_adjustment(entry_band, exit_band, move_exit) > entry_band + adjustment:
                    adjustment = -self.last_exit_band_adjustment - (entry_band - exit_band) + self.minimum_distance
            else:
                self.flag = True
            self.last_entry_band_adjustment = adjustment
            return adjustment
        else:
            adjustment = 0
            # Calculates the quanto loss for the quanto_loss.
            if self.quanto_loss - (entry_band - exit_band) > 0 and self.trailing_value != 0 and \
                    self.quanto_loss >= self.disable_when_below:
                box_movement = 0
                additional_quanto_profit = self.quanto_loss - (entry_band - exit_band)
                # The box movement ratio of the quanto profit.
                if self.price_box_params and self.price_box_params.lower_threshold_crossed:
                    box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
                adjustment = box_movement + self.ratio_entry_band_mov * additional_quanto_profit
            return adjustment

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        """
         @brief Exit band adjustment for quanto. It is used to adjust the exit band to a higher or equal value.
         @param entry_band The band that is at the entry of the band
         @param exit_band The band that is at the exit of the band
         @param move_exit If True the exit band is moved to the higher value. If False the exit band is not moved. If the value is less than the entry band it is moved to the higher
         @return The difference between the entry and exit band adjustments
        """
        # movement exit band adjustment.
        if move_exit:
            # Calculate the movement exit band.
            if exit_band > entry_band + self.last_entry_band_adjustment and self.flag:
                movement_exit = -  self.last_entry_band_adjustment + self.minimum_distance
                self.last_exit_band_adjustment = movement_exit
                return movement_exit

            # quanto_pnl_func average_price_eth avg_price_eth avg_price_eth avg_price_eth_p price_eth avg_price_eth_t price_eth avg_price_eth_t price_eth_p price_eth_t average_price_eth_p price_eth_t average_price_eth_t average_price_eth_p average_price_eth_p average_price_eth_t average_price_eth_p average_price_eth_t price_eth_t price_eth_t price_eth_t price_eth_btc_btc_btc_btc_btc_t average_btc_btc_btc_
            if self.ratio_exit_band_mov != 0:
                volume = - 1 / (self.price_btc_p * 0.000001)
                quanto_loss_exit = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                   avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                   coin_volume=volume)

                # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
                movement_exit = self.ratio_exit_band_mov * quanto_loss_exit
                # print(f'movement exit in exit_band_adjustment: {movement_exit}')
                # Calculate the movement between entry and exit band
                if abs(self.last_entry_band_adjustment) > 0.05:
                    movement_to_entry = - (entry_band + self.last_entry_band_adjustment - exit_band
                                           - self.minimum_distance)
                    movement_exit = max(movement_exit + self.minimum_distance, movement_to_entry)
            else:
                movement_exit = 0
            self.last_exit_band_adjustment = movement_exit
            return movement_exit
        else:
            # The minimum distance between entry and exit band.
            if exit_band > entry_band + self.entry_band_adjustment(entry_band, exit_band):
                return - self.entry_band_adjustment(entry_band, exit_band) + self.minimum_distance

            # quanto_pnl_func average_price_eth avg_price_eth avg_price_eth_p price_eth avg_price_eth_t price_eth avg_price_eth_p price_eth_t price_eth_t average_price_eth_p price_eth_t average_price_eth_t average_price_eth_t average_price_eth_p price_eth_t average_price_eth_p price_eth_p price_eth_p price_eth_t price_eth_t average_price_eth_btc_btc_btc_btc_t price_eth_t price_eth_btc_btc_t average_
            if self.ratio_exit_band_mov != 0:
                volume = - 1 / (self.price_btc_p * 0.000001)
                quanto_loss_exit = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                   avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                   coin_volume=volume)
                # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
                movement_exit = self.ratio_exit_band_mov * quanto_loss_exit
                # print(f'movement exit in exit_band_adjustment: {movement_exit}')
                # Return the movement exit band.
                if exit_band - movement_exit < entry_band + self.entry_band_adjustment(entry_band, exit_band):
                    return movement_exit
                else:
                    movement_exit = - (entry_band + self.entry_band_adjustment(entry_band, exit_band) - exit_band
                                       - self.minimum_distance)
                    # print(f'movement exit in exit_band_adjustment else condition: {movement_exit}')
                    return movement_exit
            else:
                return 0

    def quanto_trailing_func(self):
        """
         @brief This function is used to determine quanto profit. The quanto profit is based on the max quanto loss and the disable_when_below value.
         @return The quanto profit as a float between 0 and
        """
        # max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_profit max_quanto_loss max_quanto_loss max_quanto_loss max_quanto_loss max_quanto_loss max_quanto_profit
        if self.quanto_loss > self.max_quanto_profit and self.quanto_loss >= self.disable_when_below:
            max_quanto_profit = self.quanto_loss
        elif self.quanto_loss < self.disable_when_below:
            max_quanto_profit = 0
        else:
            max_quanto_profit = self.max_quanto_profit
        return max_quanto_profit

    def allow_posting(self, side):
        """
         @brief Checks if posting is allowed. This is a callback for : meth : ` get_posting `.
         @param side The side of the posting. Can be'left'or'right '.
         @return True if profit is allowed False otherwise. The return value is used to determine if the posting should be allowed
        """
        # If the quanto_profit is less than the minimum value and the maximum value is less than the maximum value the quanto_loss and the maximum value is less than the maximum value.
        if self.minimum_value < self.quanto_loss < self.max_quanto_profit - self.trailing_value:
            self.quanto_profit_triggered = True
        else:
            self.quanto_profit_triggered = False
        return self.quanto_profit_triggered
