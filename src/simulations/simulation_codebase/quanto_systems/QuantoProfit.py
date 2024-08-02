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
        super().update(timestamp, 0)
        if self.coin_volume != 0 or self.contracts != 0:
            self.max_quanto_profit = self.quanto_trailing_func()

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

        if self.price_box_params:
            self.price_box_params.idx_p = \
                self.quanto_tp_signal.loc[self.price_box_params.idx_p:, 'timems'].\
                    searchsorted(timestamp, side='left') + self.price_box_params.idx_p

            if self.price_box_params.idx_p >= self.quanto_tp_signal.index[-1]:
                self.price_box_params.idx_p = self.quanto_tp_signal.index[-1]

            if self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal > self.price_box_params.upper_threshold:
                self.price_box_params.upper_threshold_crossed = True
            if self.price_box_params.upper_threshold_crossed and self.quanto_tp_signal.\
                    iloc[self.price_box_params.idx_p].signal < self.price_box_params.lower_threshold:
                self.price_box_params.lower_threshold_crossed = True
                self.price_box_params.upper_threshold_crossed = False
            if position == 0:
                self.price_box_params.upper_threshold_crossed = False
                self.price_box_params.lower_threshold_crossed = False
            if self.price_box_params.lower_threshold_crossed:
                if self.price_box_params.idx_p < len(self.quanto_tp_signal):
                    self.quanto_tp_signal.loc[self.price_box_params.idx_p, 'triggered'] = True

    def band_adjustments(self, entry_band, exit_band, move_exit):
        if move_exit:
            entry_adjustment_before = self.last_entry_band_adjustment
            entry = self.entry_band_adjustment(entry_band, exit_band, move_exit)
            exit_adjustment_before = self.last_exit_band_adjustment
            exit = self.exit_band_adjustment(entry_band, exit_band, move_exit)
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

        if move_exit:
            adjustment = 0
            if self.quanto_loss - (entry_band - exit_band) > 0 and self.trailing_value != 0 and \
                    self.quanto_loss >= self.disable_when_below:
                box_movement = 0
                additional_quanto_profit = self.quanto_loss - (entry_band - exit_band)
                if self.price_box_params and self.price_box_params.lower_threshold_crossed:
                    box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
                adjustment = box_movement + self.ratio_entry_band_mov * additional_quanto_profit

            if abs(adjustment) < 0.05:
                self.flag = False
                if exit_band - self.exit_band_adjustment(entry_band, exit_band, move_exit) > entry_band + adjustment:
                    adjustment = -self.last_exit_band_adjustment - (entry_band - exit_band) + self.minimum_distance
            else:
                self.flag = True
            self.last_entry_band_adjustment = adjustment
            return adjustment
        else:
            adjustment = 0
            if self.quanto_loss - (entry_band - exit_band) > 0 and self.trailing_value != 0 and \
                    self.quanto_loss >= self.disable_when_below:
                box_movement = 0
                additional_quanto_profit = self.quanto_loss - (entry_band - exit_band)
                if self.price_box_params and self.price_box_params.lower_threshold_crossed:
                    box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
                adjustment = box_movement + self.ratio_entry_band_mov * additional_quanto_profit
            return adjustment

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        if move_exit:
            if exit_band > entry_band + self.last_entry_band_adjustment and self.flag:
                movement_exit = -  self.last_entry_band_adjustment + self.minimum_distance
                self.last_exit_band_adjustment = movement_exit
                return movement_exit

            if self.ratio_exit_band_mov != 0:
                volume = - 1 / (self.price_btc_p * 0.000001)
                quanto_loss_exit = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                   avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                   coin_volume=volume)

                # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
                movement_exit = self.ratio_exit_band_mov * quanto_loss_exit
                # print(f'movement exit in exit_band_adjustment: {movement_exit}')
                if abs(self.last_entry_band_adjustment) > 0.05:
                    movement_to_entry = - (entry_band + self.last_entry_band_adjustment - exit_band
                                       - self.minimum_distance)
                    movement_exit = max(movement_exit + self.minimum_distance, movement_to_entry)
            else:
                movement_exit = 0
            self.last_exit_band_adjustment = movement_exit
            return movement_exit
        else:
            if exit_band > entry_band + self.entry_band_adjustment(entry_band, exit_band):
                return - self.entry_band_adjustment(entry_band, exit_band) + self.minimum_distance

            if self.ratio_exit_band_mov != 0:
                volume = - 1 / (self.price_btc_p * 0.000001)
                quanto_loss_exit = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                                   avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                                   coin_volume=volume)
                # print(f'quanto loss in exit_band_adjustment: {quanto_loss_exit}')
                movement_exit = self.ratio_exit_band_mov * quanto_loss_exit
                # print(f'movement exit in exit_band_adjustment: {movement_exit}')
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
        if self.quanto_loss > self.max_quanto_profit and self.quanto_loss >= self.disable_when_below:
            max_quanto_profit = self.quanto_loss
        elif self.quanto_loss < self.disable_when_below:
            max_quanto_profit = 0
        else:
            max_quanto_profit = self.max_quanto_profit
        return max_quanto_profit

    def allow_posting(self, side):
        if self.minimum_value < self.quanto_loss < self.max_quanto_profit - self.trailing_value:
            self.quanto_profit_triggered = True
        else:
            self.quanto_profit_triggered = False
        return self.quanto_profit_triggered