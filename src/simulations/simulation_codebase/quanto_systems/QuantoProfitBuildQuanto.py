from src.common.utils.quanto_utils import get_price_box_signal, quanto_pnl_func
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty, PriceBoxParams


class QuantoProfitBuildQuanto(QuantoSystemEmpty):
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
        super(QuantoProfitBuildQuanto, self).update(timestamp, position)
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
                self.quanto_tp_signal.loc[self.price_box_params.idx_p:, 'timems'].searchsorted(timestamp, side='left') \
                + self.price_box_params.idx_p
            if self.price_box_params.idx_p >= self.quanto_tp_signal.index[-1]:
                self.price_box_params.idx_p = self.quanto_tp_signal.index[-1]

            if self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal > self.price_box_params.upper_threshold:
                self.price_box_params.upper_threshold_crossed = True
            if self.price_box_params.upper_threshold_crossed and \
                    self.quanto_tp_signal.iloc[self.price_box_params.idx_p].signal < self.price_box_params.lower_threshold:
                self.price_box_params.lower_threshold_crossed = False
                self.price_box_params.upper_threshold_crossed = False
            if self.price_box_params.lower_threshold_crossed:
                if self.price_box_params.idx_p < len(self.quanto_tp_signal):
                    self.quanto_tp_signal.loc[self.price_box_params.idx_p, 'triggered'] = True

    def entry_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        volume = 1 / (self.price_btc_p * 0.000001)
        quanto_loss_virtual = quanto_pnl_func(avg_price_eth=self.price_eth_p, price_eth=self.price_eth_t,
                                              avg_price_btc=self.price_btc_p, price_btc=self.price_btc_t,
                                              coin_volume=volume)
        box_movement = 0
        additional_quanto_profit = max(quanto_loss_virtual - (entry_band - exit_band),0)
        if self.price_box_params and self.price_box_params.upper_threshold_crossed:
            box_movement = self.price_box_params.entry_movement_ratio * additional_quanto_profit
        # + self.ratio_entry_band_mov * additional_quanto_profit
        return box_movement

    def exit_band_adjustment(self, entry_band=0, exit_band=0, move_exit=False):
        if move_exit:
            entry_adjustment = self.entry_band_adjustment(entry_band, exit_band)
            if abs(entry_adjustment) >= 0.05:
                return - (entry_band + entry_adjustment - exit_band) + self.minimum_distance
            return - entry_adjustment
        else:
            return 0