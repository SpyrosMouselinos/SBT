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
            try:
                return self.ratio_entry_band_mov * self.quanto_loss
            except:
                print(f'quanto loss in exit_band_adjustment: {self.quanto_loss}')
                print(f'ratio entry band mov: {self.ratio_entry_band_mov}')
        return 0