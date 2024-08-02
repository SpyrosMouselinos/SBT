import numba
import pandas as pd
import time
import functools
from transitions import Machine
import numpy as np
from src.common.constants.constants import one_hour
from src.common.queries.queries import get_funding_for_symbol, \
    get_predicted_funding_for_symbol, get_realtime_funding_values
import logging
from pytictoc import TicToc
import warnings
from src.common.utils.local_funding import funding_classes, funding_systems
from src.common.utils.utils import bp_to_dollars, create_price_dataframe_local_folder, get_index_left, pop_old, \
    count_from_end, spread_entry_func_numba, spread_exit_func_numba, spread_entry_func_bp_numba, \
    spread_exit_func_bp_numba, get_index_right
from src.simulations.simulation_codebase.quanto_systems.QuantoBoth import QuantoBothSystem
from src.simulations.simulation_codebase.quanto_systems.QuantoBothExtended import QuantoBothExtendedSystem
from src.simulations.simulation_codebase.quanto_systems.QuantoExponential import QuantoProfitSystemExponential
from src.simulations.simulation_codebase.quanto_systems.QuantoProfit import QuantoProfitSystem
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitBuildQuanto import QuantoProfitBuildQuanto
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import QuantoSystemEmpty, QuantoLossSystem
from src.simulations.taker_maker.TakerMakerFunctions import get_taker_trades
from src.streamlit.streamlit_page_taker_maker_at_depth import ConstantDepthPosting

logging.disable(logging.DEBUG)
logging.disable(logging.INFO)

warnings.filterwarnings("ignore")
t = TicToc()
t.tic()


class TraderExpectedExecutions(object):
    """
    Variables used in the simulator:
    source = the source of the transition in the state machine
    final_spread =  the spread after the spot balance state
    df = dataframe with all the input values, Entry Exit Band Values, Area Spread, ...
    timestamp = the time in millis also the index of df
    stop_timestamp =  the timestamp when the stop trading is enabled in the Quanto loss stop_trading mode
    side = the side of the execution, entry or exit
    position_current_timestamp =  the position of the timestamp in df
    spot_fee = fee in the spot market
    swap_fee = swap_fee
    area_spread_threshold = area_spread_threshold
    area_spread_value = the area spread parameter used in trading
    latency_spot = latency to spot market
    latency_swap = latency to swap market
    latency_try_post = latency while trying to post in the swap market
    latency_cancel = latency while trying to cancel in the swap market
    latency_spot_balance = latency while we spot_balance in spot market
    environment = environment of the data, staging or production

    exchange_swap = exchange_swap
    swap_instrument = swap instrument, ticker of the coin
    spot_instrument = spot instrument, ticker of the coin

    max_position = max position of our trade in USD
    max_trade_volume = max trade volume of a single trade in USD

    funding_system = when Quanto contract the side of the trade, values: Quanto_loss, Quanto_profit, No

    stop_trading_enabled = boolean variable to initiate the procedure for stopping the trading in Quanto_loss
    halt_trading_flag = boolean variable to flag the stop trading timestamp in Quanto_loss

    ratio_entry_band_mov = ratio of the band movement additional to the allowed movement in Quanto_profit and
    Quanto_loss

    ratio_entry_band_mov_ind = ratio of the band movement additional to the allowed movement in Quanto_profit, this
    variable allows the entry band to move independent of the exit band whenever the exit is not close to entry

    ratio_exit_band_mov = ratio of the band movement in the exit band in Quanto_profit new band movement process

    rolling_time_window_size =  the window size we want to go into the past in order to compute the theoritical quanto
    profit or loss which moves the exit band in Quanto_profit new movement system or entry in Quanto_loss system

    move_exit_above_entry = boolean variable to define whether the exit band is allowed to move above the entry in
    Quanto_profit system

    funding_options = type:string defines the funding option we want to use for the combinations added especially for
    BitMEX/Deribit BTC

    exponent1 = the exponent used in the computation of quanto_profit
    exponent2 = the exponent used in the computation of quanto_profit

    current_r = the ratio of the band movement in Quanto_loss when we want top switch ratios in different volatility
    periods, this value overides the ratio_entry_band_mov parameter
    high_r =  the ratio of the band movement in Quanto_loss when we want to switch ratios in different volatility
    periods, this value overides the ratio_entry_band_mov parameter
    quanto_threshold = the maximum quanto loss allowed in Quanto_loss, also the trigger to change from current_r to high_r
    high_to_current = boolean, if enabled switches back from high_r to current_r

    minimum_distance = the minimum distance allowed between entry and exit band in Quanto_loss, Quanto_profit

    price_btc = the ask price of btc from BitMEX
    price_eth = the ask price pf eth from BitMEX

    """
    # Define the states of the trading:
    states = [{'name': 'clear', 'on_enter': ['band_funding_adjustment']},
              {'name': 'trying_to_post',
               'on_exit': ['swap_value', 'posting_f', 'trying_to_post_counter', 'keep_time_post']},
              {'name': 'not_posted'},
              {'name': 'posted'},
              {'name': 'try_to_cancel'},
              {'name': 'executing', 'on_enter': ['executing_f']},
              {'name': 'cancelled', 'on_enter': ['cancelled_f']},
              {'name': 'spot_balance', 'on_enter': ['spot_balance_f'], 'on_exit': ['swap_value']}]

    @staticmethod
    def funding_system_list_fun() -> list:
        return ['Quanto_loss', 'Quanto_profit', 'Quanto_profit_BOX', 'Quanto_profit_exp',
                'Quanto_both', 'Quanto_both_extended']

    def __init__(self, df, spot_fee, swap_fee, area_spread_threshold, latency_spot, latency_swap, latency_try_post,
                 latency_cancel, latency_spot_balance, max_position, max_trade_volume, environment, exchange_swap,
                 exchange_spot, swap_instrument, spot_instrument, funding_system, minimum_distance, minimum_value,

                 # parameters for ETHUSD sort-go-long
                 trailing_value,
                 disable_when_below, ratio_entry_band_mov, ratio_entry_band_mov_ind, stop_trading, current_r, high_r,
                 quanto_threshold, high_to_current, ratio_exit_band_mov=0.0, rolling_time_window_size: int = None,
                 hours_to_stop: int = 0, move_exit_above_entry: bool = False,

                 # new parameters for ethusd sort go long when in long position for the extended version
                 ratio_entry_band_mov_long: float = None,
                 rolling_time_window_size_long: int = None,
                 ratio_exit_band_mov_ind_long: float = None,

                 # set of parameters for the exponential quanto profit system
                 exponent1: float = None, exponent2: float = None, exponent3: float = None,
                 rolling_time_window_size2: int = None,
                 entry_upper_cap: float = 0.0,
                 entry_lower_cap: float = 0.0,
                 exit_upper_cap: float = 0.0,
                 w_theoretical_qp_entry: float = 0.0,
                 w_real_qp_entry: float = 0.0,

                 # paremeters for local minimum and maximum
                 use_local_min_max: bool = False,
                 num_of_points_to_lookback_entry: int = 0,
                 num_of_points_to_lookback_exit: int = 0,

                 # parameters for at depth posting
                 depth_posting_predictor=ConstantDepthPosting(0),
                 swap_market_tick_size=None, price_box_params=None, net_trading=None, use_bp=False,
                 funding_system_name="", funding_window=90, slow_funding_window=0,
                 funding_options: str = None, funding_ratios_params_spot=None,
                 funding_ratios_params_swap=None,
                 moving_average: int = None,
                 prediction_emitter=None):
        self.source = None
        self.final_spread = 0
        self.df = df
        self.df['quanto_profit'] = 0
        self.timestamp = df.index[100]
        self.timestamps = np.array(self.df.timems)

        # variable to stop trading if exit band above entry band when quanto loss enabled
        self.stop_timestamp = 0.0

        self.side = ''
        self.position_current_timestamp = 100
        self.spot_fee = spot_fee
        self.swap_fee = swap_fee
        self.area_spread_threshold = area_spread_threshold
        self.area_spread_value = 0
        self.latency_spot = latency_spot
        self.latency_swap = latency_swap
        self.latency_try_post = latency_try_post
        self.latency_cancel = latency_cancel
        self.latency_spot_balance = latency_spot_balance
        self.environment = environment

        self.spread_entry = 0
        self.spread_exit = 0
        self.entry_band = 0
        self.exit_band = 0
        self.spread_entry_posted = 0
        self.spread_exit_posted = 0
        self.entry_band_posted = 0
        self.exit_band_posted = 0

        self.funding_options = funding_options

        self.exchange_swap = exchange_swap
        self.exchange_spot = exchange_spot
        self.swap_instrument = swap_instrument
        self.spot_instrument = spot_instrument

        if self.exchange_swap == 'BitMEX':
            self.min_counter = 120
            self.sec_counter = 10
        elif self.exchange_swap == 'Binance':
            self.min_counter = 2300
            self.sec_counter = 75
        else:
            self.min_counter = 120
            self.sec_counter = 10

        self.max_position = max_position
        self.max_trade_volume = max_trade_volume
        self.cum_volume = 0

        self.funding_system = funding_system
        self.funding_window = funding_window
        self.slow_funding_window = slow_funding_window

        self.net_trading = net_trading

        # boolean variable to initiate the procedure for stopping the trading
        self.stop_trading_enabled = False
        # boolean variable to flag the stop trading timestamp
        self.halt_trading_flag = False

        self.moving_average = moving_average

        self.final_spread_bp = None
        self.exp1 = exponent1
        self.exp2 = exponent2
        self.exp3 = exponent3

        # parameters for local minimum and maximum
        self.use_local_min_max = use_local_min_max
        self.num_of_points_to_lookback_entry = num_of_points_to_lookback_entry
        self.num_of_points_to_lookback_exit = num_of_points_to_lookback_exit

        self.final_spread_bp = None
        # variables to switch from current to high R values
        self.current_r = current_r
        self.high_r = high_r
        self.quanto_threshold = quanto_threshold
        self.high_to_current = high_to_current
        self.depth_posting_predictor = depth_posting_predictor
        self.swap_market_tick_size = swap_market_tick_size
        self.predicted_depth = None
        self.max_targeted_depth = None
        self.hours_to_stop = hours_to_stop
        self.move_exit_above_entry = move_exit_above_entry
        if self.high_to_current:
            self.revert = True
        else:
            self.revert = False

        self.ratio_exit_band_mov = ratio_exit_band_mov
        self.rolling_time_window_size = rolling_time_window_size

        self.price_btc = pd.DataFrame()
        self.price_eth = pd.DataFrame()
        self.use_bp = use_bp
        self.allow_only_exit = False

        self.funding_system_name = funding_system_name
        if (((self.exchange_swap == 'BitMEX' and self.swap_instrument == 'ETHUSD') or
             (self.exchange_swap == 'Deribit' and self.swap_instrument == 'ETH-PERPETUAL')) and
                self.funding_system in self.funding_system_list_fun()):
            self.price_btc, self.price_eth = create_price_dataframe_local_folder(t_start=self.df.index[0] - 1000 * 60 *
                                                                                         int(self.rolling_time_window_size + 10),
                                                                                 t_end=self.df.index[
                                                                                           -1] + 1000 * 60 * 60 * 10,
                                                                                 spot_exchange='BitMEX',
                                                                                 spot_symbol='XBTUSD',
                                                                                 swap_exchange='BitMEX',
                                                                                 swap_symbol='ETHUSD', side='Ask')

            self.price_btc['timestamp'] = self.price_btc.Time.view(np.int64) // 10 ** 6
            self.price_btc.reset_index(drop=True, inplace=True)
            self.price_btc = self.price_btc[['timestamp', 'price']]

            self.price_eth['timestamp'] = self.price_eth.Time.view(np.int64) // 10 ** 6
            self.price_eth.reset_index(drop=True, inplace=True)
            self.price_eth = self.price_eth[['timestamp', 'price']]

        self.quanto_system = QuantoSystemEmpty(self.price_btc, self.price_eth)

        if self.funding_system == 'Quanto_profit':
            if price_box_params:
                price_box_params.t0 = self.df.timems.iloc[0] - 5 * one_hour
                price_box_params.t1 = self.df.timems.iloc[-1] + 5 * one_hour
            self.quanto_system = QuantoProfitSystem(price_btc=self.price_btc,
                                                    price_eth=self.price_eth,
                                                    distance=minimum_distance,
                                                    perc_entry=ratio_entry_band_mov,
                                                    perc_exit=ratio_exit_band_mov,
                                                    minimum_value=minimum_value,
                                                    trailing_value=trailing_value,
                                                    below=disable_when_below,
                                                    window=rolling_time_window_size,
                                                    price_box_params=price_box_params)
        if self.funding_system == 'Quanto_profit_BOX':
            if price_box_params:
                price_box_params.t0 = self.df.timems.iloc[0] - 5 * one_hour
                price_box_params.t1 = self.df.timems.iloc[-1] + 5 * one_hour
            self.quanto_system = QuantoProfitBuildQuanto(price_btc=self.price_btc,
                                                         price_eth=self.price_eth,
                                                         distance=minimum_distance,
                                                         perc_entry=ratio_entry_band_mov,
                                                         perc_exit=ratio_exit_band_mov,
                                                         minimum_value=minimum_value,
                                                         trailing_value=trailing_value,
                                                         below=disable_when_below,
                                                         window=rolling_time_window_size,
                                                         price_box_params=price_box_params)

        if self.funding_system == 'Quanto_loss':
            self.quanto_system = QuantoLossSystem(price_btc=self.price_btc, price_eth=self.price_eth,
                                                  current_r=current_r, high_r=high_r,
                                                  quanto_threshold=quanto_threshold,
                                                  distance=minimum_distance,
                                                  high_to_current=high_to_current,
                                                  ratio_entry_band_mov=ratio_entry_band_mov,
                                                  window=rolling_time_window_size,
                                                  ratio_entry_band_mov_ind=ratio_entry_band_mov_ind)

        if self.funding_system == 'Quanto_profit_exp':
            self.quanto_system = QuantoProfitSystemExponential(price_btc=self.price_btc,
                                                               price_eth=self.price_eth,
                                                               distance=minimum_distance,
                                                               window=self.rolling_time_window_size,
                                                               window2=rolling_time_window_size2,
                                                               exp1=exponent1,
                                                               exp2=exponent2,
                                                               exp3=exponent3,
                                                               cap_entry_pos=entry_upper_cap,
                                                               cap_entry_neg=entry_lower_cap,
                                                               cap_exit_pos=exit_upper_cap,
                                                               weight1=w_theoretical_qp_entry,
                                                               weight2=w_real_qp_entry
                                                               )
        if self.funding_system == "Quanto_both":
            # change to QuantoBoth if testing not successful
            self.quanto_system = QuantoBothSystem(price_btc=self.price_btc, price_eth=self.price_eth,
                                                  current_r=current_r, high_r=high_r,
                                                  quanto_threshold=quanto_threshold,
                                                  distance=minimum_distance,
                                                  high_to_current=high_to_current,
                                                  ratio_entry_band_mov=ratio_entry_band_mov,
                                                  window=rolling_time_window_size,
                                                  ratio_entry_band_mov_ind=ratio_entry_band_mov_ind)

        if self.funding_system == "Quanto_both_extended":
            self.quanto_system = QuantoBothExtendedSystem(price_btc=self.price_btc, price_eth=self.price_eth,
                                                          current_r=current_r, high_r=high_r,
                                                          quanto_threshold=quanto_threshold,
                                                          distance=minimum_distance,
                                                          high_to_current=high_to_current,
                                                          ratio_entry_band_mov=ratio_entry_band_mov,
                                                          window=rolling_time_window_size,
                                                          ratio_entry_band_mov_ind=ratio_entry_band_mov_ind,
                                                          ratio_entry_band_mov_long=ratio_entry_band_mov_long,
                                                          window_long=rolling_time_window_size_long,
                                                          ratio_exit_band_mov_ind_long=ratio_exit_band_mov_ind_long
                                                          )

        self.df[['Entry Band', 'Exit Band', 'price_swap_entry', 'price_swap_exit']] = self.df[
            ['Entry Band', 'Exit Band', 'price_swap_entry', 'price_swap_exit']]  # .astype(np.float32)
        if self.funding_system in self.funding_system_list_fun():
            self.quanto_loss_pnl = 0
            self.df['Exit Band with Quanto loss'] = self.df['Exit Band']
            self.df['Entry Band with Quanto loss'] = self.df['Entry Band']
            self.df['quanto_profit'] = 0
            self.quanto_loss = 0
            self.btc_idx = 0
            self.eth_idx = 0
            self.counter = 0
        if self.funding_system_name != '':
            self.df["Exit Band with Funding adjustment"] = self.df['Exit Band']
            self.df["Entry Band with Funding adjustment"] = self.df['Entry Band']

        if self.funding_system in ['Quanto_loss', 'Quanto_both', 'Quanto_both_extended']:
            # variable to stop trading if exit band above entry band when quanto loss enabled
            self.stop_trading = stop_trading
        else:
            self.stop_trading = False

        if self.stop_trading:
            self.quanto_system.ratio_entry_band_mov = self.current_r

        # initialize variables used in conditions
        self.list_trying_post_counter = numba.typed.List([0])

        self.traded_volume = 0
        self.total_volume_traded = 0
        self.time_post = 0
        self.entry_quanto_adjustment = 0
        self.exit_quanto_adjustment = 0
        # additional arguments for entry, exit_band_fn
        self.idx = 0
        self.previous_timestamp = 0
        if self.funding_system in self.funding_system_list_fun():
            cols = ['timems', 'Entry Band', 'Exit Band', 'price_swap_entry', 'price_swap_exit', 'price_spot_entry',
                    'price_spot_exit',
                    'Entry Band with Quanto loss',
                    'Exit Band with Quanto loss', 'quanto_profit', "has_prices_spot", "has_prices_swap"]
            self.df[cols] = self.df[cols]
        else:
            cols = ['timems', 'Entry Band', 'Exit Band', 'price_swap_entry', 'price_swap_exit', 'price_spot_entry',
                    'price_spot_exit', "has_prices_spot", "has_prices_swap"]

        if self.funding_options is not None:
            cols = ['timems', 'Entry Band', 'Exit Band', 'Entry Band Enter to Zero',
                    'Exit Band Exit to Zero', 'price_swap_entry', 'price_swap_exit', 'price_spot_entry',
                    'price_spot_exit', "has_prices_spot", "has_prices_swap"]
        if self.funding_system_name != '':
            cols += ["Entry Band with Funding adjustment", "Exit Band with Funding adjustment"]
        self.df_array = self.df[cols].to_numpy()

        self.columns_to_positions = {ix: x for x, ix in enumerate(cols)}

        # Query the funding here, because at the moment the funding has fewer points than the rest of the data
        if exchange_spot == "Binance":
            funding_spot = get_predicted_funding_for_symbol(self.timestamps[0], self.timestamps[-1], exchange_spot,
                                                            spot_instrument)
        elif self.moving_average is not None and exchange_spot == 'Deribit':
            funding_spot = get_realtime_funding_values(t0=self.timestamps[0], t1=self.timestamps[-1],
                                                       exchange=exchange_spot, symbol=spot_instrument,
                                                       moving_average=self.moving_average)
        else:
            funding_spot = get_funding_for_symbol(self.timestamps[0], self.timestamps[-1], exchange_spot,
                                                  spot_instrument)
        if exchange_swap == "Binance":
            funding_swap = get_predicted_funding_for_symbol(self.timestamps[0], self.timestamps[-1], exchange_swap,
                                                            swap_instrument)
        elif self.moving_average is not None and exchange_swap == 'Deribit':
            funding_swap = get_realtime_funding_values(t0=self.timestamps[0], t1=self.timestamps[-1],
                                                       exchange=exchange_swap, symbol=swap_instrument,
                                                       moving_average=self.moving_average)
        else:
            funding_swap = get_funding_for_symbol(self.timestamps[0], self.timestamps[-1], exchange_swap,
                                                  swap_instrument)
        funding_spot['Time'] = funding_spot.index
        funding_swap['Time'] = funding_swap.index
        funding_spot['timems'] = funding_spot.Time.view(np.int64) // 10 ** 6
        funding_swap['timems'] = funding_swap.Time.view(np.int64) // 10 ** 6

        temporary_funding_spot = funding_classes[exchange_spot](
            funding_spot[['timems', 'funding']].to_numpy().astype(np.float64), funding_ratios_params_spot)
        temporary_funding_swap = funding_classes[exchange_swap](
            funding_swap[['timems', 'funding']].to_numpy().astype(np.float64), funding_ratios_params_swap)
        self.temp_funding_system = funding_systems[funding_system_name](funding_spot=temporary_funding_spot,
                                                                        funding_swap=temporary_funding_swap)
        self.entry_funding_adjustment = 0
        self.exit_funding_adjustment = 0
        self.entry_funding_adjustment_to_zero = 0
        self.exit_funding_adjustment_to_zero = 0
        self.exit_funding_adjustment = 0
        self.idx_previous_band_adjustment = 0

        self.taker_volume_df = None
        self.time_spent_spread_unavailable = 0
        self.time_time_movement_correct = 0
        self.time_order_too_deep = 0
        self.time_activation_function_cancel = 0
        self.time_activation_function_try_post = 0
        self.time_try_post_condition = 0

        # Initialize the state machine
        self.machine = Machine(model=self, states=TraderExpectedExecutions.states, send_event=True,
                               initial='clear')
        if self.funding_system in self.funding_system_list_fun():

            # At spread not available
            self.machine.add_transition(trigger='initial_condition', source='clear', dest='clear',
                                        unless=['try_post_condition'], after=['move_time_forward'],
                                        before=['quanto_loss_func'])

            # At spread available and area spread above area spread threshold
            self.machine.add_transition(trigger='initial_condition', source='clear', dest='trying_to_post',
                                        conditions=['try_post_condition'])

            # Since we are in trying to post state we have to check the condition and if it is not meet we move
            # to not posted state
            self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='not_posted',
                                        unless=['activation_function_try_post'], after=['update_time_latency_try_post',
                                                                                        'move_time_forward'],
                                        before=['quanto_loss_func'])

            # # Since we are in the no posting state we have to return to the beginning
            self.machine.add_transition(trigger='reset', source='not_posted', dest='clear', after=['move_time_forward'],
                                        before=['quanto_loss_func'])

            # If the condition is met we move from trying to post to posted state
            self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='posted',
                                        conditions=['activation_function_try_post'],
                                        after=['update_time_latency_try_post', 'move_time_forward'],
                                        before=['quanto_loss_func'])

            # From posted state we move to try to cancel state
            self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                        conditions=['spread_unavailable_post'])

            # From posted state we move to try to cancel state
            if self.funding_system == 'Quanto_profit':
                self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                            conditions=['is_order_too_deep'])

                # Or from posted state we can move to executing state
                self.machine.add_transition(trigger='move_from_post', source='posted', dest='executing',
                                            conditions=['swap_price_movement_correct'],
                                            unless=['is_order_too_deep'])
                # loop in the posted state until spread is available
                self.machine.add_transition(trigger='move_from_post', source='posted', dest='posted',
                                            conditions=['swap_price_no_movement'], unless=['is_order_too_deep'],
                                            after=['move_time_forward'],
                                            before=['quanto_loss_func'])

            else:
                self.machine.add_transition(trigger='move_from_post', source='posted', dest='executing',
                                            conditions=['swap_price_movement_correct'])
                # loop in the posted state until spread is available
                self.machine.add_transition(trigger='move_from_post', source='posted', dest='posted',
                                            conditions=['swap_price_no_movement'],
                                            after=['move_time_forward'],
                                            before=['quanto_loss_func'])

            # From try to cancel state we can move to two directions either cancelled or executing
            self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='executing',
                                        conditions=['activation_function_cancel'])

            self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='cancelled',
                                        unless=['activation_function_cancel'])

            # From spot balance go to initial state clear
            # From executing state we go to spot balance state
            self.machine.add_transition(trigger='reset', source=['cancelled', 'spot_balance'], dest='clear',
                                        after=['move_time_forward'], before=['quanto_loss_func'])

            # From executing state we go to spot balance state
            self.machine.add_transition(trigger='move_from_executing', source='executing', dest='spot_balance',
                                        after=['compute_final_spread', 'volume_traded', 'quanto_loss_w_avg'])
        else:
            # At spread not available
            self.machine.add_transition(trigger='initial_condition', source='clear', dest='clear',
                                        unless=['try_post_condition'], after=['move_time_forward'])

            self.machine.add_transition(trigger='initial_condition', source='clear', dest='clear',
                                        unless=['has_prices'], after=['move_time_forward'])

            # At spread available and area spread above area spread threshold
            self.machine.add_transition(trigger='initial_condition', source='clear', dest='trying_to_post',
                                        conditions=['try_post_condition', "has_prices"], unless=[])

            # Since we are in trying to post state we have to check the condition and if it is not meet we move
            # to not posted state
            self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='not_posted',
                                        unless=['activation_function_try_post'], after=['update_time_latency_try_post',
                                                                                        'move_time_forward'])

            # # Since we are in the no posting state we have to return to the beginning
            self.machine.add_transition(trigger='reset', source='not_posted', dest='clear', after=['move_time_forward'])

            # If the condition is met we move from trying to post to posted state
            self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='posted',
                                        conditions=['activation_function_try_post'],
                                        after=['update_time_latency_try_post',
                                               'move_time_forward'])

            self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                        conditions=['not_has_prices'])

            # From posted state we move to try to cancel state
            self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                        conditions=['spread_unavailable_post'], unless=["not_has_prices"])

            # Or from posted state we can move to executing state
            self.machine.add_transition(trigger='move_from_post', source='posted', dest='executing',
                                        conditions=['swap_price_movement_correct'])

            # loop in the posted state until spread is available
            self.machine.add_transition(trigger='move_from_post', source='posted', dest='posted',
                                        conditions=['swap_price_no_movement'], unless=["not_has_prices"],
                                        after=['move_time_forward'])

            # From try to cancel state we can move to two directions either cancelled or executing
            self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='executing',
                                        conditions=['activation_function_cancel'])

            self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='cancelled',
                                        unless=['activation_function_cancel'])

            # From spot balance go to initial state clear
            # From executing state we go to spot balance state
            self.machine.add_transition(trigger='reset', source=['cancelled', 'spot_balance'], dest='clear',
                                        after=['move_time_forward'])

            # From executing state we go to spot balance state
            self.machine.add_transition(trigger='move_from_executing', source='executing', dest='spot_balance',
                                        after=['compute_final_spread', 'volume_traded'])

    def compute_volume_size(self):
        traded_volume = self.max_trade_volume
        return traded_volume

    def find_known_values_index(self):
        '''
        Find the index (timestamp) of the known spot and swap prices.
        Known values: values we know at any time moment and include a latency concern the time we receive the
        information of a new price.
        '''

        df = self.df
        if self.timestamp - self.latency_spot >= df.index[0]:
            # position_current_timestamp = df.index.get_loc(self.timestamp)
            position_current_timestamp = df.index.searchsorted(self.timestamp)
            idx_lat_spot = df.index[
                get_index_left(self.timestamps, position_current_timestamp, latency=self.latency_spot)]
            idx_lat_swap = df.index[
                get_index_left(self.timestamps, position_current_timestamp, latency=self.latency_swap)]

            if idx_lat_spot is None:
                idx_lat_spot = self.timestamp
            if idx_lat_swap is None:
                idx_lat_swap = self.timestamp
        else:
            idx_lat_spot = self.timestamp
            idx_lat_swap = self.timestamp

        return idx_lat_spot, idx_lat_swap

    @property
    def has_prices(self):
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        temp_ix_spot = self.df.index.get_loc(idx_lat_spot)
        cond_spot = self.df_array[temp_ix_spot, self.columns_to_positions['has_prices_spot']]
        cond_swap = self.df_array[temp_ix_swap, self.columns_to_positions['has_prices_swap']]
        if not (cond_swap and cond_spot):
            pass
        return cond_swap and cond_spot

    @property
    def not_has_prices(self):
        return not self.has_prices

    def entry_band_fn(self, event, cum_volume):
        if self.funding_options is not None and cum_volume < 0:
            try:
                funding_adjustment = self.entry_funding_adjustment_to_zero
                if not self.use_bp:
                    funding_adjustment = funding_adjustment / 10000 * self.df_array[
                        self.idx, self.columns_to_positions['price_swap_entry']]
                return self.df_array[self.idx, self.columns_to_positions['Entry Band Enter to Zero']] + \
                    self.entry_quanto_adjustment + funding_adjustment
            except:
                return self.df_array[self.idx, self.columns_to_positions['Entry Band Enter to Zero']] + \
                    self.entry_quanto_adjustment

        elif cum_volume < 0:
            funding_adjustment = self.entry_funding_adjustment_to_zero
        else:
            funding_adjustment = self.entry_funding_adjustment
        if not self.use_bp:
            funding_adjustment = funding_adjustment / 10000 * self.df_array[
                self.idx, self.columns_to_positions['price_swap_entry']]

        return self.df_array[self.idx, self.columns_to_positions['Entry Band']] + self.entry_quanto_adjustment + \
            funding_adjustment

    def exit_band_fn(self, event, cum_volume):
        if self.funding_options is not None and cum_volume > 0:
            try:
                funding_adjustment = self.exit_funding_adjustment_to_zero
                if not self.use_bp:
                    funding_adjustment = funding_adjustment / 10000 * self.df_array[
                        self.idx, self.columns_to_positions['price_swap_entry']]
                return self.df_array[self.idx, self.columns_to_positions['Exit Band Exit to Zero']] - \
                    self.exit_quanto_adjustment - funding_adjustment
            except:
                return self.df_array[self.idx, self.columns_to_positions['Exit Band Exit to Zero']] - \
                    self.exit_quanto_adjustment

        elif cum_volume > 0:
            funding_adjustment = self.exit_funding_adjustment_to_zero
        else:
            funding_adjustment = self.exit_funding_adjustment
        if not self.use_bp:
            funding_adjustment = funding_adjustment / 10000 * self.df_array[
                self.idx, self.columns_to_positions['price_swap_entry']]
        return self.df_array[self.idx, self.columns_to_positions['Exit Band']] - self.exit_quanto_adjustment - \
            funding_adjustment

    def local_min_max(self):
        """
        function to compute if local minimum (or maximum) is above entry band (or below exit band)
        this function is activated if the use_local_min_max is TRUE
        """

        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        temp_ix_spot = self.df.index.get_loc(idx_lat_spot)
        len_array = min(len(self.df_array[:temp_ix_swap, :]), len(self.df_array[:temp_ix_spot, :])) - 1
        if len_array < min(self.num_of_points_to_lookback_entry, self.num_of_points_to_lookback_exit):
            entry_swap = self.df_array[temp_ix_swap - len_array:temp_ix_swap,
                         self.columns_to_positions['price_swap_entry']]
            exit_swap = self.df_array[temp_ix_swap - len_array:temp_ix_swap,
                        self.columns_to_positions['price_swap_exit']]

            entry_spot = self.df_array[temp_ix_spot - len_array:temp_ix_spot,
                         self.columns_to_positions['price_spot_entry']]
            exit_spot = self.df_array[temp_ix_spot - len_array:temp_ix_spot,
                        self.columns_to_positions['price_spot_exit']]
        else:
            entry_swap = self.df_array[temp_ix_swap - self.num_of_points_to_lookback_entry:temp_ix_swap,
                         self.columns_to_positions['price_swap_entry']]
            exit_swap = self.df_array[temp_ix_swap - self.num_of_points_to_lookback_exit:temp_ix_swap,
                        self.columns_to_positions['price_swap_exit']]

            entry_spot = self.df_array[temp_ix_spot - self.num_of_points_to_lookback_entry:temp_ix_spot,
                         self.columns_to_positions['price_spot_entry']]
            exit_spot = self.df_array[temp_ix_spot - self.num_of_points_to_lookback_entry:temp_ix_spot,
                        self.columns_to_positions['price_spot_exit']]

        self.idx = max(temp_ix_spot, temp_ix_swap)

        self.entry_band = self.entry_band_fn(event=None, cum_volume=self.cum_volume)
        self.exit_band = self.exit_band_fn(event=None, cum_volume=self.cum_volume)

        if not self.use_bp:
            spread_entry = spread_entry_func_numba(entry_swap, entry_spot, swap_fee=self.swap_fee,
                                                   spot_fee=self.spot_fee)
            spread_exit = spread_exit_func_numba(exit_swap, exit_spot, swap_fee=self.swap_fee,
                                                 spot_fee=self.spot_fee)
        if self.use_bp:
            spread_entry = spread_entry_func_bp_numba(entry_swap, entry_spot,
                                                      swap_fee=self.swap_fee, spot_fee=self.spot_fee)
            spread_exit = spread_exit_func_bp_numba(exit_swap, exit_spot, swap_fee=self.swap_fee,
                                                    spot_fee=self.spot_fee)

        if len(spread_entry) != 0:
            self.spread_entry = spread_entry[-1]
            min_entry_spread = spread_entry.min()
        else:
            self.spread_entry = spread_entry
            min_entry_spread = spread_entry

        if len(spread_exit) != 0:
            self.spread_exit = spread_exit[-1]
            max_exit_spread = spread_exit.max()
        else:
            self.spread_exit = spread_exit
            max_exit_spread = spread_exit

        if self.spread_entry >= self.entry_band:
            self.side = 'entry'
        elif self.spread_exit <= self.exit_band:
            self.side = 'exit'
        else:
            self.side = ''

        local_min_max_cond = ((min_entry_spread >= self.entry_band) & (self.side == 'entry')) | \
                             ((max_exit_spread <= self.exit_band) & (self.side == 'exit'))

        return local_min_max_cond

    def spread_available(self):
        '''
        Find the spread of the known values
        '''

        df = self.df
        # print(df)
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        temp_ix_spot = self.df.index.get_loc(idx_lat_spot)
        entry_swap = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_entry']]
        exit_swap = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_exit']]
        if self.predicted_depth is None:
            self.predicted_depth = self.depth_posting_predictor.where_to_post(self.timestamp)
        predicted_depth_usd = bp_to_dollars(self.predicted_depth, entry_swap)
        if not pd.isna(predicted_depth_usd / self.swap_market_tick_size):
            predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)
        else:
            predicted_depth_usd = 0
        entry_spot = self.df_array[temp_ix_spot, self.columns_to_positions['price_spot_entry']]
        exit_spot = self.df_array[temp_ix_spot, self.columns_to_positions['price_spot_exit']]

        # lat is a timestamp in milliseconds.
        self.idx = max(temp_ix_spot, temp_ix_swap)

        self.entry_band = self.entry_band_fn(event=None, cum_volume=self.cum_volume)
        self.exit_band = self.exit_band_fn(event=None, cum_volume=self.cum_volume)

        if not self.use_bp:
            self.spread_entry = spread_entry_func_numba(entry_swap + predicted_depth_usd, entry_spot,
                                                        swap_fee=self.swap_fee,
                                                        spot_fee=self.spot_fee)
            self.spread_exit = spread_exit_func_numba(exit_swap - predicted_depth_usd, exit_spot,
                                                      swap_fee=self.swap_fee,
                                                      spot_fee=self.spot_fee)
        if self.use_bp:
            self.spread_entry = spread_entry_func_bp_numba(entry_swap + predicted_depth_usd, entry_spot,
                                                           swap_fee=self.swap_fee, spot_fee=self.spot_fee)
            self.spread_exit = spread_exit_func_bp_numba(exit_swap - predicted_depth_usd, exit_spot,
                                                         swap_fee=self.swap_fee,
                                                         spot_fee=self.spot_fee)
        if self.spread_entry >= self.entry_band:
            self.side = 'entry'
        elif self.spread_exit <= self.exit_band:
            self.side = 'exit'
        else:
            self.side = ''

        if self.funding_system in self.funding_system_list_fun():
            spread_cond = ((self.spread_entry >= self.entry_band) & (self.side == 'entry') & (
                    self.spread_entry <= 4)) | \
                          ((self.spread_exit <= self.exit_band) & (self.side == 'exit'))
        elif (self.exchange_swap == "BitMEX" or self.exchange_spot == "BitMEX") and \
                (self.swap_instrument == "XBTUSD" or self.spot_instrument == "XBTUSD"):
            # additional condition to avoid posting when spread is too wide in BitMEX XBTUSD
            spread_cond = ((self.spread_entry >= self.entry_band) & (self.side == 'entry') &
                           (abs(self.spread_entry - self.entry_band) <= 30)) | \
                          ((self.spread_exit <= self.exit_band) & (self.side == 'exit') &
                           (abs(self.spread_exit - self.exit_band) <= 30))
        else:
            spread_cond = ((self.spread_entry >= self.entry_band) & (self.side == 'entry')) | \
                          ((self.spread_exit <= self.exit_band) & (self.side == 'exit'))

        return spread_cond or self.quanto_system.allow_posting(self.side)

    def area_spread(self):
        if (self.side == 'entry') & (self.df.loc[self.timestamp, 'entry_area_spread'] >= self.area_spread_threshold):
            return True
        elif (self.side == 'exit') & (self.df.loc[self.timestamp, 'exit_area_spread'] >= self.area_spread_threshold):
            return True
        else:
            return False

    def rate_limit(self):
        if len(self.list_trying_post_counter) < 1:
            return True
        self.list_trying_post_counter = pop_old(self.list_trying_post_counter, self.timestamp, 1000 * 60)
        counter = len(self.list_trying_post_counter)
        if counter < self.min_counter:
            if count_from_end(self.list_trying_post_counter, self.timestamp, 1000) < self.sec_counter:
                return True
            else:
                return False
        else:
            return False

    @functools.lru_cache(maxsize=2)
    def inner_try_post_condition(self, timestamp):
        if self.use_local_min_max:
            if not self.local_min_max():
                return False
        else:
            if not self.spread_available():
                return False

        if self.stop_trading:
            if self.timestamp < self.stop_timestamp + 1000 * 60 * 60 * self.hours_to_stop:
                # resetting to default
                self.stop_trading_enabled = False
                self.halt_trading_flag = False
                if self.high_to_current:
                    self.revert = False
                self.counter = 0
                if self.allow_only_exit:
                    if self.side == 'exit' or self.cum_volume < 0:
                        return self.helper_conditions_to_post()
                return False
            # the high_to_current needs to be debugged
            self.allow_only_exit = False
            if not self.revert:
                self.quanto_system.ratio_entry_band_mov = self.current_r
            if self.stop_trading_enabled and self.side == 'entry':
                return False
        return self.helper_conditions_to_post()

    def helper_conditions_to_post(self):
        if self.area_spread_threshold != 0:
            if (self.area_spread() == True) & (self.rate_limit() == True) & (self.trading_condition() == True):
                return True
            else:
                return False
        else:
            if (self.rate_limit() == True) & (self.trading_condition() == True):
                return True
            else:
                return False

    @property
    def try_post_condition(self):
        """
        conditions used in the transition clear -> clear and clear -> try_to_post
        This condition checks:
        - if the spread is available
        - If the stop trading is enabled in Quanto_loss and stops the trading for 8 hours
        - If we are rate limited
        - If the cumulative volume is above the max position allowed
        - If the area spread condition  is met.
        """
        start = time.time_ns()
        temp = self.inner_try_post_condition(self.timestamp)
        self.time_try_post_condition += time.time_ns() - start
        return temp

    def update_time_latency_try_post(self, event):
        df = self.df
        self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                          self.latency_try_post) - 1
        self.timestamp = df.index[self.position_current_timestamp]

    @property
    def activation_function_try_post(self):
        '''
        Condition to exit the trying_to-post state
        This conditions performs the following:
        * Adds a latency to the values.
        * Checks if in this time interval there is a price change in the swap prices
        * returns the side of the execution that will be used later on
        '''
        start = time.time_ns()
        df = self.df
        a = self.position_current_timestamp
        try:
            b_idx = get_index_right(self.timestamps, self.position_current_timestamp, self.latency_try_post)
        except:
            b_idx = 0

        b = b_idx - 1
        idx_list = df.index[a:b]

        if len(idx_list) > 1:
            idx = 1

            while idx < len(idx_list):
                if (self.side == 'entry') & (
                        df.loc[idx_list[idx - 1], 'price_swap_entry'] < df.loc[idx_list[idx], 'price_swap_entry']):
                    self.time_activation_function_try_post += time.time_ns() - start
                    return False
                elif (self.side == 'exit') & (
                        df.loc[idx_list[idx - 1], 'price_swap_exit'] > df.loc[idx_list[idx], 'price_swap_exit']):
                    self.time_activation_function_try_post += time.time_ns() - start
                    return False
                else:
                    idx += 1

            if idx == len(idx_list):
                return True

        else:
            return True

    @functools.lru_cache(maxsize=2)
    def inner_spread(self, timestamp):
        df = self.df
        # print(df)
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()

        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        temp_ix_spot = self.df.index.get_loc(idx_lat_spot)
        idx_p = temp_ix_swap - 1
        entry_swap = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_entry']]
        entry_spot = self.df_array[temp_ix_spot, self.columns_to_positions['price_spot_entry']]
        exit_spot = self.df_array[temp_ix_spot, self.columns_to_positions['price_spot_exit']]

        self.idx = max(temp_ix_swap, temp_ix_spot)
        self.entry_band_posted = self.entry_band_fn(event=None, cum_volume=self.cum_volume)
        self.exit_band_posted = self.exit_band_fn(event=None, cum_volume=self.cum_volume)
        # Computing both spread with the same swap price because only the spread on the correct size will be used
        if self.use_bp:
            spread_entry = spread_entry_func_bp_numba(self.swap_price, entry_spot, swap_fee=self.swap_fee,
                                                      spot_fee=self.spot_fee)
            spread_exit = spread_exit_func_bp_numba(self.swap_price, exit_spot, swap_fee=self.swap_fee,
                                                    spot_fee=self.spot_fee)

        else:
            spread_entry = spread_entry_func_numba(self.swap_price, entry_spot, swap_fee=self.swap_fee,
                                                   spot_fee=self.spot_fee)
            spread_exit = spread_exit_func_numba(self.swap_price, exit_spot, swap_fee=self.swap_fee,
                                                 spot_fee=self.spot_fee)
        self.spread_entry_posted = spread_entry
        self.spread_exit_posted = spread_exit
        targeted_depth_usd = bp_to_dollars(self.max_targeted_depth, entry_swap)
        targeted_depth_usd = self.swap_market_tick_size * round(targeted_depth_usd / self.swap_market_tick_size)

        if self.quanto_system.allow_posting(self.side):
            return False

        if self.funding_options is not None:
            if self.cum_volume < 0:
                self.entry_band_posted = self.entry_band_fn(event=None, cum_volume=self.cum_volume)
            elif self.cum_volume > 0:
                self.exit_band_posted = self.exit_band_fn(event=None, cum_volume=self.cum_volume)

        res = check_spread_and_depth_in_posted(side=self.side, spread_entry=spread_entry,
                                               entry_band=self.entry_band_posted,
                                               swap_price=self.swap_price,
                                               price_swap_entry=self.df_array[temp_ix_swap,
                                               self.columns_to_positions['price_swap_entry']],
                                               targeted_depth_usd=targeted_depth_usd,
                                               spread_exit=spread_exit,
                                               exit_band=self.exit_band_posted,
                                               price_swap_exit=self.df_array[temp_ix_swap,
                                               self.columns_to_positions['price_swap_exit']]
                                               )

        return res

    @property
    def spread_unavailable_post(self):
        '''
        Define if the spread is unavailable or the swap price moves in the wrong direction.
        Return True if spread is unavailable or swap price moved in wrong direction.
        * in entry side wrong direction is previous value larger than current value
        * in exit side wrong direction is previous value smaller than current value
        The spread is calculated on the known values and not on the real values
        '''
        start = time.time_ns()
        temp = self.inner_spread(self.timestamp)
        self.time_spent_spread_unavailable += time.time_ns() - start
        return temp

    @functools.lru_cache(maxsize=2)
    def inner_too_deep(self, timestamp):
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()

        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        entry_swap = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_entry']]
        exit_swap = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_exit']]

        if self.side == 'entry':
            targeted_depth_usd = bp_to_dollars(self.max_targeted_depth, self.swap_price)
            if targeted_depth_usd < self.swap_price - entry_swap:
                return True

        elif self.side == 'exit':
            targeted_depth_usd = bp_to_dollars(self.max_targeted_depth, self.swap_price)
            if targeted_depth_usd < exit_swap - self.swap_price:
                return True
        if self.swap_price > self.swap_price + self.predicted_depth:
            return True
        return False

    @property
    def is_order_too_deep(self):
        start = time.time_ns()
        temp = self.inner_too_deep(self.timestamp)
        self.time_order_too_deep += time.time_ns() - start
        return temp

    def keep_time_post(self, event):
        if event.transition.dest == 'posted':
            self.time_post = self.timestamp
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]

    @property
    def swap_price_no_movement(self):
        '''
        When there is no movement in the swap price this function will return True
        In order to archive that we compare the current known value with the previous one
        '''
        return not (self.spread_unavailable_post or self.swap_price_movement_correct)

    @functools.lru_cache(maxsize=2)
    def inner_price_movement(self, timestamp):
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        idx_c = self.df.index.get_loc(idx_lat_swap)
        if self.time_post == 0:
            return False
        takers_volume = self.get_taker_trades(self.time_post, self.timestamp)
        mask_bid = takers_volume.side == 'Bid'
        mask_ask = takers_volume.side == 'Ask'
        if (self.side == 'entry' and not mask_ask.any()) or (self.side == 'exit' and not mask_bid.any()):
            return False

        if self.side == 'entry':
            if self.swap_price < self.df_array[idx_c, self.columns_to_positions['price_swap_entry']]:
                if mask_bid.any():
                    # @TODO Is the next condition correct?
                    if (mask_bid * (takers_volume.price >= self.swap_price)).any():
                        return True
                    else:
                        return False
            else:
                return False
        else:
            if self.swap_price > self.df_array[idx_c, self.columns_to_positions['price_swap_exit']]:
                if mask_ask.any():
                    if (mask_ask * (takers_volume.price <= self.swap_price)).any() > 0:
                        return True
                    else:
                        return False

            else:
                return False

    @property
    def swap_price_movement_correct(self):
        '''
        Condition to check if we have a movement of the known swap price towards the correct side
        (price increase on entry and price decrease on exit)
        '''
        start = time.time_ns()
        temp = self.inner_price_movement(self.timestamp)
        self.time_time_movement_correct += time.time_ns() - start
        return temp

    def get_taker_trades(self, from_ts, to_ts):
        if self.taker_volume_df is None or to_ts > self.taker_volume_df.iloc[-1]['timems']:
            load_from = from_ts - 1000 * 60 * 15
            load_to = to_ts + 1000 * 60 * 60 * 25
            # print(f"Loading taker trades from {datetime.fromtimestamp(load_from // 1000)} to {datetime.fromtimestamp(load_to // 1000)}")
            # 25 hours - force loading from disk which checks if time period is >24h
            self.taker_volume_df = get_taker_trades(load_from, load_to, self.exchange_swap, self.swap_instrument)
            self.taker_volume_df = self.taker_volume_df[self.taker_volume_df['price'] != 0]
            if to_ts > self.taker_volume_df.iloc[-1]['timems']:
                # print(f"Last timestamp in trades smaller than to_ts. Last timestamp {self.taker_volume_df.iloc[-1]['timems']}, to_ts {to_ts}")
                self.taker_volume_df = get_taker_trades(load_from + 1000 * 60 * 60 * 24, load_to + 1000 * 60 * 60 * 24,
                                                        self.exchange_swap, self.swap_instrument)
                self.taker_volume_df = self.taker_volume_df[self.taker_volume_df['price'] != 0]
                if to_ts > self.taker_volume_df.iloc[-1]['timems']:
                    print("Takers missing!")

        index_start = np.searchsorted(self.taker_volume_df['timems'], from_ts, side="left")
        index_end = np.searchsorted(self.taker_volume_df['timems'], to_ts, side="right")
        return self.taker_volume_df[index_start:index_end]

    @functools.lru_cache(maxsize=2)
    def inner_activation_function_cancel(self, timestamp):
        '''
        We get cancelled when the price swap moves against us in the time interval between now and now+latency
        '''
        start = time.time_ns()
        df = self.df
        a = self.position_current_timestamp
        b = get_index_right(self.timestamps, self.position_current_timestamp,
                            self.latency_cancel) - 1
        idx_list = df.index[a:b]
        if self.time_post == 0:
            return False
        takers_volume = self.get_taker_trades(self.time_post, df.index[b])

        mask_bid = takers_volume.side == 'Bid'
        mask_ask = takers_volume.side == 'Ask'
        if (self.side == 'exit' and not mask_ask.any()) or \
                (self.side == 'entry' and not mask_bid.any()):
            return False

        if len(idx_list) > 1:
            idx = 1
            while idx < len(idx_list):
                if (self.side == 'entry') & (self.swap_price < self.df_array[
                    self.df.index.get_loc(idx_list[idx]), self.columns_to_positions['price_swap_entry']]):
                    if len(takers_volume[mask_bid].index) > 0:
                        if (mask_bid * (takers_volume.price >= self.swap_price)).sum() > 0:
                            return True
                        else:
                            return False
                    else:
                        return False


                elif (self.side == 'exit') & (self.df_array[
                                                  self.df.index.get_loc(idx_list[idx - 1]), self.columns_to_positions[
                                                      'price_swap_exit']] > self.swap_price):
                    if len(takers_volume[mask_ask].index) > 0:
                        if (mask_ask * (takers_volume.price <= self.swap_price)).sum() > 0:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    idx += 1

            if idx == len(idx_list):
                return False
        else:
            return False

    @property
    def activation_function_cancel(self):
        start = time.time_ns()
        temp = self.inner_activation_function_cancel(self.timestamp)
        self.time_activation_function_cancel += time.time_ns() - start
        return temp

    def move_time_forward(self, event):
        '''
        move time forward from one transition to another when no latency is Included.
        When condition is met compute also the quanto loss
        '''
        if self.timestamp < self.df.index[-1]:
            self.position_current_timestamp += 1
            self.timestamp = self.df.index[self.position_current_timestamp]

    def swap_value(self, event):
        '''
        Find the swap price on the exit of the trying to post state in order to use it later in order to define the
        executing spread.
        '''
        idx_lat_spot, idx_lat_swap = self.find_known_values_index()
        temp_ix_swap = self.df.index.get_loc(idx_lat_swap)
        # temp_ix_spot = self.df.index.get_loc(idx_lat_spot)
        if self.predicted_depth is None:
            self.predicted_depth = self.depth_posting_predictor.where_to_post(self.timestamp)
        if self.side == 'entry':
            price = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_entry']]
            # price = self.df.loc[idx_lat_swap, 'price_swap_entry']
            predicted_depth_usd = bp_to_dollars(self.predicted_depth, price)
            predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)
            self.swap_price = price + predicted_depth_usd
            self.max_targeted_depth = self.predicted_depth
        else:
            price = self.df_array[temp_ix_swap, self.columns_to_positions['price_swap_exit']]
            # price = self.df.loc[idx_lat_swap, 'price_swap_exit']
            predicted_depth_usd = bp_to_dollars(self.predicted_depth, price)
            predicted_depth_usd = self.swap_market_tick_size * round(predicted_depth_usd / self.swap_market_tick_size)
            self.swap_price = price - predicted_depth_usd
            self.max_targeted_depth = self.predicted_depth

    def trying_to_post_counter(self, event):
        self.list_trying_post_counter.append(self.timestamp)

    # On entry and on exit functions of the states
    def posting_f(self, event):
        # print('currently posting')
        df = self.df

        if self.timestamp + self.latency_try_post in df.index:
            self.timestamp = self.timestamp + self.latency_try_post
            self.position_current_timestamp = df.index.get_loc(self.timestamp)
        else:
            # self.timestamp = df.index[df.index.searchsorted(self.timestamp + self.latency_try_post) - 1]
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_try_post) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def executing_f(self, event):
        # print('currently executing')
        df = self.df

        if event.transition.source == 'try_cancel':
            latency = self.latency_cancel
        else:
            latency = 0

        if self.timestamp + latency in df.index:
            self.timestamp = self.timestamp + latency
            self.position_current_timestamp = df.index.get_loc(self.timestamp)

        else:
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              latency) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        self.source = event.transition.source

        return self.timestamp

    def cancelled_f(self, event):
        '''
        Add latency when entering the cancel state
        '''
        df = self.df
        if self.timestamp + self.latency_cancel in df.index:
            self.timestamp = self.timestamp + self.latency_cancel
            self.position_current_timestamp = df.index.get_loc(self.timestamp)
        else:
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_cancel) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def spot_balance_f(self, event):
        '''
        Add a spot_balance latency in order to enter the spot_balance state
        '''
        df = self.df
        # print('currently spot balance')
        if self.timestamp + self.latency_spot_balance in df.index:
            self.timestamp = self.timestamp + self.latency_spot_balance
            self.position_current_timestamp = df.index.get_loc(self.timestamp)

        else:
            self.position_current_timestamp = get_index_right(self.timestamps, self.position_current_timestamp,
                                                              self.latency_spot_balance) - 1
            self.timestamp = df.index[self.position_current_timestamp]

        return self.timestamp

    def compute_final_spread(self, event):
        '''
        Define the final spread:
        The computation uses the Swap price stored in the trying_to_post state and the real value of the Spot price
        '''
        df = self.df

        if self.side == 'entry':
            self.final_spread = spread_entry_func_numba(self.swap_price, df.loc[self.timestamp, 'price_spot_entry'],
                                                        self.swap_fee, self.spot_fee)
            self.final_spread_bp = None
            if self.use_bp:
                self.final_spread_bp = spread_entry_func_bp_numba(self.swap_price,
                                                                  df.loc[self.timestamp, 'price_spot_entry'],
                                                                  swap_fee=self.swap_fee, spot_fee=self.spot_fee)

        else:
            self.final_spread = spread_exit_func_numba(self.swap_price, df.loc[self.timestamp, 'price_spot_exit'],
                                                       self.swap_fee, self.spot_fee)
            self.final_spread_bp = None
            if self.use_bp:
                self.final_spread_bp = spread_exit_func_bp_numba(self.swap_price,
                                                                 df.loc[self.timestamp, 'price_spot_exit'],
                                                                 swap_fee=self.swap_fee, spot_fee=self.spot_fee)

    def volume_traded(self, event):
        '''
        function to compute the cumulative volume.
        if side = entry we add the traded volume, if the side = exit we subtract the volume.
        '''
        self.traded_volume = self.compute_volume_size()

        if self.side == 'entry':
            self.cum_volume = self.cum_volume + self.traded_volume
        elif self.side == 'exit':
            self.cum_volume = self.cum_volume - self.traded_volume

        self.total_volume_traded = self.total_volume_traded + self.traded_volume

    def quanto_loss_w_avg(self, event):
        '''
        function to compute the average weighted price for Quanto profit-loss.
        The average price is computed when cum_volume > 0  on the entry side,
        and when cum_volume < 0  on the exit side
        '''

        self.quanto_system.update_trade(self.timestamp, self.cum_volume, self.side, self.traded_volume)

    def band_funding_adjustment(self, event):
        if self.funding_system_name == '':
            return
        if self.timestamp > self.temp_funding_system.timestamp_last_update + self.temp_funding_system.update_interval:
            self.temp_funding_system.update(self.timestamp)
            self.entry_funding_adjustment, self.exit_funding_adjustment = self.temp_funding_system.band_adjustments()
            self.entry_funding_adjustment_to_zero, self.exit_funding_adjustment_to_zero = self.temp_funding_system.band_adjustments_to_zero()
            self.temp_funding_system.timestamp_last_update = self.timestamp

        self.df_array[self.idx_previous_band_adjustment: self.idx,
        self.columns_to_positions['Exit Band with Funding adjustment']] = self.exit_band_fn(event=None,
                                                                                            cum_volume=self.cum_volume)
        self.df_array[self.idx_previous_band_adjustment: self.idx,
        self.columns_to_positions['Entry Band with Funding adjustment']] = self.entry_band_fn(event=None,
                                                                                              cum_volume=self.cum_volume)
        if "Exit Band Exit to Zero" in list(self.df.columns):
            self.df_array[self.idx_previous_band_adjustment: self.idx,
            self.columns_to_positions['Exit Band Exit to Zero']] = self.exit_band_fn(event=None, cum_volume=1)
            self.df_array[self.idx_previous_band_adjustment: self.idx,
            self.columns_to_positions['Entry Band Enter to Zero']] = self.entry_band_fn(event=None, cum_volume=-1)

        self.idx_previous_band_adjustment = self.idx

    def quanto_loss_func(self, event):
        '''
        function to compute the quanto profit or quanto loss whenever conditions are met.
        In order to reduce the computation we have set a condition: for the quanto profit to be computed the previous
        computation must have happened some seconds in the past
        '''
        if self.timestamp - self.previous_timestamp >= 5000 or (event.transition.source == 'spot_balance'):
            self.quanto_system.update(self.timestamp, self.cum_volume)
            self.quanto_system.update_exponential_quanto(self.exp1)
            self.quanto_loss_pnl = self.quanto_system.quanto_loss_pnl
            self.quanto_loss = self.quanto_system.quanto_loss
            self.previous_timestamp = self.timestamp

        self.df_array[self.idx, self.columns_to_positions['quanto_profit']] = self.quanto_system.quanto_loss

        self.entry_quanto_adjustment, self.exit_quanto_adjustment = self.quanto_system.band_adjustments(
            entry_band=self.df_array[self.idx, self.columns_to_positions['Entry Band']],
            exit_band=self.df_array[self.idx, self.columns_to_positions['Exit Band']],
            move_exit=self.move_exit_above_entry, position=self.cum_volume)

        self.df_array[self.idx, self.columns_to_positions['Exit Band with Quanto loss']] = (
            self.exit_band_fn(event=None, cum_volume=self.cum_volume))

        self.df_array[self.idx, self.columns_to_positions['Entry Band with Quanto loss']] = self.entry_band_fn(
            event=None, cum_volume=self.cum_volume)

        if self.stop_trading and self.funding_system in ['Quanto_both', 'Quanto_both_extended', 'Quanto_loss']:
            if self.stop_trading and self.quanto_loss < - self.quanto_threshold:
                if self.high_to_current:
                    self.revert = True
                self.quanto_system.ratio_entry_band_mov = self.quanto_system.high_r
                self.stop_trading_enabled = True
                self.counter += 1

            # reset bands after stopping the trading
            if self.stop_trading_enabled and self.cum_volume <= 0 and (not self.allow_only_exit):
                self.stop_timestamp = self.timestamp
                self.allow_only_exit = True

    def trading_condition(self):
        if self.swap_instrument == 'ETHUSD' and self.exchange_swap == 'BitMEX':
            if self.funding_system == 'Quanto_profit' or self.funding_system == 'Quanto_profit_BOX' \
                    or self.funding_system == 'Quanto_profit_exp':
                if (self.side == 'entry') & (self.cum_volume >= 0):
                    return False
                elif self.side == 'entry':
                    return True
                if (self.side == 'exit') & (self.cum_volume <= - self.max_position):
                    return False
                elif self.side == 'exit':
                    return True
            elif self.funding_system == 'Quanto_loss':
                if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                    return False
                elif self.side == 'entry':
                    return True
                if (self.side == 'exit') & (self.cum_volume <= 0):
                    return False
                elif self.side == 'exit':
                    return True
            elif (self.funding_system == 'Quanto_both') | (self.funding_system == 'Quanto_both_extended'):
                if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                    return False
                elif self.side == 'entry':
                    return True
                if (self.side == 'exit') & (self.cum_volume <= - self.max_position):
                    return False
                elif self.side == 'exit':
                    return True
        elif self.net_trading == 'net_enter':
            if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                return False
            elif self.side == 'entry':
                return True
            if (self.side == 'exit') & (self.cum_volume <= 0):
                return False
            elif self.side == 'exit':
                return True
        elif self.net_trading == 'net_exit':
            if (self.side == 'entry') & (self.cum_volume >= 0):
                return False
            elif self.side == 'entry':
                return True
            if (self.side == 'exit') & (self.cum_volume <= -self.max_position):
                return False
            elif self.side == 'exit':
                return True
        else:
            if (self.side == 'entry') & (self.cum_volume >= self.max_position):
                return False
            elif self.side == 'entry':
                return True
            if (self.side == 'exit') & (self.cum_volume <= - self.max_position):
                return False
            elif self.side == 'exit':
                return True


def check_spread_and_depth_in_posted(side, spread_entry, entry_band, swap_price, price_swap_entry, targeted_depth_usd,
                                     spread_exit, exit_band, price_swap_exit):
    if side == 'entry':
        if (spread_entry < entry_band) | (swap_price - price_swap_entry > targeted_depth_usd):
            return True
        return False
    else:
        if (spread_exit > exit_band) | (price_swap_exit - swap_price > targeted_depth_usd):
            return True
        return False

