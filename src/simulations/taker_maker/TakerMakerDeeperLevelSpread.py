from transitions import Machine
import numpy as np

from pytictoc import TicToc

import warnings

from src.common.constants.constants import one_day
from src.common.utils.quanto_utils import bitmex_btc_prices, bitmex_eth_prices
from src.simulations.taker_maker.TakerMakerFunctions import TakerMakerFunctions, LimitOrder, TakerExecution, \
    get_taker_trades

warnings.filterwarnings("ignore")
t = TicToc()
t.tic()


class TakerMakerDeeperLevelSpread(TakerMakerFunctions):
    # Define the states of the trading:
    states = [{'name': 'clear', 'on_exit': ['reset_depth']},
              {'name': 'trying_to_post',
               'on_exit': ['swap_value', 'posting_f', 'trying_to_post_counter', 'keep_time_post']},
              {'name': 'not_posted'},
              {'name': 'posted'},
              {'name': 'try_to_cancel'},
              {'name': 'executing', 'on_enter': ['executing_f']},
              {'name': 'cancelled', 'on_enter': ['cancelled_f'], 'on_exit': ["cancel_open_swap"]},
              {'name': 'spot_balance', 'on_enter': ['spot_balance_f'],
               "on_exit": ["update_executions", "cancel_open_swap"]}]

    def __init__(self, df, spot_fee, swap_fee, area_spread_threshold, latency_spot, latency_swap, latency_try_post,
                 latency_cancel, latency_spot_balance, max_position, max_trade_volume, environment, exchange_swap,
                 swap_instrument, spot_instrument, funding_system, minimum_distance, minimum_value, trailing_value,
                 disable_when_below, depth_posting_predictor, swap_market_tick_size=0.5, verbose=False):
        self.source = None
        self.final_spread = 0
        self.df = df
        self.timestamp = df.index[100]
        self.timestamps = np.array(self.df.timems)
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
        self.verbose = verbose

        self.exchange_swap = exchange_swap
        self.swap_instrument = swap_instrument
        self.spot_instrument = spot_instrument

        self.max_position = max_position
        self.max_trade_volume = max_trade_volume
        self.depth_posting_predictor = depth_posting_predictor

        self.funding_system = funding_system
        self.temporary_order_swap: LimitOrder = None
        self.limit_orders_swap: [LimitOrder] = []
        self.cancelled_orders_swap: [LimitOrder] = []
        self.executed_while_cancelling_orders_swap: [LimitOrder] = []
        self.executions = []
        self.taker_execution_spot: TakerExecution = None
        self.taker_execution_swap: TakerExecution = None
        self.need_to_post_deeper__ = False
        # In dollars
        self.swap_market_tick_size = swap_market_tick_size

        if self.funding_system == 'Quanto_loss' or self.funding_system == 'Quanto_profit':
            self.w_avg_price_btc = 0
            self.w_avg_price_eth = 0
            self.coin_volume = 0
            self.minimum_distance = minimum_distance
            self.df['Exit Band with Quanto loss'] = self.df['Exit Band']
            self.df['Entry Band with Quanto loss'] = self.df['Entry Band']
            self.quanto_loss = 0
            self.btc_idx = 0
            self.eth_idx = 0

            # Quanto Profit trailing values
            self.minimum_value = minimum_value
            self.trailing_value = trailing_value
            self.disable_when_below = disable_when_below
            self.max_quanto_profit = 0
            self.quanto_profit_triggered = False

        # initialize variables used in conditions
        self.cum_volume = 0
        self.list_trying_post_counter = []

        self.traded_volume = 0
        self.total_volume_traded = 0
        self.time_post = 0
        self.entry_band_adjustment = 0
        self.exit_band_adjustment = 0
        self.order_depth = 0
        self.predicted_depth = None
        # additional arguments for entry, exit_band_fn
        self.idx = 0
        self.previous_timestamp = 0

        if self.exchange_swap == 'BitMEX' and self.swap_instrument == 'ETHUSD' and (
                self.funding_system == 'Quanto_loss' or self.funding_system == 'Quanto_profit'):
            self.price_btc = bitmex_btc_prices(t0=self.df.index[0], t1=self.df.index[-1] + one_day,
                                               environment=self.environment)
            self.price_btc['timestamp'] = self.price_btc.Time.view(np.int64) // 10 ** 6
            self.price_btc.reset_index(drop=True, inplace=True)
            self.price_btc = self.price_btc[['timestamp', 'price_ask']]
            self.price_eth = bitmex_eth_prices(t0=self.df.index[0], t1=self.df.index[-1] + one_day,
                                               environment=self.environment)
            self.price_eth['timestamp'] = self.price_eth.Time.view(np.int64) // 10 ** 6
            self.price_eth.reset_index(drop=True, inplace=True)
            self.price_eth = self.price_eth[['timestamp', 'price_ask']]

        self.taker_volume_df = get_taker_trades(self.df.index[0], self.df.index[-1], self.exchange_swap,
                                                self.swap_instrument)

        base_after = ['move_time_forward']
        if self.funding_system == 'Quanto_loss' or self.funding_system == 'Quanto_profit':
            base_after.append('quanto_loss_func')

        # Initialize the state machine
        self.machine = Machine(model=self, states=TakerMakerDeeperLevelSpread.states, send_event=True, initial='clear')

        # At spread not available
        self.machine.add_transition(trigger='initial_condition', source='clear', dest='clear',
                                    unless=['try_post_condition'], after=base_after)

        # At spread available and area spread above area spread threshold
        self.machine.add_transition(trigger='initial_condition', source='clear', dest='trying_to_post',
                                    conditions=['try_post_condition'])

        # Since we are in trying to post state we have to check the condition and if it is not meet we move
        # to not posted state
        self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='not_posted',
                                    unless=['try_to_post'], after=['update_time_latency_try_post'] + base_after)

        # # Since we are in the no posting state we have to return to the beginning
        self.machine.add_transition(trigger='reset', source='not_posted', dest='clear', after=base_after)

        # If the condition is met we move from trying to post to posted state
        self.machine.add_transition(trigger='move_from_trying_post', source='trying_to_post', dest='posted',
                                    conditions=['try_to_post'],
                                    after=['add_temp_order_to_orders', 'update_time_latency_try_post'] + base_after)

        # From posted state we move to try to cancel state
        self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                    conditions=['spread_unavailable_post'], after=['reset_depth'])
        self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                    conditions=['need_to_post_deeper'], after=['reset_depth'])
        self.machine.add_transition(trigger='move_from_post', source='posted', dest='try_to_cancel',
                                    conditions=['is_order_too_deep'])

        # Or from posted state we can move to executing state
        self.machine.add_transition(trigger='move_from_post', source='posted', dest='executing',
                                    conditions=['swap_price_movement_correct'],
                                    after=['update_order_after_executed', 'reset_depth'])

        # loop in the posted state until spread is available
        self.machine.add_transition(trigger='move_from_post', source='posted', dest='posted',
                                    conditions=['swap_price_no_movement'], unless=['need_to_post_deeper'],
                                    after=base_after + ['reset_depth'])

        # From try to cancel state we can move to two directions either cancelled or executing
        self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='executing',
                                    conditions=['activation_function_cancel'],
                                    after=['update_order_after_executed', 'reset_boolean_depth'])

        self.machine.add_transition(trigger='move_from_try_cancel', source='try_to_cancel', dest='cancelled',
                                    unless=['activation_function_cancel'], after=['reset_boolean_depth'])

        # From spot balance go to initial state clear
        # From executing state we go to spot balance state
        self.machine.add_transition(trigger='reset', source=['cancelled', 'spot_balance'], dest='clear',
                                    after=base_after)

        # From executing state we go to spot balance state
        self.machine.add_transition(trigger='move_from_executing', source='executing', dest='spot_balance',
                                    after=['send_market_order_spot', 'volume_traded', 'quanto_loss_w_avg'])
