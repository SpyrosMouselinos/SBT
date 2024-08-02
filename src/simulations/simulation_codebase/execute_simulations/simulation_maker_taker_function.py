import datetime
import time
import os
import numpy as np
import wandb
from dotenv import load_dotenv, find_dotenv
from pytictoc import TicToc
import pandas as pd
from src.common.constants.constants import TAKER_MAKER_PROJECT, WANDB_ENTITY
from src.common.clients.backblaze_client import BackblazeClient
from tqdm import tqdm
from scipy.stats import bernoulli
from sklearn.linear_model import LinearRegression
import string, random
import urllib.parse

from src.common.queries.funding_queries import FundingRatiosParams, funding_implementation
from src.common.queries.queries import get_data_for_trader
from src.common.utils.utils import sharpe_sortino_ratio_fun
from src.simulations.simulation_codebase.core_code.base_new import TraderExpectedExecutions
from src.simulations.simulation_codebase.pnl_computation_functions.pnl_computation import compute_rolling_pnl
from src.simulations.simulation_codebase.quanto_systems.QuantoProfitSystem import PriceBoxParams
from src.streamlit.streamlit_page_taker_maker_at_depth import ConstantDepthPosting

load_dotenv(find_dotenv())


def pnl_smotheness_func(df):
    y_array = df['last'].to_numpy()
    x_array = np.arange(len(df)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_array, y_array)
    r_sq = model.score(x_array, y_array)
    aad = sum(np.abs(model.predict(x_array[1:]) / y_array[1:] - 1)) / len(df)
    avg_dist = np.mean(np.abs(
        (model.coef_ * x_array[1:] - y_array[1:] - model.intercept_) / np.sqrt(model.coef_ ** 2 + 1) / y_array[1:]))

    return r_sq, aad, avg_dist


def simulation_trader(params):
    t = TicToc()
    t.tic()

    # file_id = randint(10 ** 6, 10 ** 7)

    params_dict = dict(params)

    # band and parameters
    band = params['band']
    lookback = params['lookback']
    recomputation_time = params['recomputation_time']
    target_percentage_exit = params['target_percentage_exit']
    target_percentage_entry = params['target_percentage_entry']
    entry_opportunity_source = params['entry_opportunity_source']
    exit_opportunity_source = params['exit_opportunity_source']

    # start - end time
    t_start = params['t_start']
    t_end = params['t_end']

    # family and environment
    family = params['family']
    environment = params['environment']
    strategy = params['strategy']

    # exchanges
    exchange_spot = params['exchange_spot']
    exchange_swap = params['exchange_swap']
    spot_instrument = params['spot_instrument']
    swap_instrument = params['swap_instrument']
    # fees
    spot_fee = params['spot_fee']
    swap_fee = params['swap_fee']
    area_spread_threshold = params['area_spread_threshold']

    # parameter to define if we want to trade as net_enter or net_exit. If None then it trades both directions.
    net_trading = params.get('net_trading', None)

    # latancies
    latency_spot = params['latency_spot']
    latency_swap = params['latency_swap']
    latency_try_post = params['latency_try_post']

    latency_cancel = params['latency_cancel']
    latency_spot_balance = params['latency_spot_balance']

    # trade volume
    max_trade_volume = params['max_trade_volume']
    max_position = params['max_position']
    funding_system = params['funding_system']
    funding_window = params['funding_window']
    funding_periods_lookback = params.get('funding_periods_lookback', 0)
    slow_funding_window = params.get('slow_funding_window', 0)
    minimum_distance = params['minimum_distance']

    # quanto bands values
    minimum_value = params['minimum_value']
    trailing_value = params['trailing_value']
    disable_when_below = params['disable_when_below']

    # band variables
    window_size = params['window_size']
    entry_delta_spread = params['entry_delta_spread']
    exit_delta_spread = params['exit_delta_spread']
    band_funding_system = params['band_funding_system']

    # parameters for funding_continuous_weight_concept funding system
    hoursBeforeSwapList = params.get('hoursBeforeSwapList', [])
    slowWeightSwapList = params.get('slowWeightSwapList', [])
    fastWeightSwapList = params.get('fastWeightSwapList', [])
    hoursBeforeSpotList = params.get('hoursBeforeSpotList', [])
    slowWeightSpotList = params.get('slowWeightSpotList', [])
    fastWeightSpotList = params.get('fastWeightSpotList', [])
    use_same_values_generic_funding = params.get('use_same_values_generic_funding', False)

    # band variables on close
    window_size2 = params.get('window_size2', None)
    entry_delta_spread2 = params.get('entry_delta_spread2', None)
    exit_delta_spread2 = params.get('exit_delta_spread2', None)
    band_funding_system2 = params.get('band_funding_system2', None)
    funding_options = params.get('funding_options', None)

    ratio_entry_band_mov = params.get('ratio_entry_band_mov', 1)
    ratio_entry_band_mov_ind = params.get('ratio_entry_band_mov_ind', 0)
    ratio_exit_band_mov = params.get('ratio_exit_band_mov', 0)
    rolling_time_window_size = params.get('rolling_time_window_size', 0)

    force_band_creation = params['force_band_creation']
    move_bogdan_band = params.get('move_bogdan_band', "No")

    stop_trading = params.get('stop_trading', False)

    # variables to switch from current to high R values
    current_r = params.get('current_r', 0)
    high_r = params.get('high_r', 0)
    quanto_threshold = params.get('quanto_threshold', None)
    high_to_current = params.get('high_to_current', False)
    hours_to_stop = params.get('hours_to_stop', 0)
    move_exit_above_entry = params.get('move_exit_above_entry', False)

    # variables for ETHUSD extended sort-go-long
    ratio_entry_band_mov_long = params.get('ratio_entry_band_mov_long', 1)
    ratio_exit_band_mov_ind_long = params.get('ratio_exit_band_mov_ind_long', 0)
    rolling_time_window_size_long = params.get('rolling_time_window_size_long', 0)

    # exponent over the funding in bands set of parameters
    exponent1 = params.get('exponent1', None)
    exponent2 = params.get('exponent2', None)
    exponent3 = params.get('exponent3', None)
    rolling_time_window_size2 = params.get('rolling_time_window_size2', None)
    entry_upper_cap = params.get('entry_upper_cap', None)
    entry_lower_cap = params.get('entry_lower_cap', None)
    exit_upper_cap = params.get('exit_upper_cap', None)
    w_theoretical_qp_entry = params.get('w_theoretical_qp_entry', 1)
    w_real_qp_entry = params.get('w_real_qp_entry', 1)

    # paremeters for local minimum and maximum
    use_local_min_max = params.get('use_local_min_max', False)
    num_of_points_to_lookback_entry = params.get('num_of_points_to_lookback_entry', 0)
    num_of_points_to_lookback_exit = params.get('num_of_points_to_lookback_exit', 0)

    swap_market_tick_size = params.get('swap_market_tick_size', 0.5)
    depth = params.get('constant_depth', 0)
    price_box_basis_points = params.get('price_box_basis_points', None)
    price_box_upper_threshold = params.get('price_box_upper_threshold', None)
    price_box_lower_threshold = params.get('price_box_lower_threshold', None)
    price_box_aggr_window = params.get('price_box_aggr_window', None)
    price_box_span = params.get('price_box_span', None)
    price_box_entry_movement_ratio = params.get('price_box_entry_movement_ratio', 0)
    use_bp = params.get('use_bp', False)
    pair_name = params.get('pair_name', None)
    is_multicoin = params.get('multicoin', False)
    leverage = params.get('leverage', None)
    time_current_funding = params.get('funding_opp_time_c_funding', None)
    funding_system_name = params.get("funding_system_name", "")
    funding_ratios_temp_1 = params.get("funding_ratios_swap_to_zero_entry", None)
    funding_ratios_temp_2 = params.get("funding_ratios_swap_to_zero_exit", None)
    funding_ratios_temp_3 = params.get("funding_ratios_swap_entry", None)
    funding_ratios_temp_4 = params.get("funding_ratios_swap_exit", None)
    funding_ratios_temp_5 = params.get("funding_ratios_spot_to_zero_entry", None)
    funding_ratios_temp_6 = params.get("funding_ratios_spot_to_zero_exit", None)
    funding_ratios_temp_7 = params.get("funding_ratios_spot_entry", None)
    funding_ratios_temp_8 = params.get("funding_ratios_spot_exit", None)
    moving_average_window = params.get("moving_average_window", None)
    use_stored_bands = params.get("use_stored_bands", False)
    t_start_training = params.get('t_start_training', 0)
    t_end_training = params.get('t_end_training', 0)
    funding_ratios_params_swap = FundingRatiosParams(funding_ratios_temp_1, funding_ratios_temp_2,
                                                     funding_ratios_temp_3, funding_ratios_temp_4)
    funding_ratios_params_spot = FundingRatiosParams(funding_ratios_temp_5, funding_ratios_temp_6,
                                                     funding_ratios_temp_7, funding_ratios_temp_8)
    adjustment_entry_band = params.get('adjustment_entry_band', None)
    adjustment_exit_band = params.get('adjustment_exit_band', None)
    depth_posting_predictor = ConstantDepthPosting(depth)
    price_box_params = PriceBoxParams(basis_points=price_box_basis_points, upper_threshold=price_box_upper_threshold,
                                      lower_threshold=price_box_lower_threshold, aggr_window=price_box_aggr_window,
                                      span=price_box_span, entry_movement_ratio=price_box_entry_movement_ratio)

    # convert milliseconds to datetime
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)

    if funding_system == 'Quanto_profit_exp':
        minimum_distance = minimum_distance * (entry_delta_spread + exit_delta_spread)
    print("the procedure of collecting data has started")
    df, strategy_name_created = get_data_for_trader(t_start=t_start, t_end=t_end,
                                                    exchange_spot=exchange_spot,
                                                    spot_instrument=spot_instrument,
                                                    exchange_swap=exchange_swap,
                                                    swap_instrument=swap_instrument,
                                                    window_size=window_size, entry_delta_spread=entry_delta_spread,
                                                    exit_delta_spread=exit_delta_spread,
                                                    band_funding_system=band_funding_system,
                                                    hoursBeforeSwapList=hoursBeforeSwapList,
                                                    slowWeightSwapList=slowWeightSwapList,
                                                    fastWeightSwapList=fastWeightSwapList,
                                                    hoursBeforeSpotList=hoursBeforeSpotList,
                                                    slowWeightSpotList=slowWeightSpotList,
                                                    fastWeightSpotList=fastWeightSpotList,
                                                    funding_window=funding_window,
                                                    funding_periods_lookback=funding_periods_lookback,
                                                    slow_funding_window=slow_funding_window,
                                                    swap_fee=swap_fee, spot_fee=spot_fee, strategy=strategy,
                                                    area_spread_threshold=area_spread_threshold,
                                                    environment=environment,
                                                    band_type=band,
                                                    lookback=lookback,
                                                    recomputation_time=recomputation_time,
                                                    target_percentage_exit=target_percentage_exit,
                                                    target_percentage_entry=target_percentage_entry,
                                                    entry_opportunity_source=entry_opportunity_source,
                                                    exit_opportunity_source=exit_opportunity_source,
                                                    force_band_creation=force_band_creation,
                                                    move_bogdan_band=move_bogdan_band, use_bp=use_bp,
                                                    window_size2=window_size2,
                                                    exit_delta_spread2=exit_delta_spread2,
                                                    entry_delta_spread2=entry_delta_spread2,
                                                    band_funding_system2=band_funding_system2,
                                                    funding_options=funding_options, use_stored_bands=use_stored_bands)
    t.toc()
    if band == 'custom_multi' or band == 'custom_multi_symmetrical' or band == 'custom_multi_custom' or strategy == '' \
            or strategy is None:
        strategy = strategy_name_created
        params_dict['strategy'] = strategy
    prediction_emitter = None
    model = TraderExpectedExecutions(df=df, spot_fee=spot_fee, swap_fee=swap_fee,
                                     area_spread_threshold=area_spread_threshold, latency_spot=latency_spot,
                                     latency_swap=latency_swap, latency_try_post=latency_try_post,
                                     latency_cancel=latency_cancel, latency_spot_balance=latency_spot_balance,
                                     max_position=max_position, max_trade_volume=max_trade_volume,
                                     environment=environment, exchange_swap=exchange_swap, exchange_spot=exchange_spot,
                                     swap_instrument=swap_instrument, spot_instrument=spot_instrument,
                                     funding_system=funding_system, minimum_distance=minimum_distance,
                                     minimum_value=minimum_value, trailing_value=trailing_value,
                                     disable_when_below=disable_when_below, ratio_entry_band_mov=ratio_entry_band_mov,
                                     ratio_entry_band_mov_ind=ratio_entry_band_mov_ind,
                                     stop_trading=stop_trading,
                                     current_r=current_r, high_r=high_r, quanto_threshold=quanto_threshold,
                                     ratio_exit_band_mov=ratio_exit_band_mov,
                                     rolling_time_window_size=rolling_time_window_size,
                                     rolling_time_window_size_long=rolling_time_window_size_long,
                                     ratio_entry_band_mov_long=ratio_entry_band_mov_long,
                                     ratio_exit_band_mov_ind_long=ratio_exit_band_mov_ind_long,
                                     high_to_current=high_to_current, hours_to_stop=hours_to_stop,
                                     move_exit_above_entry=move_exit_above_entry,
                                     exponent1=exponent1, exponent2=exponent2, exponent3=exponent3,
                                     rolling_time_window_size2=rolling_time_window_size2,
                                     entry_upper_cap=entry_upper_cap,
                                     entry_lower_cap=entry_lower_cap,
                                     exit_upper_cap=exit_upper_cap,
                                     w_theoretical_qp_entry=w_theoretical_qp_entry,
                                     w_real_qp_entry=w_real_qp_entry,
                                     depth_posting_predictor=depth_posting_predictor,
                                     swap_market_tick_size=swap_market_tick_size,
                                     price_box_params=price_box_params, net_trading=net_trading, use_bp=use_bp,
                                     funding_system_name=funding_system_name,
                                     funding_window=funding_window,
                                     slow_funding_window=slow_funding_window,
                                     funding_options=funding_options,
                                     funding_ratios_params_spot=funding_ratios_params_spot,
                                     funding_ratios_params_swap=funding_ratios_params_swap,
                                     moving_average=moving_average_window,
                                     prediction_emitter=prediction_emitter,
                                     use_local_min_max=use_local_min_max,
                                     num_of_points_to_lookback_entry=num_of_points_to_lookback_entry,
                                     num_of_points_to_lookback_exit=num_of_points_to_lookback_exit)

    list_data = []
    list_executions = []
    list_cancelled = []
    list_executing = []
    list_not_posted = []
    duration_pos = []
    cancel_count = 0
    # previous_state = [True, True, True, True]
    pbar = tqdm(total=len(df))
    start = time.time_ns()
    rate_limit_list = []
    while model.timestamp < df.index[-1]:
        pbar.update(1)
        if True and os.getenv("ENV") == "DEBUG_LOCAL" and bernoulli.rvs(0.0002):
            print(
                f"Total time {time.time_ns() - start} Spread unavailable {model.time_spent_spread_unavailable}\tPrice movement correct {model.time_time_movement_correct}\tOrder too deep {model.time_order_too_deep}\tActiv func try cancel {model.time_activation_function_cancel}\tActiv func try post {model.time_activation_function_try_post}\tTry post {model.time_try_post_condition}")
        if model.is_clear():
            model.initial_condition()
            rate_limit_list.append([model.timestamp, len(model.list_trying_post_counter), model.rate_limit(),
                                    model.sec_counter, model.min_counter])
        if model.is_trying_to_post():
            model.move_from_trying_post()
            list_data.append([model.side, model.timestamp])
        if model.is_not_posted():
            if model.side == 'entry':
                list_not_posted.append([model.timestamp, model.state, model.side, model.spread_entry])
            elif model.side == 'exit':
                list_not_posted.append([model.timestamp, model.state, model.side, model.spread_exit])
            model.reset()
        if model.is_posted():
            if model.side == 'entry':
                list_not_posted.append([model.timestamp, model.state, model.side, model.spread_entry])
            elif model.side == 'exit':
                list_not_posted.append([model.timestamp, model.state, model.side, model.spread_exit])
            list_not_posted.append([model.timestamp, model.state, model.side, model.spread_exit])
            model.move_from_post()
        if model.is_try_to_cancel():
            model.move_from_try_cancel()
        if model.is_cancelled():
            cancel_count += 1
            model.reset()
            list_cancelled.append([model.timestamp, model.side])
        if model.is_executing():
            list_executing.append([model.timestamp, model.state, model.side, model.source])
            model.move_from_executing()
        if model.is_spot_balance():
            model.reset()
            if exchange_swap == 'BitMEX' and swap_instrument == 'ETHUSD' and exchange_spot == 'Deribit' \
                    and spot_instrument == 'ETH-PERPETUAL':
                if funding_system in ['Quanto_profit', 'Quanto_profit_BOX',
                                      'Quanto_profit_exp'] and model.cum_volume < 0:
                    list_executions.append(
                        [model.timestamp, model.side, model.final_spread, model.final_spread_bp, model.spread_entry,
                         model.traded_volume, model.swap_price, model.quanto_system.w_avg_price_btc,
                         model.quanto_system.w_avg_price_eth,
                         model.quanto_system.quanto_loss * model.traded_volume / model.swap_price])
                    if model.cum_volume >= 2 * model.max_trade_volume:
                        duration_pos.append([np.nan, np.nan, np.nan, model.timestamp, model.cum_volume])
                    elif model.cum_volume <= - 2 * model.max_trade_volume:
                        duration_pos.append([np.nan, model.cum_volume, np.nan, model.timestamp, model.cum_volume])
                    else:
                        duration_pos.append([np.nan, np.nan, model.cum_volume, model.timestamp, model.cum_volume])

                elif funding_system == 'Quanto_loss' and model.cum_volume > 0:
                    list_executions.append(
                        [model.timestamp, model.side, model.final_spread, model.final_spread_bp, model.spread_exit,
                         model.traded_volume, model.swap_price, model.quanto_system.w_avg_price_btc,
                         model.quanto_system.w_avg_price_eth,
                         model.quanto_system.quanto_loss * model.traded_volume / model.swap_price])
                    if model.cum_volume >= 2 * model.max_trade_volume:
                        duration_pos.append([model.cum_volume, np.nan, np.nan, model.timestamp, model.cum_volume])
                    elif model.cum_volume <= - 2 * model.max_trade_volume:
                        duration_pos.append([np.nan, model.cum_volume, np.nan, model.timestamp, model.cum_volume])
                    else:
                        duration_pos.append([np.nan, np.nan, model.cum_volume, model.timestamp, model.cum_volume])
                elif funding_system in ['Quanto_both', 'Quanto_both_extended']:
                    if model.cum_volume < 0:
                        trade = 'long'
                    else:
                        trade = 'short'
                    list_executions.append(
                        [model.timestamp, model.side, trade, model.final_spread, model.final_spread_bp,
                         model.spread_exit,
                         model.traded_volume, model.swap_price, model.quanto_system.w_avg_price_btc,
                         model.quanto_system.w_avg_price_eth,
                         model.quanto_system.quanto_loss * model.traded_volume / model.swap_price])
                    if model.cum_volume >= 2 * model.max_trade_volume:
                        duration_pos.append([model.cum_volume, np.nan, np.nan, model.timestamp, model.cum_volume])
                    elif model.cum_volume <= - 2 * model.max_trade_volume:
                        duration_pos.append([np.nan, model.cum_volume, np.nan, model.timestamp, model.cum_volume])
                    else:
                        duration_pos.append([np.nan, np.nan, model.cum_volume, model.timestamp, model.cum_volume])
                else:
                    list_executions.append(
                        [model.timestamp, model.side, model.final_spread, model.final_spread_bp, model.spread_exit,
                         model.traded_volume, model.swap_price])
                    if model.cum_volume >= 2 * model.max_trade_volume:
                        duration_pos.append([model.cum_volume, np.nan, np.nan, model.timestamp, model.cum_volume])
                    elif model.cum_volume <= - 2 * model.max_trade_volume:
                        duration_pos.append([np.nan, model.cum_volume, np.nan, model.timestamp, model.cum_volume])
                    else:
                        duration_pos.append([np.nan, np.nan, model.cum_volume, model.timestamp, model.cum_volume])
            else:
                list_executions.append(
                    [model.timestamp, model.side, model.final_spread, model.final_spread_bp, model.spread_exit,
                     model.traded_volume, model.swap_price])
                if model.cum_volume >= 2 * model.max_trade_volume:
                    duration_pos.append([model.cum_volume, np.nan, np.nan, model.timestamp, model.cum_volume])
                elif model.cum_volume <= - 2 * model.max_trade_volume:
                    duration_pos.append([np.nan, model.cum_volume, np.nan, model.timestamp, model.cum_volume])
                else:
                    duration_pos.append([np.nan, np.nan, model.cum_volume, model.timestamp, model.cum_volume])
    pbar.close()
    if funding_system in model.funding_system_list_fun():
        model.df['quanto_profit'] = model.df_array[:, model.columns_to_positions['quanto_profit']]
        model.df['Entry Band with Quanto loss'] = model.df_array[:,
                                                  model.columns_to_positions['Entry Band with Quanto loss']]
        model.df['Exit Band with Quanto loss'] = model.df_array[:,
                                                 model.columns_to_positions['Exit Band with Quanto loss']]
    if funding_system_name != '':
        model.df['Entry Band with Funding adjustment'] = model.df_array[:,
                                                         model.columns_to_positions[
                                                             'Entry Band with Funding adjustment']]
        model.df['Exit Band with Funding adjustment'] = model.df_array[:,
                                                        model.columns_to_positions['Exit Band with Funding adjustment']]
    t.toc()
    print('the simulations has ended')

    duration_pos_df = pd.DataFrame(duration_pos,
                                   columns=['in_pos_entry', 'in_pos_exit', 'out_pos', 'timems', 'traded_volume'])

    executions_origin = pd.DataFrame(list_executing, columns=['timems', 'current_state', 'side', 'previous_state'])

    duration_pos_df['Time'] = pd.to_datetime(duration_pos_df['timems'], unit='ms')
    if ((exchange_swap == 'BitMEX' and swap_instrument == 'ETHUSD' and exchange_spot == 'Deribit'
         and spot_instrument == 'ETH-PERPETUAL') or (exchange_spot == 'BitMEX' and
                                                     spot_instrument == 'ETHUSD' and
                                                     exchange_swap == 'Deribit' and
                                                     swap_instrument == 'ETH-PERPETUAL')
            and funding_system in model.funding_system_list_fun()):
        if funding_system not in ['Quanto_both', 'Quanto_both_extended']:
            simulated_executions = pd.DataFrame(list_executions,
                                                columns=['timems', 'side', 'executed_spread', 'executed_spread_bp',
                                                         'initial_spread', 'traded_volume', 'swap_price',
                                                         'weighted_avg_btc', 'weighted_avg_eth', 'quanto_profit'])
        else:
            simulated_executions = pd.DataFrame(list_executions,
                                                columns=['timems', 'side', 'trade', 'executed_spread',
                                                         'executed_spread_bp',
                                                         'initial_spread', 'traded_volume', 'swap_price',
                                                         'weighted_avg_btc', 'weighted_avg_eth', 'quanto_profit'])
    else:
        simulated_executions = pd.DataFrame(list_executions,
                                            columns=['timems', 'side', 'executed_spread', 'executed_spread_bp',
                                                     'initial_spread', 'traded_volume', 'swap_price'])
    simulated_executions['spread_diff'] = - simulated_executions['initial_spread'] + \
                                          simulated_executions['executed_spread']
    simulated_executions['Time'] = pd.to_datetime(simulated_executions['timems'], unit='ms')

    band_values = df
    band_values['timems'] = band_values.index
    band_values.drop_duplicates(subset=['Entry Band', 'Exit Band'], inplace=True)
    band_values.reset_index(drop=True, inplace=True)

    if 'Entry Band Enter to Zero' in band_values.columns:
        simulated_executions = pd.merge_ordered(band_values[['Entry Band', 'Entry Band Enter to Zero', 'Exit Band',
                                                             'Exit Band Exit to Zero', 'timems']],
                                                simulated_executions, on='timems')
        simulated_executions = pd.merge_ordered(duration_pos_df[['traded_volume', 'timems']],
                                                simulated_executions, on='timems', suffixes=('_cum', ''))
        simulated_executions['Entry Band Enter to Zero'].ffill(inplace=True)
        simulated_executions['Exit Band Exit to Zero'].ffill(inplace=True)
    else:
        simulated_executions = pd.merge_ordered(band_values[['Entry Band', 'Exit Band', 'timems']],
                                                simulated_executions, on='timems')

    simulated_executions['Entry Band'].ffill(inplace=True)
    simulated_executions['Exit Band'].ffill(inplace=True)
    simulated_executions['exit_diff'] = np.nan
    simulated_executions['entry_diff'] = np.nan

    if 'Entry Band Enter to Zero' in simulated_executions.columns:
        for ix in simulated_executions[(~simulated_executions['side'].isna())].index:
            if simulated_executions.loc[ix, 'traded_volume_cum'] >= 0:
                exit_band = simulated_executions['Exit Band Exit to Zero'].iloc[ix]
                entry_band = simulated_executions['Entry Band'].iloc[ix]
            elif simulated_executions.loc[ix, 'traded_volume_cum'] < 0:
                exit_band = simulated_executions['Exit Band'].iloc[ix]
                entry_band = simulated_executions['Entry Band Enter to Zero'].iloc[ix]
            else:
                exit_band = 0
                entry_band = 0

            if simulated_executions.loc[ix, 'side'] == 'exit':
                simulated_executions.loc[ix, 'exit_diff'] = -simulated_executions.loc[ix, 'executed_spread'] + exit_band
            elif simulated_executions.loc[ix, 'side'] == 'entry':
                simulated_executions.loc[ix, 'entry_diff'] = simulated_executions.loc[
                                                                 ix, 'executed_spread'] - entry_band

    else:
        simulated_executions['exit_diff'] = -simulated_executions.loc[
            simulated_executions['side'] == 'exit', 'executed_spread'] + simulated_executions['Exit Band']
        simulated_executions['entry_diff'] = simulated_executions.loc[
                                                 simulated_executions['side'] == 'entry', 'executed_spread'] - \
                                             simulated_executions['Entry Band']

    simulated_executions = simulated_executions[(~simulated_executions['exit_diff'].isna()) |
                                                (~simulated_executions['entry_diff'].isna())]

    # additional filter for BitMEX XBTUSD to avoid outliers
    if (exchange_swap == "BitMEX" or exchange_spot == "BitMEX") and \
            (swap_instrument == "XBTUSD" or spot_instrument == "XBTUSD"):
        simulated_executions = simulated_executions[(abs(simulated_executions['exit_diff']) <= 30) |
                                                    (abs(simulated_executions['entry_diff']) <= 30)]

    simulated_executions.drop_duplicates(subset=['timems'], keep='last', inplace=True)
    simulated_executions['time_diff'] = simulated_executions['timems'].diff()
    simulated_executions['volume_over_price'] = \
        simulated_executions['traded_volume'] / simulated_executions['swap_price']

    # rate limit data
    rate_limit_df = pd.DataFrame(rate_limit_list, columns=['timestamp', 'counter', 'condition_state', 'second_limit',
                                                           'minute_limit'])
    # funding system should
    funding_spot, funding_swap, funding_total, spot_df, swap_df = funding_implementation(t0=t_start, t1=t_end,
                                                                                         swap_exchange=exchange_swap,
                                                                                         swap_symbol=swap_instrument,
                                                                                         spot_exchange=exchange_spot,
                                                                                         spot_symbol=spot_instrument,
                                                                                         position_df=duration_pos_df,
                                                                                         environment=environment)

    if funding_system in model.funding_system_list_fun():
        exec_df = pd.merge_ordered(simulated_executions[['timems', 'side', 'executed_spread']],
                                   model.df[['timems', 'Entry Band', 'Exit Band', 'Entry Band with Quanto loss',
                                             'Exit Band with Quanto loss']], on='timems')

        for col in ['Entry Band', 'Exit Band', 'Entry Band with Quanto loss', 'Exit Band with Quanto loss']:
            exec_df[col].ffill(inplace=True)
        exec_df.dropna(subset=['side'], inplace=True)
        try:
            exit_impact = len(exec_df[(exec_df['Exit Band'] < exec_df['Exit Band with Quanto loss']) &
                                      (exec_df['Exit Band'] < exec_df['executed_spread']) &
                                      (exec_df['side'] == 'exit')].index) / \
                          len(exec_df[exec_df['side'] == 'exit'].index)
        except ZeroDivisionError:
            exit_impact = 0.0
        exit_impact_perc = round(exit_impact, 4) * 100

        if ratio_entry_band_mov < 0:
            entry_impact = len(exec_df[(exec_df['Entry Band'] > exec_df['Entry Band with Quanto loss']) &
                                       (exec_df['Entry Band'] > exec_df['executed_spread']) &
                                       (exec_df['side'] == 'entry')].index) / \
                           len(exec_df[exec_df['side'] == 'entry'].index)
        else:
            try:
                entry_impact = len(exec_df[(exec_df['Entry Band'] < exec_df['Entry Band with Quanto loss']) &
                                           (exec_df['Entry Band with Quanto loss'] < exec_df['executed_spread']) &
                                           (exec_df['side'] == 'entry')].index) / \
                               len(exec_df[exec_df['side'] == 'entry'].index)
            except ZeroDivisionError:
                entry_impact = 0.0
        entry_impact_perc = round(entry_impact, 4) * 100

    else:
        entry_impact_perc = None
        exit_impact_perc = None

    # average entry - exit delta spread
    if len(simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume']) == 0:
        avg_entry_spread = np.nan
        if use_bp:
            avg_entry_spread_bp \
                = np.nan
    else:
        avg_entry_spread = np.sum(
            simulated_executions.loc[simulated_executions.side == 'entry', 'executed_spread'].values * \
            simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume'].values) / \
                           np.sum(
                               simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume'].values)
        if use_bp:
            avg_entry_spread_bp = np.sum(
                simulated_executions.loc[simulated_executions.side == 'entry', 'executed_spread_bp'].values * \
                simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume'].values) / \
                                  np.sum(
                                      simulated_executions.loc[
                                          simulated_executions.side == 'entry', 'traded_volume'].values)
    if len(simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume']) == 0:
        avg_exit_spread = np.nan
        if use_bp:
            avg_exit_spread_bp = np.nan
    else:
        avg_exit_spread = np.sum(
            simulated_executions.loc[simulated_executions.side == 'exit', 'executed_spread'].values * \
            simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume'].values) / \
                          np.sum(simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume'].values)
        if use_bp:
            avg_exit_spread_bp = np.sum(
                simulated_executions.loc[simulated_executions.side == 'exit', 'executed_spread_bp'].values * \
                simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume'].values) / \
                                 np.sum(
                                     simulated_executions.loc[
                                         simulated_executions.side == 'exit', 'traded_volume'].values)
    # dataframe with executions
    df_total = pd.merge_ordered(simulated_executions, executions_origin, on='timems')
    if pair_name is not None:
        df_total['pair_name'] = pair_name
        # df_total['time_current_funding'] = time_current_funding
    # days in period
    days_in_period = (t_end - t_start) // (1000 * 60 * 60 * 24)
    if days_in_period == 0:
        days_in_period = 1
    # simulation descriptive results
    if len(simulated_executions.describe().columns) >= 8:
        entry_exec_q = round(simulated_executions.describe()['entry_diff'].iloc[1], 2)

        exit_exec_q = round(simulated_executions.describe()['exit_diff'].iloc[1], 2)

        min_coin_volume = min(
            simulated_executions.loc[simulated_executions.side == 'exit', 'volume_over_price'].sum(),
            simulated_executions.loc[simulated_executions.side == 'entry', 'volume_over_price'].sum())

        if pd.isna(min_coin_volume):
            min_coin_volume = 0

        pnl = (avg_entry_spread - avg_exit_spread) * min_coin_volume
        if pd.isna(pnl):
            pnl = 0
        if adjustment_entry_band is not None and adjustment_exit_band is not None:
            pnl_adj = pnl + adjustment_entry_band * min_coin_volume + adjustment_exit_band * min_coin_volume

        if params.get('adjust_pnl_automatically', False):
            if entry_exec_q < params.get('maximum_quality', 1000000):
                entry_adj = 0
            else:
                entry_adj = params.get('maximum_quality', 1000000) - abs(entry_exec_q)
            if exit_exec_q < params.get('maximum_quality', 1000000):
                exit_adj = 0
            else:
                exit_adj = params.get('maximum_quality', 1000000) - abs(exit_exec_q)

            pnl_adj = pnl + entry_adj * min_coin_volume + exit_adj * min_coin_volume
        else:
            pnl_adj = pnl

        simulation_describe = pd.DataFrame(
            {'Entry Execution Quality': entry_exec_q,
             'Exit Execution Quality': exit_exec_q,
             'Successfully Cancelled': cancel_count,
             'Total Traded Volume in this period': simulated_executions['traded_volume'].sum(),
             'Total Traded Volume in this Period in Coin Volume': round(simulated_executions['volume_over_price'].sum(),
                                                                        4),
             'Estimated PNL': pnl,
             'Average Daily Traded Volume in this period':
                 int(simulated_executions['traded_volume'].sum() // days_in_period),
             'Average Daily Traded Volume in this period in Coin':
                 round(simulated_executions['volume_over_price'].sum() / days_in_period, 4),
             'Average Entry Spread': round(avg_entry_spread, 2) if not use_bp else round(avg_entry_spread_bp, 2),
             'Average Exit Spread': round(avg_exit_spread, 2) if not use_bp else round(avg_exit_spread_bp, 2),
             'Avg Fixed Spread': round(avg_entry_spread - avg_exit_spread, 2) if not use_bp else round(
                 avg_entry_spread_bp - avg_exit_spread_bp, 2),
             'Funding in Spot Market': funding_spot,
             'Funding in Swap Market': funding_swap,
             'Funding Total': funding_total,
             'Estimated PNL with Funding': round(pnl + funding_total, 4),
             'Net Trading': net_trading
             }, index=[0])
        simulation_describe.fillna(0, inplace=True)
    else:
        pnl = 0.0
        entry_exec_q = 0
        exit_exec_q = 0
        simulation_describe = pd.DataFrame(
            {'Entry Execution Quality': 0.0,
             'Exit Execution Quality': 0.0,
             'Successfully Cancelled': 0.0,
             'Total Traded Volume in this period': 0.0,
             'Total Traded Volume in this Period in Coin Volume': 0.0,
             'Average Daily Traded Volume in this period': 0.0,
             'Average Daily Traded Volume in this period in Coin': 0.0,
             'Average Entry Spread': 0.0,
             'Average Exit Spread': 0.0,
             'Avg Fixed Spread': 0.0,
             'Funding in Spot Market': 0.0,
             'Funding in Swap Market': 0.0,
             'Funding Total': 0.0,
             'Estimated PNL with Funding': pnl,
             'Net Trading': net_trading
             }, index=[0])

    # send a message for simulation end
    now = datetime.datetime.now()
    try:
        if params['re_compute_simulations']:
            strategy = strategy + '_' + 'recomputed'
    except:
        print('no re-computation parameter is given')
    current_date = datetime.datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    file_id = f'{current_month}_{current_year}+' + ''.join(random.choices(string.ascii_letters + string.digits, k=24))
    page_in_report = "generate_report_maker_taker" if not is_multicoin else "generate_report_maker_taker_multicoin"
    data = {
        "message": f"Simulation of {strategy} from {date_start} to {date_end}"
                   f"duration: {int(t.tocvalue())} sec  "
                   f"Entry Exec Quality: {round(simulation_describe.loc[0, 'Entry Execution Quality'], 2)} "
                   f"Exit Exec Quality: {round(simulation_describe.loc[0, 'Exit Execution Quality'], 2)}"
                   f" Average Fixed Spread: {round(avg_entry_spread - avg_exit_spread, 2)} "
                   f" Total Num of Executions: {int(simulated_executions.describe().iloc[0, 0])}"
                   f" link: https://streamlit.staging.equinoxai.com/{page_in_report}?file_id={file_id}",
    }
    # requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(data), headers={
    #     "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})

    if funding_system == 'Quanto_profit' or funding_system == 'Quanto_profit_BOX' or \
            funding_system == 'Quanto_profit_exp':
        qp = simulated_executions.loc[
                 simulated_executions['side'] == 'entry', 'quanto_profit'].sum() + model.quanto_loss_pnl
        simulation_describe['quanto_profit'] = qp
        qp_unrealised = model.quanto_loss_pnl
        simulation_describe['quanto_profit_unrealised'] = qp_unrealised
    elif funding_system == 'Quanto_loss':
        qp = simulated_executions.loc[
                 simulated_executions['side'] == 'exit', 'quanto_profit'].sum() + model.quanto_loss_pnl
        simulation_describe['quanto_profit'] = qp
        qp_unrealised = model.quanto_loss_pnl
        simulation_describe['quanto_profit_unrealised'] = qp_unrealised
    elif funding_system in ['Quanto_both', 'Quanto_both_extended']:
        qp = simulated_executions.loc[
                 (simulated_executions['side'] == 'exit') & (
                         simulated_executions['trade'] == 'short'), 'quanto_profit'].sum() \
             + simulated_executions.loc[
                 (simulated_executions['side'] == 'entry') & (
                         simulated_executions['trade'] == 'long'), 'quanto_profit'].sum() \
             + model.quanto_loss_pnl
        simulation_describe['quanto_profit'] = qp
        qp_unrealised = model.quanto_loss_pnl
        simulation_describe['quanto_profit_unrealised'] = qp_unrealised
    else:
        qp = 0
        qp_unrealised = 0
        simulation_describe['quanto_profit'] = 0
        simulation_describe['quanto_profit_unrealised'] = 0

    simulation_describe['quanto_profit'] = qp
    simulation_describe['quanto_profit_unrealized'] = qp_unrealised
    if price_box_params and model.funding_system == 'Quanto_profit' or model.funding_system == 'Quanto_profit_BOX':
        duration_pos_df = pd.merge_ordered(duration_pos_df, model.quanto_system.quanto_tp_signal, on='timems')
    if pair_name is not None:
        duration_pos_df['pair_name'] = pair_name
        # duration_pos_df['time_current_funding'] = time_current_funding
    volatility = None
    if funding_system == "Quanto_profit":
        october_14_to_17_volume_entry, october_26_to_28_volume_entry, november_08_volume_entry, dif1, dif2, dif3, dummy, dummy2 = \
            custom_formula_data(t0=t_start, t1=t_end, df=simulated_executions)
    else:
        october_14_to_17_volume_entry = None
        october_26_to_28_volume_entry = None
        november_08_volume_entry = None
        dif1, dif2, dif3, dummy, dummy2 = 0, 0, 0, 0, 0
    try:
        if params['re_compute_simulations']:
            if t_start == 1656028800000:
                volatility = 'low'
            elif t_start == 1654905600000:
                volatility = 'high'
            else:
                volatility = None
    except:
        volatility = None

    spot_df.reset_index(drop=True, inplace=True)
    swap_df.reset_index(drop=True, inplace=True)
    funding_df = pd.merge_ordered(spot_df, swap_df, on='timems', suffixes=['_spot', '_swap'])
    funding_df['cum_spot'] = funding_df['value_spot'].cumsum()
    funding_df['cum_swap'] = funding_df['value_swap'].cumsum()
    funding_df['total'] = funding_df['cum_spot'] + funding_df['cum_swap'].ffill()
    funding_df['Time'] = pd.to_datetime(funding_df['timems'], unit='ms')

    pnl_chart = compute_rolling_pnl(funding_df, simulated_executions, funding_system)

    pnl_chart.index = pnl_chart['Time']
    pnl_download = pnl_chart['pnl_generated_new'].resample('1D').last()
    pnl_download1 = pnl_chart['pnl_generated_new'].resample('1D').max() - pnl_chart['pnl_generated_new'].resample(
        '1D').min()
    pnl_download2 = pnl_chart['pnl_generated_new'].resample('1D').last() - pnl_chart['pnl_generated_new'].resample(
        '1D').first()
    pnl_daily_df = pd.concat([pnl_download, pnl_download1, pnl_download2], axis=1,
                             keys=['last', 'diff max-min', 'diff last-first'])

    num_of_days_in_period = (pnl_daily_df.index[-1] - pnl_daily_df.index[0]).days
    aum = int(2 * max_position / 9)
    groups_d = pnl_chart['pnl_generated_new'].groupby(pd.Grouper(freq='1D'))
    groups_w = pnl_chart['pnl_generated_new'].groupby(pd.Grouper(freq='1W'))
    drawdowns_d = []
    drawdowns_w = []
    for group_name, df_group in groups_d:
        try:
            max_before_min = df_group[:df_group.index.get_loc(df_group.idxmin())].max()
        except:
            max_before_min = np.nan
        if np.isnan(max_before_min):
            continue
        min_of_day = df_group.min()
        drawdowns_d.append([group_name, max_before_min - min_of_day])
    for group_name, df_group in groups_w:
        try:
            max_before_min = df_group[:df_group.index.get_loc(df_group.idxmin())].max()
        except:
            max_before_min = np.nan
        if np.isnan(max_before_min):
            continue
        min_of_day = df_group.min()
        drawdowns_w.append([group_name, max_before_min - min_of_day])

    drawups_d = []
    drawups_w = []
    for group_name, df_group in groups_d:
        try:
            min_before_max = df_group[:df_group.index.get_loc(df_group.idxmax())].min()
        except:
            min_before_max = np.nan
        if np.isnan(min_before_max):
            continue
        max_of_day = df_group.max()
        drawups_d.append([group_name, min_before_max - max_of_day])
    for group_name, df_group in groups_w:
        try:
            min_before_max = df_group[:df_group.index.get_loc(df_group.idxmax())].min()
        except:
            min_before_max = np.nan
        if np.isnan(min_before_max):
            continue
        max_of_day = df_group.max()
        drawups_w.append([group_name, min_before_max - max_of_day])

    if leverage is not None:
        aum = float(2 * max_position / leverage)
    max_drawdown_d = round(100 * np.array(drawdowns_d)[:, 1].astype(np.float64).max() / aum, 3)
    max_drawdown_w = round(100 * np.array(drawdowns_w)[:, 1].astype(np.float64).max() / aum, 3)

    try:
        max_drawup_d = abs(round(100 * np.array(drawups_d)[:, 1].astype(np.float64).min() / aum, 3))
    except:
        max_drawup_d = np.nan
    try:
        max_drawup_w = abs(round(100 * np.array(drawups_w)[:, 1].astype(np.float64).min() / aum, 3))
    except:
        max_drawup_w = np.nan

    mean_daily_ror, std_daily_ror, sharpe_ratio, sortino_ratio = sharpe_sortino_ratio_fun(df=pnl_daily_df, aum=aum,
                                                                                          t_start=t_start, t_end=t_end)
    try:
        ror_annualized = (pnl + funding_total + qp) / aum * 365 / \
                         num_of_days_in_period * 100
    except:
        ror_annualized = (pnl + funding_total) / aum * 365 / \
                         num_of_days_in_period * 100

    try:
        ror_annualized_adj = (pnl_adj + funding_total + qp) / aum * 365 / \
                             num_of_days_in_period * 100
    except:
        ror_annualized_adj = (pnl_adj + funding_total) / aum * 365 / \
                             num_of_days_in_period * 100

    r_sq, aad, avg_dist = pnl_smotheness_func(df=pnl_daily_df)
    if pair_name is not None:
        simulation_describe['pair_name'] = pair_name
        # simulation_describe['time_current_funding'] = time_current_funding
    wandb_summary = {'file_id': file_id,
                     'date_start': datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc).strftime(
                         "%m-%d-%Y %H:%M:%S"),
                     'date_end': datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc).strftime(
                         "%m-%d-%Y %H:%M:%S"),
                     # 'volatility': volatility,
                     # 'band': band,
                     # 'window_size': window_size,
                     # 'entry_delta_spread': entry_delta_spread,
                     # 'exit_delta_spread': exit_delta_spread,
                     # 'band_funding_system': band_funding_system,
                     # 'lookback': lookback,
                     # 'recomputation_time': recomputation_time,
                     # 'target_percentage_entry': target_percentage_entry,
                     # 'target_percentage_exit': target_percentage_exit,
                     # 'entry_opportunity_source': entry_opportunity_source,
                     # 'exit_opportunity_source': exit_opportunity_source,
                     'family': family,
                     'environment': environment,
                     'strategy': strategy,
                     'exchange_spot': exchange_spot,
                     'exchange_swap': exchange_swap,
                     'spot_instrument': spot_instrument,
                     'swap_instrument': swap_instrument,
                     'spot_fee': spot_fee,
                     'swap_fee': swap_fee,
                     'area_spread_threshold': area_spread_threshold,
                     'latency_spot': latency_spot,
                     'latency_swap': latency_swap,
                     'latency_try_post': latency_try_post,
                     'latency_cancel': latency_cancel,
                     'latency_spot_balance': latency_spot_balance,
                     'max_trade_volume': max_trade_volume,
                     'max_position': max_position,
                     'funding_system': funding_system,
                     'net_trading': net_trading,
                     # 'minimum_distance': minimum_distance,
                     # 'minimum_value': minimum_value,
                     # 'trailing_value': trailing_value,
                     # 'ratio_entry_band_mov': ratio_entry_band_mov,
                     # 'ratio_exit_band_mov': ratio_exit_band_mov,
                     # 'rolling_time_window_size': rolling_time_window_size,
                     'disable_when_below': disable_when_below,
                     # 'move_exit': move_exit_above_entry,
                     # 'price_box_basis_points': price_box_basis_points,
                     # 'price_box_upper_threshold': price_box_upper_threshold,
                     # 'price_box_lower_threshold': price_box_lower_threshold,
                     # 'price_box_aggr_window': price_box_aggr_window,
                     # 'price_box_span': price_box_aggr_window,
                     # 'price_box_entry_movement_ratio': price_box_entry_movement_ratio,
                     # 'function': 'simulation_trader',
                     # 'move_bogdan_band': move_bogdan_band,
                     'Entry Execution Quality': entry_exec_q,
                     'Exit Execution Quality': exit_exec_q,
                     'Successfully Cancelled': cancel_count,
                     'Total Traded Volume in this period': simulated_executions['traded_volume'].sum(),
                     'Total Traded Volume in this Period in Coin Volume':
                         round(simulated_executions['volume_over_price'].sum(), 4),
                     'Average Daily Traded Volume in this period':
                         int(simulated_executions['traded_volume'].sum() // days_in_period),
                     'Average Daily Traded Volume in this period in Coin':
                         round(simulated_executions['volume_over_price'].sum() / days_in_period, 4),
                     'Average Entry Spread': round(avg_entry_spread, 2),
                     'Average Exit Spread': round(avg_exit_spread, 2),
                     'Avg Fixed Spread': round(avg_entry_spread - avg_exit_spread, 2),
                     'Funding in Spot Market': round(funding_spot, 2),
                     'Funding in Swap Market': round(funding_swap, 2),
                     'Funding in Total': round(funding_total, 2),
                     'Total Traded Volume in Underlying Coin': round(
                         simulation_describe.loc[0, 'Total Traded Volume in this Period in Coin Volume'], 4),
                     'Coin Volume in entry side': simulated_executions.loc[
                         simulated_executions.side == 'entry', 'volume_over_price'].sum(),
                     'Coin Volume in exit side': simulated_executions.loc[
                         simulated_executions.side == 'exit', 'volume_over_price'].sum(),
                     # 'Entry 14 to 17 Volume Oct in USD': october_14_to_17_volume_entry,
                     # 'Entry 26 to 28 Volume Oct in USD': october_26_to_28_volume_entry,
                     # 'Entry 08 Nov Volume in USD': november_08_volume_entry,
                     # 'Net Entry 14 to 17 Oct in USD': dif1,
                     # 'Net Entry 26 to 28 Oct in USD': dif2,
                     # 'Net Entry 08 Nov in USD': dif3,
                     'Average Daily Traded Coin Volume in this period':
                         round(simulation_describe.loc[0, 'Average Daily Traded Volume in this period in Coin'], 2),
                     'Estimated PNL': pnl,
                     'Estimated PNL adj': pnl_adj,
                     'Estimated PNL with Funding': pnl + funding_total,
                     'Entry Impact': entry_impact_perc,
                     'Exit Impact': exit_impact_perc,
                     'Quanto Profit': qp,
                     'Quanto Profit Unrealised': qp_unrealised,
                     'Estimated PNL with Quanto_profit': pnl + funding_total + qp,
                     'Estimated PNL with Realized Quanto_profit': pnl + funding_total + qp - qp_unrealised,
                     'Estimated PNL with Funding minus Interest': pnl + funding_total - 0.09 * aum / 365 * days_in_period,
                     # 'F1': qp + pnl + 3 * funding_total,
                     # 'F2': qp - qp_unrealised + pnl + 5 * funding_total,
                     # 'F3': qp - qp_unrealised + pnl + 10 * funding_total,
                     # 'F4': pnl + 7.5 * qp + 15 * funding_total,
                     # 'F5': pnl + qp - qp_unrealised + 20 * funding_total,
                     # 'F6': pnl + qp - qp_unrealised + 40 * funding_total,
                     # 'F7': dummy * (dif1 / 275 + dif2 / 275 + dif3 / 275) + 1.8 * funding_total,
                     # 'F8': dummy2 * (pnl + funding_total + qp - qp_unrealised),
                     'Mean_daily_ROR': mean_daily_ror,
                     'Std_daily_ROR': std_daily_ror,
                     'Sharpe Ratio': sharpe_ratio,
                     'Sortino Ratio': sortino_ratio,
                     'ROR Annualized': ror_annualized,
                     'ROR Annualized adj': ror_annualized_adj,
                     'link': f"https://streamlit.staging.equinoxai.com/{page_in_report}?file_id={urllib.parse.quote(file_id)}",
                     'constant_depth': depth,
                     'max_drawdown_d': max_drawdown_d,
                     'max_drawdown_w': max_drawdown_w,
                     'max_drawup_d': max_drawup_d,
                     'max_drawup_w': max_drawup_w,
                     'r_squarred': r_sq,
                     'Average_Absolute_Deviation': aad,
                     'Average_Distance': avg_dist
                     }
    print("Simulation ended")
    t.toc()
    return (params_dict, simulation_describe, df_total, duration_pos_df, file_id, wandb_summary, band_values, model,
            pnl_daily_df, rate_limit_df)


def upload_to_backblaze(params_dict, simulation_describe, df_total, duration_pos_df, params, file_id, bands_df,
                        rate_limit_df):
    strategy = params['strategy']
    t_start = params['t_start']
    t_end = params['t_end']
    funding_system = params['funding_system']
    record_rate_limit = params.get('record_rate_limit', False)
    funding_system_name = params.get('funding_system_name', '')
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)
    # upload the results
    # print filenames
    print('file_names to be uploaded in backblaze')
    print(
        f'InputData_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')
    print(
        f'Parameters_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')
    print(
        f'Results_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')
    print(f'Sim_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')
    print(
        f'Position_duration_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')
    if record_rate_limit:
        print(
            f'Rate_Limit_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}')

    backblaze = BackblazeClient()
    backblaze.authorize()
    backblaze.get_upload_url(backblaze.get_bucket_id("equinoxai-trades-db"))

    if '/' in strategy:
        strategy_l = [s.replace('/', '-') for s in strategy]
        strategy = ''.join(strategy_l)
    try:
        if params['re_compute_simulations']:
            strategy = strategy + '_' + 'recomputed'
    except:
        print('no re-computation parameter is given')

    band_columns = [x for x in list(bands_df.columns) if " Band" in x]
    if params_dict['funding_system'] in TraderExpectedExecutions.funding_system_list_fun():
        band_columns = ["timems"] + band_columns + ['quanto_profit']
    else:
        band_columns = ["timems"] + band_columns
    if 'pair_name' in list(bands_df.columns):
        band_columns = band_columns + ['pair_name']
    # print("##################################### upload_to_backblaze ###############################################")
    # print(f"input data: {bands_df}")
    bands_to_upload = resample_band_df(bands_df, band_columns)
    # print("##################################### upload_to_backblaze ###############################################")
    # print(f'Uploading to backblaze bands: {bands_to_upload}')
    backblaze.upload_simulation(
        f'{file_id}_InputData_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
        bands_to_upload)

    backblaze.upload_simulation(
        f'{file_id}_Parameters_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
        params_dict)
    backblaze.upload_simulation(
        f'{file_id}_Results_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
        simulation_describe)
    backblaze.upload_simulation(
        f'{file_id}_Sim_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
        df_total)
    backblaze.upload_simulation(
        f'{file_id}_Position_duration_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
        duration_pos_df)
    if record_rate_limit:
        backblaze.upload_simulation(
            f'{file_id}_Rate_Limit_MT_{strategy}_from_{date_start.strftime("%m-%d-%Y")}_to_{date_end.strftime("%m-%d-%Y")}_id_{file_id}',
            rate_limit_df)


def resample_band_df(bands_df, band_columns):
    # print("##################################### resample_band_df ###############################################")
    # print(f"bands_df: {bands_df}")
    bands_df[band_columns].drop_duplicates(subset=band_columns, keep='last', inplace=True)
    bands_df[band_columns].reset_index(drop=True, inplace=True)
    bands_df['Time'] = pd.to_datetime(bands_df['timems'], unit='ms')
    entry_bands = [x for x in band_columns if "Entry" in x]
    entry_bands = entry_bands + ['timems', 'Time']
    # print(f"entry_bands: {entry_bands}")
    exit_bands = [x for x in band_columns if "Exit" in x]
    exit_bands = exit_bands + ['timems', 'Time']
    # print(f"exit_bands: {exit_bands}")
    bands_to_upload_entry = bands_df[entry_bands].resample('5T', on='Time').max()
    bands_to_upload_exit = bands_df[exit_bands].resample('5T', on='Time').min()
    bands_to_upload = pd.merge_ordered(bands_to_upload_entry, bands_to_upload_exit, on='timems')
    # print(f"bands_to_upload: {bands_to_upload}")
    if 'Time_x' in bands_to_upload.columns or 'Time_y' in bands_to_upload.columns:
        bands_to_upload = bands_to_upload.drop(columns=['Time_x', 'Time_y'])
    if 'Time' in bands_to_upload.columns:
        bands_to_upload = bands_to_upload.drop(columns=['Time'])
    return bands_to_upload


def upload_to_wandb(params, wandb_summary):
    if not params.get('already_logged_in', False):
        wandb.login(host=os.getenv("WANDB_HOST"))
        wandb.init(project=TAKER_MAKER_PROJECT, entity=WANDB_ENTITY)
    wandb.log(wandb_summary)
    wandb.finish()
    print("Uploaded to wandb")
    # except BaseException as e:
    # print(f"Something went wrong..{e}")


def upload_altcoin_summary_to_wandb(params):
    if not params.get('already_logged_in', False):
        wandb.login(host=os.getenv("WANDB_HOST"))
        wandb.init(project=TAKER_MAKER_PROJECT, entity=WANDB_ENTITY)
        wandb.log(params)
        wandb.finish()


def custom_formula_data(t0, t1, df):
    ########################################################################
    # 1665663600000 = Thu Oct 13 2022 12:20:00 UTC
    # 1665756000000 = Fri Oct 14 2022 14:00:00 UTC
    # 1666713600000 = Tue Oct 25 2022 16:00:00 UTC
    # 1666849200000 = Thu Oct 27 2022 05:40:00 UTC
    # 1667931600000 = Tue Nov 08 2022 18:20:00 UTC
    # 1667952000000 = Wed Nov 09 2022 00:00:00 UTC
    ########################################################################
    critical_dates = [1665663600000, 1665756000000, 1666713600000, 1666849200000, 1667931600000, 1667952000000]
    if all(x in range(t0, t1 + 100) for x in critical_dates):
        october_14_to_17_volume_entry = df.loc[
            (df.timems <= critical_dates[1]) & (
                    df.timems >= critical_dates[0]) & (
                    df.side == 'entry'), 'traded_volume'].sum()
        october_26_to_28_volume_entry = df.loc[
            (df.timems <= critical_dates[3]) & (
                    df.timems >= critical_dates[2]) & (
                    df.side == 'entry'), 'traded_volume'].sum()

        november_08_volume_entry = df.loc[
            (df.timems <= critical_dates[5]) & (
                    df.timems >= critical_dates[4]) & (
                    df.side == 'entry'), 'traded_volume'].sum()

        october_14_to_17_volume_exit = df.loc[
            (df.timems <= critical_dates[1]) & (df.timems >= critical_dates[0]) & (
                    df.side == 'exit'), 'traded_volume'].sum()
        october_26_to_28_volume_exit = df.loc[
            (df.timems <= critical_dates[3]) & (df.timems >= critical_dates[2]) & (
                    df.side == 'exit'), 'traded_volume'].sum()
        november_08_volume_exit = df.loc[
            (df.timems <= critical_dates[5]) & (
                    df.timems >= critical_dates[4]) & (
                    df.side == 'exit'), 'traded_volume'].sum()

    else:
        october_14_to_17_volume_entry = None
        october_26_to_28_volume_entry = None
        november_08_volume_entry = None
        october_14_to_17_volume_exit = None
        october_26_to_28_volume_exit = None
        november_08_volume_exit = None

    cr_dates_list = [october_14_to_17_volume_entry, october_26_to_28_volume_entry, november_08_volume_entry,
                     october_14_to_17_volume_exit, october_26_to_28_volume_exit, november_08_volume_exit]
    if all(x is not None for x in cr_dates_list):
        dif1 = october_14_to_17_volume_entry - october_14_to_17_volume_exit
        dif2 = october_26_to_28_volume_entry - october_26_to_28_volume_exit
        dif3 = november_08_volume_entry - november_08_volume_exit
    else:
        dif1 = 0
        dif2 = 0
        dif3 = 0

    if dif1 <= 10 ^ 5 or dif2 <= 10 ^ 5 or november_08_volume_entry <= 10 ^ 5:
        dummy = 0
    else:
        dummy = 1

    if dif1 <= 30000 or dif2 <= 10 ^ 5 or november_08_volume_entry <= 10 ^ 5:
        dummy2 = 0
    else:
        dummy2 = 1

    results = [october_14_to_17_volume_entry, october_26_to_28_volume_entry, november_08_volume_entry,
               dif1, dif2, dif3, dummy, dummy2]
    return results
