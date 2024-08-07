import datetime
import math
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.queries.queries import get_strategy_families, get_symbol, get_strategy_influx, get_exhange_names
from src.common.clients.rpc_client import RpcClient
from src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function import simulation_trader
from src.simulations.simulations_management.download_sweep_results import sweep_rerun_simulations

load_dotenv(find_dotenv())


def streamlit_trader_message():
    st.title('Taker Maker message to send to the Queue')
    st.sidebar.write("Trader Simulator in order to discover new markets and trading opportunities")
    st.sidebar.write('Click in the box in order to run a re-computation of a sweep in two different periods')
    recomp = st.sidebar.checkbox('Click here')

    band = st.sidebar.selectbox('Select the type of bands you want to use', ('bogdan_bands', 'percentage_bogdan_bands',
                                                                             'quanto_profit',
                                                                             'quanto_profit_additional',
                                                                             'percentage_band', 'custom_multi',
                                                                             'custom_multi_symmetrical',
                                                                             'custom_multi_custom'))

    if not recomp:
        date_range = st.date_input("Input a range of time report",
                                   [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
        t_start = int(
            datetime.datetime(year=date_range[0].year, month=date_range[0].month,
                              day=date_range[0].day).timestamp() * 1000)
        t_end = int(
            datetime.datetime(year=date_range[1].year, month=date_range[1].month,
                              day=date_range[1].day).timestamp() * 1000)
        st.write('The ending time in milliseconds', t_end)
        st.text('Default time-range is 1 day')
        st.write('Select the strategies family and strategy you want to review')
        st.write('If you want to review a new combination of exchanges select "Other"')

        col1, col2, col3 = st.columns(3)
        family = col1.selectbox('Strategy family', ('deribit_xbtusd', 'deribit_eth', 'Other'))
        environment = col2.selectbox('Environment from where data are downloaded', ('production', 'staging', 'server'))
        if band == 'custom_multi' or band == 'custom_multi_symmetrical' or band == 'custom_multi_custom':
            strategy = None
        else:
            if family == 'Other':
                strategy = col3.selectbox('Give the strategy name:', get_strategy_influx(environment=environment))
            elif family == 'deribit_xbtusd':
                strategy = col3.selectbox('Select the strategy',
                                          get_strategy_families(t0=t_start, environment='production')
                                          [family], index=11)
            elif family == 'deribit_eth':
                strategy = col3.selectbox('Select the strategy',
                                          get_strategy_families(t0=t_start, environment='production')
                                          [family])
            else:
                strategy = None

        col_1, col_2, col_3, col_4 = st.columns(4)
        if family == 'Other':
            exchange_spot = col_1.selectbox('ExchangeSpot',
                                            get_exhange_names(t0=t_start, t1=t_end, environment=environment))
            exchange_swap = col_2.selectbox('Exchange Swap',
                                            get_exhange_names(t0=t_start, t1=t_end, environment=environment))
            spot_instrument = col_3.selectbox('Spot Instrument',
                                              get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                         environment=environment))
            swap_instrument = col_4.selectbox('Swap Instrument',
                                              get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap,
                                                         environment=environment))
        else:
            exchange_spot = col_1.selectbox('Exchange Spot', ('Deribit', 'BitMEX'))
            exchange_swap = col_2.selectbox('Exchange Swap', ('BitMEX', 'Deribit'))
            spot_instrument = col_3.selectbox('Spot Instrument',
                                              get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                         environment=environment))
            swap_instrument = col_4.selectbox('Swap Instrument',
                                              get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap,
                                                         environment=environment))

        # move bogdan bands
        if exchange_spot == 'Deribit' and (spot_instrument == 'ETH-PERPETUAL' or spot_instrument == 'BTC-PERPETUAL'):
            move_bogdan_band = st.selectbox('Move band values based on Deribit Funding', ('No', 'move_entry',
                                                                                          'move_both', 'move_exit'))
        else:
            move_bogdan_band = 'No'

        fee_1, fee_2 = st.columns(2)
        maker_fee, taker_fee = exchange_fees(swap_exchange=exchange_swap, swap_symbol=swap_instrument,
                                             spot_exchange=exchange_spot, spot_symbol=spot_instrument)
        if math.isnan(maker_fee):
            maker_fee = -0.0001
        if math.isnan(taker_fee):
            taker_fee = 0.0003
        spot_fee = fee_1.number_input('Spot Fee', min_value=-1.0, value=taker_fee, max_value=1.0, step=0.0001,
                                      format="%.6f")
        swap_fee = fee_2.number_input('Swap Fee', min_value=-1.0, value=maker_fee, max_value=1.0, step=0.0001,
                                      format="%.6f")

        opt_1, opt_2, opt_3 = st.columns(3)
        area_spread_threshold = opt_1.number_input('Area Spread Threshold', min_value=0.0, value=0.0, max_value=100.0)
        funding_system = opt_2.selectbox('Select the funding system', ('No', 'Quanto_loss', 'Quanto_profit'))
        minimum_distance = opt_3.number_input('Select the minimum distance between Entry-Exit Band', min_value=0.0,
                                              value=0.0, max_value=100.0)

        qb1, qb2, qb3, qb4, qb5 = st.columns(5)
        minimum_value = qb1.number_input('Quanto Band minimum value', min_value=0.0, value=0.0, max_value=10.0)
        trailing_value = qb2.number_input('Quanto Band trailing value', min_value=0.0, value=0.0, max_value=10.0)
        disable_when_below = qb3.number_input('Quanto Band disable when', min_value=0.0, value=0.0, max_value=10.0)
        band_movement_perc = qb4.number_input('Define the band movement by a percentage', min_value=0.0, value=1.0,
                                              max_value=10.0)
        # stop trading parameter
        stop_trading = qb5.checkbox('Stop trading for 8 hours when exit band is above entry band', value=False)

        # latencies default values
        ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(exchange_swap=exchange_swap,
                                                                  exchange_spot=exchange_spot)

        # latencies
        col11, col12, col13 = st.columns(3)
        latency_spot = col11.number_input('Latency Spot', min_value=-1000, value=ws_spot, max_value=1000)
        latency_swap = col12.number_input('Latency Swap', min_value=-1000, value=ws_swap, max_value=1000)
        latency_try_post = col13.number_input('Latency Trying to Post', min_value=0, value=api_swap, max_value=1000)

        col_lat1, col_lat2 = st.columns(2)
        latency_cancel = col_lat1.number_input('Latency Cancel', min_value=0, value=api_swap, max_value=1000)
        latency_spot_balance = col_lat2.number_input('Latency Spot Balance', min_value=0, value=api_spot,
                                                     max_value=1000)

        col_ts1, col_ts2 = st.columns(2)
        max_trade_volume = col_ts1.number_input('Max Trade Volume', min_value=0, value=3000, max_value=100000,
                                                step=1000)
        max_position = col_ts2.number_input('Max Position', min_value=0, value=275000, max_value=100000000)

        if band == 'percentage_band':
            lookback = st.sidebar.text_input('lookback')
            recomputation_time = st.sidebar.text_input('recomputation_time')
            target_percentage_exit = st.sidebar.number_input('target_percentage_exit')
            target_percentage_entry = st.sidebar.number_input('target_percentage_entry')
            entry_opportunity_source = st.sidebar.selectbox('entry_opportunity_source', ('0', 'entry_with_takers',
                                                                                         'entry_with_takers_latency_200'))
            exit_opportunity_source = st.sidebar.selectbox('exit_opportunity_source', ('1', 'exit_with_takers',
                                                                                       'exit_with_takers_latency_200'))
        elif band == 'custom_multi' or band == 'custom_multi_symmetrical' or band == 'custom_multi_custom':
            col_min, col_max, col_step = st.columns(3)
            window_size_min = col_min.number_input('window_size min value', min_value=0, max_value=10000)
            window_size_max = col_max.number_input('window_size max value', min_value=0, max_value=10000)
            window_size_step = col_step.number_input('window_size step', min_value=0, max_value=10000)

            entry_delta_spread_min = col_min.number_input('entry_delta_spread min value', min_value=-100.0,
                                                          max_value=+100.0)
            entry_delta_spread_max = col_max.number_input('entry_delta_spread max value', min_value=-100.0,
                                                          max_value=+100.0)
            entry_delta_spread_step = col_step.number_input('entry_delta_spread step', min_value=-100.0,
                                                            max_value=+100.0)

            if band != 'custom_multi_symmetrical':
                exit_delta_spread_min = col_min.number_input('exit_delta_spread min value', min_value=-100.0,
                                                             max_value=+100.0)
                exit_delta_spread_max = col_max.number_input('exit_delta_spread max value', min_value=-100.0,
                                                             max_value=+100.0)
                exit_delta_spread_step = col_step.number_input('exit_delta_spread step', min_value=-100.0,
                                                               max_value=+100.0)

            band_funding_system = st.selectbox('Band Funding System', ('No', 'funding_adjusted_exit_band',
                                                                       'funding_adjusted_exit_band_with_drop'))

            lookback = None,
            recomputation_time = None,
            target_percentage_exit = None,
            target_percentage_entry = None,
            entry_opportunity_source = None,
            exit_opportunity_source = None
        else:
            lookback = None,
            recomputation_time = None,
            target_percentage_exit = None,
            target_percentage_entry = None,
            entry_opportunity_source = None,
            exit_opportunity_source = None

        force_band_creation = st.sidebar.checkbox('Force the creation of the bands')
        if force_band_creation:
            window_size = st.sidebar.number_input("window_size", min_value=0, max_value=10000)
            entry_delta_spread = st.sidebar.number_input("entry_delta_spread", min_value=-10.0, max_value=10.0)
            exit_delta_spread = st.sidebar.number_input("exit_delta_spread", min_value=-10.0, max_value=10.0)
            band_funding_system = st.sidebar.selectbox('Band Funding System', ('No', 'funding_adjusted_exit_band',
                                                                               'funding_adjusted_exit_band_with_drop'))
        else:
            window_size = None
            entry_delta_spread = None
            exit_delta_spread = None
            band_funding_system = None

        st.subheader('When ready with parameter input, click the button to send the message')
        check = st.checkbox('Click Here')
        st.write('State of the checkbox: ', check)

        if check:
            if band == 'custom_multi':
                for ws in range(window_size_min, window_size_max, window_size_step):
                    for ens in np.arange(entry_delta_spread_min, entry_delta_spread_max, entry_delta_spread_step):
                        for exs in np.arange(exit_delta_spread_min, exit_delta_spread_max, exit_delta_spread_step):
                            body = {'t_start': t_start, 't_end': t_end, 'band': band,
                                    'window_size': ws,
                                    'entry_delta_spread': ens,
                                    'exit_delta_spread': exs,
                                    'band_funding_system': band_funding_system,
                                    'lookback': lookback, 'recomputation_time': recomputation_time,
                                    'target_percentage_entry': target_percentage_entry,
                                    'target_percentage_exit': target_percentage_exit,
                                    'entry_opportunity_source': entry_opportunity_source,
                                    'exit_opportunity_source': exit_opportunity_source,
                                    'family': family, 'environment': environment, 'strategy': strategy,
                                    'exchange_spot': exchange_spot,
                                    'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument,
                                    'swap_instrument': swap_instrument,
                                    'spot_fee': spot_fee, 'swap_fee': swap_fee,
                                    'area_spread_threshold': area_spread_threshold,
                                    'latency_spot': latency_spot, 'latency_swap': latency_swap,
                                    'latency_try_post': latency_try_post,
                                    'latency_cancel': latency_cancel, 'latency_spot_balance': latency_spot_balance,
                                    'max_trade_volume': max_trade_volume, 'max_position': max_position,
                                    'funding_system': funding_system,
                                    'minimum_distance': minimum_distance,
                                    'minimum_value': minimum_value, 'trailing_value': trailing_value,
                                    'disable_when_below': disable_when_below,
                                    'force_band_creation': force_band_creation,
                                    'move_bogdan_band': move_bogdan_band,
                                    'band_movement_perc': band_movement_perc,
                                    'stop_trading': stop_trading,
                                    'function': 'simulation_trader'}
                            try:
                                client = RpcClient()
                                client.call(body)
                                time.sleep(5)
                            except:
                                simulation_trader(params=body)

            elif band == 'custom_multi_symmetrical':
                for ws in range(window_size_min, window_size_max, window_size_step):
                    for ens in np.arange(entry_delta_spread_min, entry_delta_spread_max, entry_delta_spread_step):
                        body = {'t_start': t_start, 't_end': t_end, 'band': band,
                                'lookback': lookback, 'recomputation_time': recomputation_time,
                                'window_size': ws,
                                'entry_delta_spread': ens,
                                'exit_delta_spread': ens,
                                'band_funding_system': band_funding_system,
                                'target_percentage_entry': target_percentage_entry,
                                'target_percentage_exit': target_percentage_exit,
                                'entry_opportunity_source': entry_opportunity_source,
                                'exit_opportunity_source': exit_opportunity_source,
                                'family': family, 'environment': environment, 'strategy': strategy,
                                'exchange_spot': exchange_spot,
                                'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument,
                                'swap_instrument': swap_instrument,
                                'spot_fee': spot_fee, 'swap_fee': swap_fee,
                                'area_spread_threshold': area_spread_threshold,
                                'latency_spot': latency_spot, 'latency_swap': latency_swap,
                                'latency_try_post': latency_try_post,
                                'latency_cancel': latency_cancel, 'latency_spot_balance': latency_spot_balance,
                                'max_trade_volume': max_trade_volume, 'max_position': max_position,
                                'funding_system': funding_system,
                                'minimum_distance': minimum_distance,
                                'minimum_value': minimum_value, 'trailing_value': trailing_value,
                                'disable_when_below': disable_when_below,
                                'force_band_creation': force_band_creation,
                                'move_bogdan_band': move_bogdan_band,
                                'band_movement_perc': band_movement_perc,
                                'stop_trading': stop_trading,
                                'function': 'simulation_trader'}
                        try:
                            client = RpcClient()
                            client.call(body)
                            time.sleep(5)
                        except:
                            simulation_trader(params=body)
            elif band == 'custom_multi_custom':
                for ws in [1800, 2000, 2500, 2700, 3300, 3500, 4000, 4200, 4400]:
                    for ens in np.arange(entry_delta_spread_min, entry_delta_spread_max, entry_delta_spread_step):
                        for exs in np.arange(exit_delta_spread_min, exit_delta_spread_max, exit_delta_spread_step):
                            if (int(ens * 10) % 2 == 0 and int(exs * 10) % 2 == 1) or (
                                    int(ens * 10) % 2 == 1 and int(exs * 10) % 2 == 0):
                                continue

                            body = {'t_start': t_start, 't_end': t_end, 'band': band,
                                    'window_size': ws,
                                    'entry_delta_spread': ens,
                                    'exit_delta_spread': exs,
                                    'band_funding_system': band_funding_system,
                                    'lookback': lookback, 'recomputation_time': recomputation_time,
                                    'target_percentage_entry': target_percentage_entry,
                                    'target_percentage_exit': target_percentage_exit,
                                    'entry_opportunity_source': entry_opportunity_source,
                                    'exit_opportunity_source': exit_opportunity_source,
                                    'family': family, 'environment': environment, 'strategy': strategy,
                                    'exchange_spot': exchange_spot,
                                    'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument,
                                    'swap_instrument': swap_instrument,
                                    'spot_fee': spot_fee, 'swap_fee': swap_fee,
                                    'area_spread_threshold': area_spread_threshold,
                                    'latency_spot': latency_spot, 'latency_swap': latency_swap,
                                    'latency_try_post': latency_try_post,
                                    'latency_cancel': latency_cancel, 'latency_spot_balance': latency_spot_balance,
                                    'max_trade_volume': max_trade_volume, 'max_position': max_position,
                                    'funding_system': funding_system,
                                    'minimum_distance': minimum_distance,
                                    'minimum_value': minimum_value, 'trailing_value': trailing_value,
                                    'disable_when_below': disable_when_below,
                                    'force_band_creation': force_band_creation,
                                    'move_bogdan_band': move_bogdan_band,
                                    'band_movement_perc': band_movement_perc,
                                    'stop_trading': stop_trading,
                                    'function': 'simulation_trader'}
                            try:
                                client = RpcClient()
                                client.call(body)
                                time.sleep(5)
                            except:
                                simulation_trader(params=body)

            else:
                body = {'t_start': t_start, 't_end': t_end, 'band': band,
                        'lookback': lookback, 'recomputation_time': recomputation_time,
                        'window_size': window_size,
                        'entry_delta_spread': entry_delta_spread,
                        'exit_delta_spread': exit_delta_spread,
                        'band_funding_system': band_funding_system,
                        'target_percentage_entry': target_percentage_entry,
                        'target_percentage_exit': target_percentage_exit,
                        'entry_opportunity_source': entry_opportunity_source,
                        'exit_opportunity_source': exit_opportunity_source,
                        'family': family, 'environment': environment, 'strategy': strategy,
                        'exchange_spot': exchange_spot,
                        'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument,
                        'swap_instrument': swap_instrument,
                        'spot_fee': spot_fee, 'swap_fee': swap_fee,
                        'area_spread_threshold': area_spread_threshold,
                        'latency_spot': latency_spot, 'latency_swap': latency_swap,
                        'latency_try_post': latency_try_post,
                        'latency_cancel': latency_cancel, 'latency_spot_balance': latency_spot_balance,
                        'max_trade_volume': max_trade_volume, 'max_position': max_position,
                        'funding_system': funding_system,
                        'minimum_distance': minimum_distance,
                        'minimum_value': minimum_value, 'trailing_value': trailing_value,
                        'disable_when_below': disable_when_below,
                        'force_band_creation': force_band_creation,
                        'move_bogdan_band': move_bogdan_band,
                        'band_movement_perc': band_movement_perc,
                        'function': 'simulation_trader',
                        'stop_trading': stop_trading,
                        're_compute_simulations': None}
                st.error(f"Do you really, really, wanna do this? Please check again the parameters {body}")
                if st.button("Yes I'm ready. I have double checked them"):
                    try:
                        client = RpcClient()
                        client.call(body)
                    except:
                        simulation_trader(params=body)
    else:
        col1, col2 = st.columns(2)
        num_input = col1.number_input('Number of simulations we want to execute', min_value=1, value=30, max_value=100)
        sweep_id = col1.text_input('Enter the sweep id from wandb')

        if st.button("Start the proccess"):
            t_start_lst = [1656028800000, 1654905600000]
            t_end_lst = [1656633600000, 1655596800000]
            params_df = sweep_rerun_simulations(sweep_id=sweep_id, select_num_simulations=num_input)
            try:
                params_df['minimum_value']
                params_df['trailing_value']
                params_df['disable_when_below']
                params_df['band_movement_perc']
            except:
                params_df['minimum_value'] = None
                params_df['trailing_value'] = None
                params_df['disable_when_below'] = None
                params_df['band_movement_perc'] = None

            for t_start, t_end in zip(t_start_lst, t_end_lst):
                my_bar = st.progress(0)
                for idx in range(num_input):
                    my_bar.progress(idx / num_input)
                    body = {'t_start': int(t_start),
                            't_end': int(t_end),
                            'band': params_df['band'].iloc[idx],
                            'lookback': params_df['lookback'].iloc[idx],
                            'recomputation_time': params_df['recomputation_time'].iloc[idx],
                            'window_size': int(params_df["window_size"].iloc[idx]),
                            'entry_delta_spread': params_df['entry_delta_spread'].iloc[idx],
                            'exit_delta_spread': params_df['exit_delta_spread'].iloc[idx],
                            'band_funding_system': params_df['band_funding_system'].iloc[idx],
                            'target_percentage_entry': params_df['target_percentage_entry'].iloc[idx],
                            'target_percentage_exit': params_df['target_percentage_exit'].iloc[idx],
                            'entry_opportunity_source': params_df['entry_opportunity_source'].iloc[idx],
                            'exit_opportunity_source': params_df['exit_opportunity_source'].iloc[idx],
                            'family': params_df['family'].iloc[idx],
                            'environment': params_df['environment'].iloc[idx],
                            'strategy': params_df['strategy'].iloc[idx],
                            'exchange_spot': params_df['exchange_spot'].iloc[idx],
                            'exchange_swap': params_df['exchange_swap'].iloc[idx],
                            'spot_instrument': params_df['spot_instrument'].iloc[idx],
                            'swap_instrument': params_df['swap_instrument'].iloc[idx],
                            'spot_fee': params_df['spot_fee'].iloc[idx],
                            'swap_fee': params_df['swap_fee'].iloc[idx],
                            'area_spread_threshold': int(params_df['area_spread_threshold'].iloc[idx]),
                            'latency_swap': int(params_df['latency_swap'].iloc[idx]),
                            'latency_spot': int(params_df['latency_spot'].iloc[idx]),
                            'latency_try_post': int(params_df['latency_try_post'].iloc[idx]),
                            'latency_cancel': int(params_df['latency_cancel'].iloc[idx]),
                            'latency_spot_balance': int(params_df['latency_spot_balance'].iloc[idx]),
                            'max_trade_volume': int(params_df['max_trade_volume'].iloc[idx]),
                            'max_position': int(params_df['max_position'].iloc[idx]),
                            'funding_system': params_df['funding_system'].iloc[idx],
                            'minimum_distance': params_df['minimum_distance'].iloc[idx],
                            'minimum_value': params_df['minimum_value'].iloc[idx],
                            'trailing_value': params_df['trailing_value'].iloc[idx],
                            'disable_when_below': params_df['disable_when_below'].iloc[idx],
                            'force_band_creation': True,
                            'function': 'simulation_trader',
                            're_compute_simulations': recomp}
                    try:
                        # types1 = [type(k) for k in body.values()]
                        # st.write(types1)
                        # st.write(body)
                        client = RpcClient()
                        client.call(body)
                        time.sleep(2)
                    except:
                        simulation_trader(params=body)
            st.write('All messages have been send to process')
