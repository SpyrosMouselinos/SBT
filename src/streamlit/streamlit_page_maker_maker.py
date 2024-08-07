import datetime
import numpy as np
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pytictoc import TicToc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.queries.queries import get_strategy_families, get_symbol, get_strategy_influx, get_exhange_names, \
    get_percentage_band_values, get_entry_exit_bands, get_data_for_trader
from src.common.queries.queries import get_band_values, get_executions
import hiplot as hip
import logging


logging.disable(logging.CRITICAL)
load_dotenv(find_dotenv())


class SimulationBase:
    pass


def streamlit_maker_maker_trader():
    t = TicToc()
    if st.button("Clear All"):
        # Clears all singleton caches:
        st.experimental_singleton.clear()
    # if os.path.exists('logs_trader.txt'):
    #     os.remove("logs_trader.txt")
    st.sidebar.write("Trader Simulator in order to discover new markets and trading opportunities")
    st.title('Trading Simulator')
    actual_exists = st.sidebar.selectbox('Are there any real executions recorded? :', ('no', 'yes'))
    st.sidebar.write('"Yes": if there are real executions, "No":  if there are no real executions')
    band = st.sidebar.selectbox('Select the type of bands you want to use', (
    'bogdan_bands', 'percentage_bogdan_bands', 'quanto_profit', 'quanto_profit_additional', 'percentage_band'))
    if band == 'percentage_band':
        lookback = st.sidebar.text_input('lookback')
        recomputation_time = st.sidebar.text_input('recomputation_time')
        target_percentage_exit = st.sidebar.number_input('target_percentage_exit')
        target_percentage_entry = st.sidebar.number_input('target_percentage_entry')
        entry_opportunity_source = st.sidebar.selectbox('entry_opportunity_source',
                                                        ('0', 'entry_with_takers', 'entry_with_takers_latency_200'))
        exit_opportunity_source = st.sidebar.selectbox('exit_opportunity_source',
                                                       ('1', 'exit_with_takers', 'exit_with_takers_latency_200'))
    else:
        lookback = None,
        recomputation_time = None,
        target_percentage_exit = None,
        target_percentage_entry = None,
        entry_opportunity_source = None,
        exit_opportunity_source = None

    date_range = st.date_input("Input a range of time report",
                               [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
    t_start = int(
        datetime.datetime(year=date_range[0].year, month=date_range[0].month, day=date_range[0].day).timestamp() * 1000)
    t_end = int(
        datetime.datetime(year=date_range[1].year, month=date_range[1].month, day=date_range[1].day).timestamp() * 1000)
    # t_start = 1653827775752
    # t_end = 1654080431553
    # t_end = t_start + 1000 * 60 * 60 * 2
    st.write('The ending time in milliseconds', t_end)
    st.text('Default time-range is 1 day')
    st.write('Select the strategies family and strategy you want to review')
    st.write('If you want to review a new combination of exchanges select "Other"')

    col1, col2, col3 = st.columns(3)
    family = col1.selectbox('Strategy family', ('deribit_xbtusd', 'deribit_eth', 'Other'))
    environment = col2.selectbox('Environment from where data are downloaded', ('production', 'staging', 'server'))
    if family == 'Other':
        strategy = col3.selectbox('Give the strategy name:', get_strategy_influx(environment=environment))
    elif family == 'deribit_xbtusd':
        strategy = col3.selectbox('Select the strategy',
                                  get_strategy_families(t0=t_start, environment='production')[family], index=9)
    else:
        strategy = col3.selectbox('Select the strategy',
                                  get_strategy_families(t0=t_start, environment='production')[family])

    col_1, col_2, col_3, col_4 = st.columns(4)

    if family == 'Other':
        exchange_spot = col_1.selectbox('ExchangeSpot',
                                        get_exhange_names(t0=t_start, t1=t_end, environment=environment))
        exchange_swap = col_2.selectbox('Exchange Swap',
                                        get_exhange_names(t0=t_start, t1=t_end, environment=environment))
        spot_instrument = col_3.selectbox('Spot Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                                        environment=environment))
        swap_instrument = col_4.selectbox('Swap Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap,
                                                                        environment=environment))
    else:
        exchange_spot = col_1.selectbox('Exchange Spot', ('Deribit', 'BitMEX'))
        exchange_swap = col_2.selectbox('Exchange Swap', ('BitMEX', 'Deribit'))
        spot_instrument = col_3.selectbox('Spot Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                                        environment=environment))
        swap_instrument = col_4.selectbox('Swap Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap,
                                                                        environment=environment), index=3)

    fee_col_1, fee_col_2, fee_col_3, fee_col_4 = st.columns(4)
    default_maker_fee_swap, default_taker_fee_swap = exchange_fees(exchange_swap, swap_instrument, exchange_swap,
                                                                   swap_instrument)
    default_maker_fee_spot, default_taker_fee_spot = exchange_fees(exchange_spot, spot_instrument, exchange_spot,
                                                                   spot_instrument)
    taker_fee_spot = fee_col_1.number_input('Spot Fee Taker', min_value=-1.0, value=default_taker_fee_spot,
                                            max_value=1.0, step=0.00001, format="%.6f")
    maker_fee_spot = fee_col_2.number_input('Spot Fee Maker', min_value=-1.0, value=default_maker_fee_spot,
                                            max_value=1.0, step=0.00001, format="%.6f")
    taker_fee_swap = fee_col_3.number_input('Swap Fee Taker', min_value=-1.0, value=default_taker_fee_swap,
                                            max_value=1.0, step=0.00001, format="%.6f")
    maker_fee_swap = fee_col_4.number_input('Swap Fee Maker', min_value=-1.0, value=default_maker_fee_swap,
                                            max_value=1.0, step=0.00001, format="%.6f")

    # latencies default values
    ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(exchange_swap, exchange_spot)
    # latencies
    latency_col_1, latency_col_2, latency_col_3, latency_col_4 = st.columns(4)
    latency_spot = latency_col_1.number_input('Latency Spot', min_value=0, value=ws_spot, max_value=1000)
    latency_try_post_spot = latency_col_2.number_input('Latency Trying to Post Spot', min_value=0, value=api_spot,
                                                       max_value=1000)
    latency_cancel_spot = latency_col_3.number_input('Latency Cancel Spot', min_value=0, value=api_spot, max_value=1000)
    latency_balance_spot = latency_col_4.number_input('Latency Balance Spot', min_value=0, value=api_swap,
                                                      max_value=1000)
    latency_col_5, latency_col_6, latency_col_7, latency_col_8 = st.columns(4)
    latency_swap = latency_col_5.number_input('Latency Swap', min_value=0, value=ws_swap, max_value=1000)
    latency_try_post_swap = latency_col_6.number_input('Latency Trying to Post Swap', min_value=0, value=api_swap,
                                                       max_value=1000)
    latency_cancel_swap = latency_col_7.number_input('Latency Cancel Swap', min_value=0, value=api_swap, max_value=1000)
    latency_balance_swap = latency_col_8.number_input('Latency Balance Swap', min_value=0, value=api_spot,
                                                      max_value=1000)

    slippage_col_1, slippage_col_2, displacement_col_1, area_spread_col_1 = st.columns(4)
    taker_slippage_spot = slippage_col_1.number_input('Slippage Spot', min_value=0.0, value=1.5, max_value=10)
    taker_slippage_swap = slippage_col_2.number_input('Slippage Swap', min_value=0.0, value=1.5, max_value=10)
    displacement = displacement_col_1.number_input('Displacement', min_value=0, value=3, max_value=50)
    area_spread_threshold = area_spread_col_1.number_input('Area Spread Threshold', min_value=0, value=0, max_value=100)

    col_ts1, col_ts2 = st.columns(2)
    max_trade_volume = col_ts1.number_input('Max Trade Volume', min_value=0, value=3000, max_value=100000)
    max_position = col_ts2.number_input('Max Position', min_value=0, value=100000, max_value=1000000)

    st.subheader('When ready with parameter input click the button')
    check = st.checkbox('Click Here')
    st.write('State of the checkbox: ', check)

    if check:
        t.tic()
        if band == 'percentage_band':
            band_values = get_percentage_band_values(t0=t_start, t1=t_end,
                                                     lookback="1700m",
                                                     recomputation_time="1h",
                                                     target_percentage_exit=10.7,
                                                     target_percentage_entry=8.0,
                                                     entry_opportunity_source="entry_with_takers_latency_200",
                                                     exit_opportunity_source="exit_with_takers_latency_200",
                                                     spot_name=exchange_spot,
                                                     spot_instrument=f"hybrid_{spot_instrument}",
                                                     swap_instrument=f"hybrid_{swap_instrument}",
                                                     environment=environment)
            band_values.rename(columns={'entry_band': 'Entry Band', 'exit_band': 'Exit Band'}, inplace=True)
        else:
            # band_values = get_band_values(t0=t_start, t1=t_end, typeb=band, strategy=strategy, environment=environment)
            band_values = get_entry_exit_bands(t_start=t_start, t_end=t_end, strategy=strategy, entry_delta_spread=0,
                                               exit_delta_spread=0, btype='central_band', environment=environment)
            # 6 is currently the hardcoded value for the entry delta spread for this strategy
            # band_values["Entry Band"] = band_values["Entry Band"] - 4.25

        band_values.rename(columns={'Band': 'Central Band'}, inplace=True)

        interval = 1000 * 60 * 60 * 24
        t0 = t_start
        t1 = t_end
        df = get_data_for_trader(t0, t1, exchange_spot, spot_instrument, exchange_swap, swap_instrument,
                                 taker_fee_spot=taker_fee_spot, maker_fee_spot=maker_fee_spot,
                                 taker_fee_swap=taker_fee_swap,
                                 maker_fee_swap=maker_fee_swap, strategy=strategy,
                                 area_spread_threshold=area_spread_threshold,
                                 environment=environment)

        st.write(f'Time elapsed to create the dataframe {t.tocvalue()} sec')

        t.tic()
        model = SimulationBase(df=df, maker_fee_swap=maker_fee_swap, taker_fee_swap=taker_fee_swap,
                               maker_fee_spot=maker_fee_spot,
                               taker_fee_spot=taker_fee_spot, area_spread_threshold=area_spread_threshold,
                               spot_instrument=spot_instrument, swap_instrument=swap_instrument,
                               latency_spot=latency_spot, latency_swap=latency_swap,
                               latency_try_post_spot=latency_try_post_spot,
                               latency_try_post_swap=latency_try_post_swap, latency_cancel_spot=latency_cancel_spot,
                               latency_cancel_swap=latency_cancel_swap, latency_balance_spot=latency_balance_spot,
                               latency_balance_swap=latency_balance_swap, displacement=displacement,
                               taker_slippage_spot=taker_slippage_spot, taker_slippage_swap=taker_slippage_swap,
                               max_position=max_position, max_trade_volume=max_trade_volume, environment=environment)

        list_executing = []
        list_not_posted = []

        while model.timestamp < t_end - 1000 * 60 * 5:
            # if model.state == 'clear':
            #     if model.timestamp > t1 - 1000 * 60 * 10:
            #         model.df = get_data_for_trader(t0, t1, exchange_spot, spot_instrument, exchange_swap,
            #                                        swap_instrument, taker_fee_spot=taker_fee_spot,
            #                                        maker_fee_spot=maker_fee_spot, taker_fee_swap=taker_fee_swap,
            #                                        maker_fee_swap=maker_fee_swap, strategy=strategy,
            #                                        area_spread_threshold=area_spread_threshold, environment=environment)
            #         model.index_timestamp = np.searchsorted(model.df.timestamps, model.timestamp) - 1
            #         model.timestamp = model.df[model.index_timestamp]
            for trigger in model.machine.get_triggers(model.state):
                if not trigger.startswith('to_'):
                    if model.trigger(trigger):
                        break
        print("Done!")

        # o.close()

        st.write(f'Time elapsed {t.tocvalue()} sec')

        # Compute spreads for each execution. Statistics about executions. How did we execute (from posted or trying to cancel)
        if len(model.executions_as_maker) == 0 and len(model.executions_as_taker) == 0:
            st.write(f'No executions for this displacement value!')
            return
        if len(model.executions_as_maker) == 0:
            simulated_executions_maker = pd.DataFrame(
                columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread",
                         "central_band", "was_trying_to_cancel_spot", "was_trying_to_cancel_swap",
                         "source_at_execution_swap", "dest_at_execution_swap", "source_at_execution_spot",
                         "dest_at_execution_spot", "is_balancing", "is_balancing_spot", "side"])
        else:
            simulated_executions_maker = pd.DataFrame(model.executions_as_maker)
        if len(model.executions_as_taker) == 0:
            model.executions_as_taker = pd.DataFrame(
                columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread",
                         "central_band", "was_trying_to_cancel_spot", "was_trying_to_cancel_swap",
                         "source_at_execution_swap", "dest_at_execution_swap", "source_at_execution_spot",
                         "dest_at_execution_spot", "is_balancing", "is_balancing_spot", "side"])
        else:
            simulated_executions_taker = pd.DataFrame(model.executions_as_taker)
        sim_ex_maker_entry_mask = simulated_executions_maker.side == 'entry'
        sim_ex_maker_exit_mask = simulated_executions_maker.side == 'exit'
        st.write(simulated_executions_taker)
        sim_ex_taker_entry_mask = simulated_executions_taker.side == 'entry'

        sim_ex_taker_exit_mask = simulated_executions_taker.side == 'exit'

        if len(simulated_executions_maker[sim_ex_maker_exit_mask].central_band) > 0 and len(
                simulated_executions_maker[sim_ex_maker_exit_mask].executed_spread) > 0:
            assert (simulated_executions_maker[sim_ex_maker_exit_mask].executed_spread < simulated_executions_maker[
                sim_ex_maker_exit_mask].central_band).all(), "Something is wrong in the computation"
        if len(simulated_executions_maker[sim_ex_maker_entry_mask].executed_spread) > 0 and len(
                simulated_executions_maker[sim_ex_maker_entry_mask].central_band) > 0:
            assert (simulated_executions_maker[sim_ex_maker_entry_mask].executed_spread > simulated_executions_maker[
                sim_ex_maker_entry_mask].central_band).all(), "Something is wrong in the computation"
        simulated_executions_maker['execution_quality'] = np.abs(
            simulated_executions_maker.central_band - simulated_executions_maker.executed_spread)
        simulated_executions_maker[
            'is_spot_first'] = simulated_executions_maker.timestamp_spot_executed < simulated_executions_maker.timestamp_swap_executed
        simulated_executions_maker['Time'] = pd.to_datetime(simulated_executions_maker['timems'], unit='ms')
        simulated_executions_taker['execution_quality'] = np.nan
        simulated_executions_taker['execution_quality'][sim_ex_taker_exit_mask] = simulated_executions_taker[
                                                                                      sim_ex_taker_exit_mask].central_band - \
                                                                                  simulated_executions_taker[
                                                                                      sim_ex_taker_exit_mask].executed_spread
        simulated_executions_taker['execution_quality'][sim_ex_taker_entry_mask] = simulated_executions_taker[
                                                                                       sim_ex_taker_entry_mask].executed_spread - \
                                                                                   simulated_executions_taker[
                                                                                       sim_ex_taker_entry_mask].central_band
        simulated_executions_taker[
            'is_spot_first'] = simulated_executions_taker.timestamp_spot_executed < simulated_executions_taker.timestamp_swap_executed
        simulated_executions_taker['Time'] = pd.to_datetime(simulated_executions_taker['timems'], unit='ms')

        maker_entries = simulated_executions_maker[sim_ex_maker_entry_mask]
        maker_exits = simulated_executions_maker[sim_ex_maker_exit_mask]
        taker_entries = simulated_executions_taker[sim_ex_taker_entry_mask]
        taker_exits = simulated_executions_taker[sim_ex_taker_exit_mask]
        percentage_executions_maker_exit = len(maker_exits) / (len(maker_exits) + len(taker_exits)) if len(
            maker_exits) + len(taker_exits) > 0 else 0
        percentage_executions_maker_entry = len(maker_entries) / (len(maker_entries) + len(taker_entries)) if len(
            maker_entries) + len(taker_entries) > 0 else 0
        percentage_executions_maker = (len(maker_entries) + len(maker_exits)) / (
                    len(maker_entries) + len(maker_exits) + len(taker_entries) + len(taker_exits))

        col_res_1, col_res_2, col_res_3 = st.columns(3)
        col_res_1.markdown(
            f"<b>Percentage Maker Maker exit: {int(percentage_executions_maker_exit * 10000) / 100}%</b>",
            unsafe_allow_html=True)
        col_res_2.markdown(
            f"<b>Percentage Maker Maker entry: {int(percentage_executions_maker_entry * 10000) / 100}%</b>",
            unsafe_allow_html=True)
        col_res_3.markdown(f"<b>Percentage Maker Maker: {int(percentage_executions_maker * 10000) / 100}%</b>",
                           unsafe_allow_html=True)
        col_res_4, col_res_5, col_res_6, col_res_7 = st.columns(4)
        col_res_4.markdown(f"<b>Number of Maker Maker executions entry: {len(maker_entries)}</b>",
                           unsafe_allow_html=True)
        col_res_5.markdown(f"<b>Number of Maker Maker executions exit: {len(maker_exits)}</b>", unsafe_allow_html=True)
        col_res_6.markdown(f"<b>Number of Taker Maker executions entry: {len(taker_entries)}</b>",
                           unsafe_allow_html=True)
        col_res_7.markdown(f"<b>Number of Taker Maker executions exit: {len(taker_exits)}</b>", unsafe_allow_html=True)
        col_res_8, col_res_9, col_res_10, col_res_11 = st.columns(4)
        col_res_8.markdown(
            f"<b>Avg executed spread entry: {round((taker_entries.executed_spread.sum() + maker_entries.executed_spread.sum()) / (len(taker_entries) + len(maker_entries)), 2)}</b>",
            unsafe_allow_html=True)
        col_res_9.markdown(
            f"<b>Avg executed spread exit: {round((taker_exits.executed_spread.sum() + maker_exits.executed_spread.sum()) / (len(taker_exits) + len(maker_exits)), 2)}</b>",
            unsafe_allow_html=True)
        col_res_10.markdown(
            f"<b>Avg executed balancing spread entry: {round(taker_entries.executed_spread.mean(), 2)}</b>",
            unsafe_allow_html=True)
        col_res_11.markdown(
            f"<b>Avg executed balancing spread exit: {round(taker_exits.executed_spread.mean(), 2)}</b>",
            unsafe_allow_html=True)
        col_res_12, col_res_13, col_res_14, col_res_15 = st.columns(4)
        col_res_12.markdown(
            f"<b>Avg execution quality entry: {round((taker_entries.execution_quality.sum() + maker_entries.execution_quality.sum()) / (len(taker_entries) + len(maker_entries)), 2)}</b>",
            unsafe_allow_html=True)
        col_res_13.markdown(
            f"<b>Avg execution quality exit: {round((taker_exits.execution_quality.sum() + maker_exits.execution_quality.sum()) / (len(taker_exits) + len(maker_exits)), 2)}</b>",
            unsafe_allow_html=True)
        col_res_14.markdown(
            f"<b>Avg execution quality balancing entry: {round(taker_entries.execution_quality.mean(), 2)}</b>",
            unsafe_allow_html=True)
        col_res_15.markdown(
            f"<b>Avg execution quality balancing exit: {round(taker_exits.execution_quality.mean(), 2)}</b>",
            unsafe_allow_html=True)
        col_res_16, col_res_17, col_res_18, col_res_19 = st.columns(4)
        col_res_16.markdown(f"<b>Balancings entry (spot first): {len(taker_entries[taker_entries.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_18.markdown(f"<b>Balancings exit (spot first): {len(taker_entries[~taker_entries.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_17.markdown(f"<b>Balancings entry (swap first): {len(taker_exits[taker_exits.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_19.markdown(f"<b>Balancings exit (swap first): {len(taker_exits[~taker_exits.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_20, col_res_21, col_res_22, col_res_23 = st.columns(4)
        col_res_20.markdown(f"<b>Maker Maker entry (spot first): {len(maker_entries[maker_entries.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_22.markdown(f"<b>Maker Maker exit (spot first): {len(maker_entries[~maker_entries.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_21.markdown(f"<b>Maker Maker entry (swap first): {len(maker_exits[maker_exits.is_spot_first])}</b>",
                            unsafe_allow_html=True)
        col_res_23.markdown(f"<b>Maker Maker exit (swap first): {len(maker_exits[~maker_exits.is_spot_first])}</b>",
                            unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center; color: grey;'>Descriptive statics of Simulated Maker Maker Executions in the given period</h3>",
            unsafe_allow_html=True)
        temp_col_1, temp_col_2, temp_col_3 = st.columns(3)
        temp_col_1.text('')
        temp_col_2.dataframe(simulated_executions_maker.describe())
        temp_col_3.text('')
        st.markdown(
            "<h3 style='text-align: center; color: grey;'>Descriptive statics of Simulated Taker Maker Executions in the given period</h3>",
            unsafe_allow_html=True)
        temp_col_4, temp_col_5, temp_col_6 = st.columns(3)
        temp_col_4.text('')
        temp_col_5.dataframe(simulated_executions_taker.describe())
        temp_col_6.text('')

        if len(simulated_executions_maker) > 0:
            xp_maker = hip.Experiment.from_dataframe(simulated_executions_maker[
                                                         ['was_trying_to_cancel_spot', 'was_trying_to_cancel_swap',
                                                          'is_spot_first', 'execution_quality', 'side']])
            st.markdown("<h3 style='text-align: center; color: grey;'>Maker Maker executions</h3>",
                        unsafe_allow_html=True)
            xp_maker.to_streamlit().display()
        if len(simulated_executions_taker) > 0:
            xp_taker = hip.Experiment.from_dataframe(simulated_executions_taker[
                                                         ['was_trying_to_cancel_spot', 'was_trying_to_cancel_swap',
                                                          'is_balancing_spot', 'is_spot_first', 'execution_quality',
                                                          'side']])
            st.markdown("<h3 style='text-align: center; color: grey;'>Taker Maker executions</h3>",
                        unsafe_allow_html=True)
            xp_taker.to_streamlit().display()

        # duration_position has 4 elements per list the column names are as below
        # duration_pos_df = pd.DataFrame(duration_pos, columns=['in_pos_entry', 'in_pos_exit', 'out_pos', 'timems', 'traded_volume'])
        # duration_pos_df['Time'] = pd.to_datetime(duration_pos_df['timems'], unit='ms')

        executions_origin = pd.DataFrame(list_executing, columns=['timems', 'current_state', 'side', 'previous_state'])

        not_posted_df = pd.DataFrame(list_not_posted, columns=['timems', 'state', 'side', 'initial_spread'])

        band_values['timems'] = band_values['Time'].view(np.int64) // 10 ** 6
        band_values = pd.merge_ordered(band_values[['Time', 'Central Band', 'timems']], model.df[
            ['breakeven_band_exit_balancing_spot', 'breakeven_band_exit_balancing_swap',
             'breakeven_band_entry_balancing_spot', 'breakeven_band_entry_balancing_swap']], on='timems')

        simulated_executions_maker = pd.merge_ordered(band_values[['Central Band', 'timems']],
                                                      simulated_executions_maker, on='timems')
        simulated_executions_maker['Central Band'].ffill(inplace=True)
        simulated_executions_maker['time_diff'] = simulated_executions_maker['timems'].diff()

        simulated_executions_taker = pd.merge_ordered(band_values[['Central Band', 'timems']],
                                                      simulated_executions_taker, on='timems')
        simulated_executions_taker['Central Band'].ffill(inplace=True)
        simulated_executions_taker['time_diff'] = simulated_executions_taker['timems'].diff()
        # simulated_executions['exit_diff'] = -simulated_executions.loc[simulated_executions['side'] == 'exit', 'executed_spread'] + simulated_executions['Exit Band']
        # simulated_executions['entry_diff'] = simulated_executions.loc[simulated_executions['side'] == 'entry', 'executed_spread'] - simulated_executions['Entry Band']
        # simulated_executions = simulated_executions[(~simulated_executions['exit_diff'].isna()) | (~simulated_executions['entry_diff'].isna())]
        # simulated_executions.drop_duplicates(subset=['timems'], keep='last', inplace=True)

        if actual_exists == 'yes':
            executions = get_executions(t0=t_start, t1=t_end, strategy=strategy, environment=environment)
            actual_band_values = get_band_values(t0=t_start, t1=t_end, typeb='bogdan_bands', strategy=strategy,
                                                 environment=environment)
            actual_executions = pd.merge_ordered(executions, actual_band_values, on='Time')
            actual_executions['Entry Band'].ffill(inplace=True)
            actual_executions['Exit Band'].ffill(inplace=True)
            actual_executions['exit_diff'] = -actual_executions['exit_executions'] + actual_executions['Exit Band']
            actual_executions['entry_diff'] = actual_executions['entry_executions'] - actual_executions['Entry Band']
            actual_executions = actual_executions[
                (~actual_executions['exit_diff'].isna()) | (~actual_executions['entry_diff'].isna())]
            actual_executions['timems'] = actual_executions['Time'].view(int) // 10 ** 6
            actual_executions['time_diff'] = actual_executions['timems'].diff()

            st.text('Descriptive statics of Actual Executions in the given period')
            st.dataframe(actual_executions.describe())

        df_total = pd.merge_ordered(simulated_executions_maker, executions_origin, on='timems')
        st.download_button("Press to Download", simulated_executions_maker.to_csv(),
                           f"Sim_{strategy}_from_{date_range[0]}_to_{date_range[1]}.csv", "text/csv")
        st.download_button("Press to Download", simulated_executions_taker.to_csv(),
                           f"Sim_{strategy}_from_{date_range[0]}_to_{date_range[1]}.csv", "text/csv")

        # col_text_maker_maker_1, col_text_maker_maker_2, col_text_maker_maker_3 = st.columns(3)
        # col_text_maker_maker_1.subheader(f'Entry Execution Quality (Maker both sides): {round(maker_entries.describe().iloc[1, 8], 2)}')
        # col_text_maker_maker_2.subheader(f'Exit Execution Quality (Maker both sides): {round(maker_exits.describe().iloc[1, 7], 2)}')
        # col_text_maker_maker_3.subheader('')
        # col_text_maker_maker_1.subheader(f"Total Traded Volume in this period: {simulated_executions['traded_volume'].sum()}")
        days_in_period = (t_end - t_start) // (1000 * 60 * 60 * 24)
        # col_text_maker_maker_2.subheader(f"Average Daily Traded Volume in this period:"f" {int(simulated_executions['traded_volume'].sum() // days_in_period)}")
        # col_text_maker_maker_3.subheader('_')
        # avg_entry_spread = np.sum(simulated_executions.loc[simulated_executions.side == 'entry', 'executed_spread'].values * simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume'].values) / \
        #                    np.sum(simulated_executions.loc[simulated_executions.side == 'entry', 'traded_volume'].values)

        # avg_exit_spread = np.sum(simulated_executions.loc[simulated_executions.side == 'exit', 'executed_spread'].values * simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume'].values) / \
        #                   np.sum(simulated_executions.loc[simulated_executions.side == 'exit', 'traded_volume'].values)
        # col_text1.subheader(f"Average Entry Spread: {round(avg_entry_spread, 2)}")
        # col_text2.subheader(f"Average Exit Spread: {round(avg_exit_spread, 2)}")
        # col_text3.subheader(f"Avg Fixed Spread: {round(avg_entry_spread - avg_exit_spread, 2)}")

        if actual_exists == 'yes':
            # Skipping for now
            pass
            # fig = make_subplots(rows=2, cols=1, subplot_titles=("Simulated vs Actual Executions", "Actual Executions"))
            # fig.append_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'entry', 'Time'],
            #                             y=simulated_executions.loc[simulated_executions['side'] == 'entry', 'executed_spread'],
            #                             marker=dict(color='green'),
            #                             mode='markers',
            #                             name='Sim Entry Exec'), row=1, col=1)
            # fig.append_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'exit', 'Time'],
            #                             y=simulated_executions.loc[simulated_executions['side'] == 'exit', 'executed_spread'],
            #                             marker=dict(color='yellow'),
            #                             mode='markers',
            #                             name='Sim Exit Exec'), row=1, col=1)
            #
            # fig.append_trace(go.Scatter(x=actual_executions.loc[~actual_executions['entry_executions'].isna(), 'Time'],
            #                             y=actual_executions.loc[~actual_executions['entry_executions'].isna(), 'entry_executions'],
            #                             marker=dict(color='white'),
            #                             mode='markers',
            #                             name='Actual Entry Exec'), row=1, col=1)
            # fig.append_trace(go.Scatter(x=actual_executions.loc[~actual_executions['exit_executions'].isna(), 'Time'],
            #                             y=actual_executions.loc[~actual_executions['exit_executions'].isna(), 'exit_executions'],
            #                             opacity=0.5,
            #                             marker=dict(color='red'),
            #                             mode='markers',
            #                             name='Actual Exit Exec'), row=1, col=1)
            #
            # fig.append_trace(go.Scatter(
            #     x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Time'],
            #     y=band_values.loc[
            #         (~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Entry Band'],
            #     line=dict(color="green"),
            #     line_shape='vh',
            #     mode='lines',
            #     name='Entry Band'), row=1, col=1)
            # fig.append_trace(go.Scatter(
            #     x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Time'],
            #     y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Exit Band'],
            #     line=dict(color="red"),
            #     line_shape='vh',
            #     mode='lines',
            #     name='Exit Band'), row=1, col=1)
            #
            # fig.append_trace(go.Scatter(x=actual_executions.loc[~actual_executions['entry_executions'].isna(), 'Time'],
            #                             y=actual_executions.loc[~actual_executions['entry_executions'].isna(), 'entry_executions'],
            #                             marker=dict(color='green'),
            #                             mode='markers',
            #                             name='Actual Entry Exec'), row=2, col=1)
            # fig.append_trace(go.Scatter(x=actual_executions.loc[~actual_executions['exit_executions'].isna(), 'Time'],
            #                             y=actual_executions.loc[~actual_executions['exit_executions'].isna(), 'exit_executions'],
            #                             marker=dict(color='yellow'),
            #                             mode='markers',
            #                             name='Actual Exit Exec'), row=2, col=1)
            #
            # fig.append_trace(go.Scatter(
            #     x=actual_band_values.loc[(~actual_band_values['Entry Band'].isna()) & (actual_band_values['Entry Band'] <= 200), 'Time'],
            #     y=band_values.loc[(~actual_band_values['Entry Band'].isna()) & (actual_band_values['Entry Band'] <= 200), 'Entry Band'],
            #     line=dict(color="green"),
            #     line_shape='hv',
            #     mode='lines',
            #     name='Entry Band'), row=2, col=1)
            # fig.append_trace(go.Scatter(
            #     x=actual_band_values.loc[(~actual_band_values['Exit Band'].isna()) & (actual_band_values['Exit Band'] >= -200), 'Time'],
            #     y=actual_band_values.loc[(~actual_band_values['Exit Band'].isna()) & (actual_band_values['Exit Band'] >= -200), 'Exit Band'],
            #     line=dict(color="red"),
            #     line_shape='hv',
            #     mode='lines',
            #     name='Exit Band'), row=2, col=1)
            #
            # fig.update_layout(title='Simulated vs Actual Executions',
            #                   template='plotly_dark',
            #                   autosize=False,
            #                   width=2000,
            #                   height=1000,
            #                   xaxis_title='Date',
            #                   yaxis_title='Spread in USD')
            #
            # st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=maker_entries['Time'],
                                     y=maker_entries['executed_spread'],
                                     marker=dict(color='green'),
                                     mode='markers',
                                     marker_size=10,
                                     name='Sim Entry (Maker both)'))
            fig.add_trace(go.Scatter(x=maker_exits['Time'],
                                     y=maker_exits['executed_spread'],
                                     marker=dict(color='red'),
                                     mode='markers',
                                     marker_size=10,
                                     name='Sim Exit (Maker both)'))
            fig.add_trace(go.Scatter(x=taker_entries['Time'],
                                     y=taker_entries['executed_spread'],
                                     marker=dict(color='green'),
                                     mode='markers',
                                     marker_symbol='x',
                                     marker_size=10,
                                     name='Sim Entry (Maker Taker)'))
            fig.add_trace(go.Scatter(x=taker_exits['Time'],
                                     y=taker_exits['executed_spread'],
                                     marker=dict(color='red'),
                                     mode='markers',
                                     marker_symbol='x',
                                     marker_size=10,
                                     name='Sim Exit (Maker Taker)'))

            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Time'],
                y=band_values.loc[
                    (~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Central Band'],
                line=dict(color="blue"),
                line_shape='hv',
                mode='lines',
                name='Central Band'))

            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Time'],
                y=band_values.loc[(~band_values['Central Band'].isna()) & (
                            band_values['Central Band'] <= 200), 'breakeven_band_entry_balancing_spot'],
                line=dict(color="rgb(8,116,54)"),
                line_shape='hv',

                mode='lines',
                name='Breakeven Band Entry Balance Spot'))
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Time'],
                y=band_values.loc[(~band_values['Central Band'].isna()) & (
                            band_values['Central Band'] <= 200), 'breakeven_band_exit_balancing_swap'],
                line=dict(color="rgb(255,153,153)"),
                line_shape='hv',
                mode='lines',
                name='Breakeven Band Exit Balance Swap'))
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Time'],
                y=band_values.loc[(~band_values['Central Band'].isna()) & (
                            band_values['Central Band'] <= 200), 'breakeven_band_exit_balancing_spot'],
                line=dict(color="rgb(133,50,41)"),
                line_shape='hv',
                mode='lines',
                name='Breakeven Band Exit Balance Spot'))
            fig.add_trace(go.Scatter(

                x=band_values.loc[(~band_values['Central Band'].isna()) & (band_values['Central Band'] <= 200), 'Time'],
                y=band_values.loc[(~band_values['Central Band'].isna()) & (
                            band_values['Central Band'] <= 200), 'breakeven_band_entry_balancing_swap'],
                line=dict(color="rgb(0,241,0)"),
                line_shape='hv',
                mode='lines',
                name='Breakeven Band Entry Balance Swap'))
            # fig.add_trace(go.Scatter(
            #     x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Time'],
            #     y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Exit Band'],
            #     line=dict(color="red"),
            #     line_shape='hv',
            #     mode='lines',
            #     name='Exit Band'))
            fig.update_layout(title='Simulated vs Actual Executions',
                              autosize=False,
                              height=1000,
                              xaxis_title='Date',
                              yaxis_title='Spread in USD')
            st.plotly_chart(fig, use_container_width=True)

        # Have the following 2
        # skipping for now
        if False:
            fig1 = px.histogram(
                executions_origin[['side', 'previous_state', 'timems']].sort_values(by=['previous_state']),
                x='side', y="timems", color='previous_state', barmode='group',
                barnorm='percent', text_auto=True,
                title="Bar-plot of Executions Percentage", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(
                executions_origin[['side', 'previous_state', 'timems']].sort_values(by=['previous_state']),
                x='side', y="timems", color='previous_state', barmode='group', text_auto=True,
                histfunc='count', title="Bar-plot of Executions Count", height=400)
            st.plotly_chart(fig2, use_container_width=True)

            fig1 = px.histogram(
                executions_origin[['side', 'previous_state', 'timems']].sort_values(by=['previous_state']),
                x='side', y="timems", color='previous_state', barmode='group',
                barnorm='percent', text_auto=True,
                title="Bar-plot of Executions Percentage", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.histogram(
                executions_origin[['side', 'previous_state', 'timems']].sort_values(by=['previous_state']),
                x='side', y="timems", color='previous_state', barmode='group', text_auto=True,
                histfunc='count', title="Bar-plot of Executions Count", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # # Missing the following 2
        # fig5 = px.histogram(simulated_executions, x='spread_diff', text_auto=True,
        #                     histnorm='percent', title="Final Spread - Initial Spread Density Histogram", height=400)
        # fig5.update_layout(bargap=0.01)
        # st.plotly_chart(fig5, use_container_width=True)
        #
        # fig6 = px.histogram(simulated_executions, x='spread_diff', text_auto=True,
        #                     histnorm='percent', title="Final Spread - Initial Spread Cumulative Density Histogram",
        #                     cumulative=True, height=400)
        # fig6.update_layout(bargap=0.01)
        # st.plotly_chart(fig6, use_container_width=True)
        #
        # # Have the following 2
        # fig7 = px.histogram(simulated_executions, x='exit_diff', text_auto=True,
        #                     histnorm='percent', title="Exit Executions Diff from Exit Band",
        #                     cumulative=True, height=400)
        # fig7.update_layout(bargap=0.01)
        # st.plotly_chart(fig7, use_container_width=True)
        #
        # fig8 = px.histogram(simulated_executions, x='entry_diff', text_auto=True,
        #                     histnorm='percent', title="Entry Executions Diff form Entry Band",
        #                     cumulative=True, height=400)
        # fig8.update_layout(bargap=0.01)
        # st.plotly_chart(fig8, use_container_width=True)
        #
        # fig9 = px.histogram(not_posted_df[['side', 'state', 'timems']].sort_values(by=['state'], ascending=False),
        #                     x='side', y="timems", color='state', barmode='group',
        #                     barnorm='percent', text_auto=True,
        #                     title="Bar-plot of Posted and Not Posted (Percentage)", height=400)
        # st.plotly_chart(fig9, use_container_width=True)
        #
        # fig10 = px.histogram(not_posted_df[['side', 'state', 'timems']].sort_values(by=['state'], ascending=False),
        #                      x='side', y="timems", color='state', barmode='group', text_auto=True, histfunc='count',
        #                      title="Bar-plot of Posted and Not Posted (Count)", height=400)
        # st.plotly_chart(fig10, use_container_width=True)
        #
        # # Missing the following 3
        # #fig11 = go.Figure()
        # #fig11.add_trace(go.Scatter(x=duration_pos_df['Time'], y=duration_pos_df['out_pos'], opacity=0.5, connectgaps=False, line=dict(color="blue"), mode='lines', name='Out 0f Position'))
        # #fig11.add_trace(go.Scatter(x=duration_pos_df['Time'], y=duration_pos_df['in_pos_entry'], line=dict(color="green"), mode='lines', connectgaps=False, name='In Position Enter'))
        # #fig11.add_trace(go.Scatter(x=duration_pos_df['Time'], y=duration_pos_df['in_pos_exit'], line=dict(color="gold"), mode='lines', connectgaps=False, name='In Position Exit'))
        # #fig11.update_layout(title='Traded Volume in Time', autosize=False, height=400, xaxis_title='Date', yaxis_title='Volume in USD')
        # #st.plotly_chart(fig11, use_container_width=True)
        #
        #
        # fig12 = px.histogram(simulated_executions[simulated_executions['time_diff']<=2000], x='time_diff', text_auto=True, histnorm='percent', cumulative=True,
        #                      marginal="rug",
        #                      title="Timedelta between two sequential executions", height=400)
        # st.plotly_chart(fig12, use_container_width=True)
        #
        # if st.button("Clear All"):
        #     # Clears all singleton caches:
        #     st.experimental_singleton.clear()
