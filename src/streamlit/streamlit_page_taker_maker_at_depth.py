import numpy as np
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from pytictoc import TicToc

from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.queries.queries import get_strategy_families, get_symbol, get_strategy_influx, get_exhange_names, \
    get_entry_exit_bands, get_executions, get_band_values, get_data_for_trader
import datetime
import random
import plotly.graph_objects as go
import streamlit as st
from src.simulations.taker_maker.TakerMakerDeeperLevelSpread import TakerMakerDeeperLevelSpread

load_dotenv(find_dotenv())


class ConstantDepthPosting:
    pass


def run_simulation():
    t = TicToc()
    if st.button("Clear All"):
        # Clears all singleton caches:
        st.experimental_singleton.clear()
    # if os.path.exists('logs_trader.txt'):
    #     os.remove("logs_trader.txt")
    st.sidebar.write("Taker maker trading with at-depth posting")
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
    st.write('The ending time in milliseconds', t_end)
    st.text('Default time-range is 1 day')
    st.write('Select the strategies family and strategy you want to review')
    st.write('If you want to review a new combination of exchanges select "Other"')

    col1, col2, col3 = st.columns(3)
    family = col1.selectbox('Strategy family', ('deribit_xbtusd', 'deribit_eth', 'Other'))
    environment = col2.selectbox('Environment from where data are downloaded', ('production', 'staging', 'server'))
    if family == 'Other':
        strategy = get_strategy_influx(environment=environment)
    elif family == 'deribit_xbtusd':
        strategy = col3.selectbox('Select the strategy',
                                  get_strategy_families(t0=t_start, environment='production')[family])
    else:
        strategy = col3.selectbox('Select the strategy',
                                  get_strategy_families(t0=t_start, environment='production')[family])

    if family == 'Other':
        exchange_spot = get_exhange_names(t0=t_start, t1=t_end, environment=environment)
        exchange_swap = get_exhange_names(t0=t_start, t1=t_end, environment=environment)
        spot_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot, environment=environment)[0]
        swap_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap, environment=environment)[-1]
    else:
        exchange_spot = 'Deribit'
        exchange_swap = 'BitMEX'
        spot_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot, environment=environment)[0]
        swap_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap, environment=environment)[3]

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
    col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
    max_trade_volume = col_ts1.number_input('Max Trade Volume', min_value=0, value=3000, max_value=100000)
    max_position = col_ts2.number_input('Max Position', min_value=0, value=100000, max_value=1000000)
    entry_delta_spread = col_ts3.number_input('Entry Delta Spread', min_value=0.0, value=3.7, max_value=20.0)
    exit_delta_spread = col_ts4.number_input('Exit Delta Spread', min_value=0.0, value=3.7, max_value=20.0)
    misc_col_1 = st.columns(1)[0]
    depth_posting = misc_col_1.number_input('Targeted depth (in bp)', min_value=0.0, value=1.0, max_value=10.0)

    funding_system = "No"
    minimum_distance = 0
    minimum_value = 0
    disable_when_below = 0

    area_spread_threshold = 0
    trailing_value = 0
    depth_posting_predictor = ConstantDepthPosting(depth_posting)
    swap_market_tick_size = 0.5

    file_id = random.randint(10 ** 6, 10 ** 7)

    # convert milliseconds to datetime
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)

    # send message for simulation initialization
    now = datetime.datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
    data = {
        "message": f"Simulation (TM) of {strategy} from {date_start} to {date_end} Started at {dt_string_start} UTC",
    }

    st.subheader('When ready with parameter input click the button')
    check = st.checkbox('Click Here')
    st.write('State of the checkbox: ', check)

    if check:

        band_values = get_entry_exit_bands(t_start=t_start, t_end=t_end, strategy=strategy,
                                           entry_delta_spread=entry_delta_spread, exit_delta_spread=exit_delta_spread,
                                           btype='central_band', environment=environment)
        band_values.rename(columns={'Band': 'Central Band'}, inplace=True)

        df, _ = get_data_for_trader(t_start, t_end, exchange_spot, spot_instrument, exchange_swap, swap_instrument,
                                    swap_fee=maker_fee_swap, spot_fee=taker_fee_spot, strategy=strategy,
                                    area_spread_threshold=area_spread_threshold, environment=environment,
                                    band_type="bogdan_bands", window_size=0, exit_delta_spread=exit_delta_spread,
                                    entry_delta_spread=entry_delta_spread, band_funding_system="No",
                                    generate_percentage_bands=False,
                                    lookback=None, recomputation_time=None, target_percentage_exit=None,
                                    target_percentage_entry=None, entry_opportunity_source=None,
                                    exit_opportunity_source=None,
                                    minimum_target=None, use_aggregated_opportunity_points=None, ending=None,
                                    force_band_creation=False, move_bogdan_band="No")

        model = TakerMakerDeeperLevelSpread(df=df, spot_fee=taker_fee_spot, swap_fee=maker_fee_swap,
                                            area_spread_threshold=area_spread_threshold, latency_spot=latency_spot,
                                            latency_swap=latency_swap,
                                            latency_try_post=latency_try_post_spot,
                                            latency_cancel=latency_cancel_swap,
                                            latency_spot_balance=latency_balance_spot,
                                            max_position=max_position, max_trade_volume=max_trade_volume,
                                            environment=environment, exchange_swap=exchange_swap,
                                            swap_instrument=swap_instrument,
                                            spot_instrument=spot_instrument, funding_system="No",
                                            minimum_distance=minimum_distance, minimum_value=minimum_value,
                                            trailing_value=trailing_value, disable_when_below=disable_when_below,
                                            depth_posting_predictor=depth_posting_predictor,
                                            swap_market_tick_size=swap_market_tick_size, verbose=True)

        print("Starting simulation")
        while model.timestamp < t_end - 1000 * 60 * 5:
            for trigger in model.machine.get_triggers(model.state):
                if not trigger.startswith('to_'):
                    if model.trigger(trigger):
                        break
        print("Done!")

        if len(model.executions) == 0:
            simulated_executions = pd.DataFrame(
                columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread",
                         "targeted_spread", "order_depth", "volume_executed", "entry_band", "exit_band",
                         "was_trying_to_cancel_swap", "source_at_execution_swap", "dest_at_execution_swap", "side"])
        else:
            simulated_executions = pd.DataFrame(model.executions)

        if len(model.cancelled_orders_swap) == 0:
            simulated_cancellations = pd.DataFrame(
                columns=["Time posted", "Time cancelled", "timestamp_posted", "timestamp_cancelled",
                         "cancelled_to_post_deeper", "targeted_spread", "price", "max_targeted_depth", "side"])
        else:
            temp_cancellations = []
            for order in model.cancelled_orders_swap:
                temp_cancellations.append(
                    [order.timestamp_posted, order.timestamp_cancelled, order.cancelled_to_post_deeper,
                     order.targeted_spread, order.price, order.max_targeted_depth, order.side])
            simulated_cancellations = pd.DataFrame(temp_cancellations,
                                                   columns=["timestamp_posted", "timestamp_cancelled",
                                                            "cancelled_to_post_deeper", "targeted_spread",
                                                            "price", "max_targeted_depth", "side"])
            simulated_cancellations['Time posted'] = pd.to_datetime(simulated_cancellations['timestamp_posted'],
                                                                    unit='ms')
            simulated_cancellations['Time cancelled'] = pd.to_datetime(simulated_cancellations['timestamp_cancelled'],
                                                                       unit='ms')
            simulated_cancellations_entry = simulated_cancellations[simulated_cancellations.side == 'entry']
            simulated_cancellations_exit = simulated_cancellations[simulated_cancellations.side == 'exit']

        band_values['timems'] = band_values['Time'].view(np.int64) // 10 ** 6

        band_values = band_values[~band_values["Central Band"].isnull()]
        simulated_executions = pd.merge_ordered(band_values[['Central Band', 'Entry Band', 'Exit Band', 'timems']],
                                                simulated_executions, on='timems')
        simulated_executions['Central Band'].ffill(inplace=True)
        simulated_executions['Entry Band'].ffill(inplace=True)
        simulated_executions['Exit Band'].ffill(inplace=True)
        simulated_executions['time_diff'] = simulated_executions['timems'].diff()
        # days in period
        days_in_period = (t_end - t_start) // (1000 * 60 * 60 * 24)

        # simulation descriptive results

        # send a message for simulation end
        now = datetime.datetime.now()
        dt_string_end = now.strftime("%d-%m-%Y %H:%M:%S")
        st.write(f'Time elapsed {t.tocvalue()} sec')

        # Compute spreads for each execution. Statistics about executions. How did we execute (from posted or trying to cancel)
        if len(simulated_executions) == 0:
            st.write(f'No executions!')
            return
        sim_ex_entry_mask = simulated_executions.side == 'entry'
        sim_ex_exit_mask = simulated_executions.side == 'exit'
        st.write(simulated_executions)

        simulated_executions['execution_quality'] = np.nan
        simulated_executions['execution_quality'][sim_ex_entry_mask] = np.abs(
            simulated_executions[sim_ex_entry_mask]['Entry Band'] - simulated_executions[
                sim_ex_entry_mask].executed_spread)
        simulated_executions['execution_quality'][sim_ex_exit_mask] = np.abs(
            simulated_executions[sim_ex_exit_mask]['Exit Band'] - simulated_executions[
                sim_ex_exit_mask].executed_spread)
        simulated_executions['Time'] = pd.to_datetime(simulated_executions['timems'], unit='ms')
        entries = simulated_executions[sim_ex_entry_mask]
        exits = simulated_executions[sim_ex_exit_mask]
        total_volume_traded = simulated_executions.volume_executed.sum()
        total_volume_traded_in_token = (entries.volume_executed / entries.price).sum() + (
                    exits.volume_executed / exits.price).sum()
        n_trades_executed_both_sides = min(len(entries), len(exits))

        weighted_fixed_spread = round(
            (entries.executed_spread[:n_trades_executed_both_sides].sum() / (n_trades_executed_both_sides + 0.001)) -
            (exits.executed_spread[:n_trades_executed_both_sides].sum() / (n_trades_executed_both_sides + 0.001)), 2)

        # Number of cancelled per side, number of executed per side
        col_res_1, col_res_2, col_res_3, col_res_4 = st.columns(4)
        col_res_1.markdown(f"<b>Number of executions entry: {len(entries)}</b>", unsafe_allow_html=True)
        col_res_2.markdown(f"<b>Number of executions exit: {len(exits)}</b>", unsafe_allow_html=True)
        col_res_3.markdown(
            f"<b>Avg executed spread entry: {round(entries.executed_spread.sum() / len(entries), 2)}</b>",
            unsafe_allow_html=True)
        col_res_4.markdown(f"<b>Avg executed spread exit: {round(exits.executed_spread.sum() / len(exits), 2)}</b>",
                           unsafe_allow_html=True)
        col_res_5, col_res_6, col_res_7, col_res_8 = st.columns(4)
        col_res_5.markdown(
            f"<b>Avg execution quality entry: {round(entries.execution_quality.sum() / (len(entries)), 2)}</b>",
            unsafe_allow_html=True)
        col_res_6.markdown(f"<b>Avg execution quality exit: {round(exits.execution_quality.sum() / len(exits), 2)}</b>",
                           unsafe_allow_html=True)
        col_res_7.markdown(f"<b>Total volume traded: {total_volume_traded}</b>", unsafe_allow_html=True)
        col_res_8.markdown(f"<b>Total volume traded in coin: {total_volume_traded_in_token}</b>",
                           unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center; color: grey;'>Descriptive statics of Simulated Executions in the given period</h3>",
            unsafe_allow_html=True)
        temp_col_1, temp_col_2, temp_col_3 = st.columns(3)
        temp_col_1.text('')
        temp_col_2.dataframe(simulated_executions.describe())
        temp_col_3.text('')

        # duration_position has 4 elements per list the column names are as below
        # duration_pos_df = pd.DataFrame(duration_pos, columns=['in_pos_entry', 'in_pos_exit', 'out_pos', 'timems', 'traded_volume'])
        # duration_pos_df['Time'] = pd.to_datetime(duration_pos_df['timems'], unit='ms')

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

        st.download_button("Press to Download", simulated_executions.to_csv(),
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=entries['Time'],
                                 y=entries['executed_spread'],
                                 marker=dict(color='green'),
                                 mode='markers',
                                 marker_size=10,
                                 name='Sim Entry'))
        fig.add_trace(go.Scatter(x=exits['Time'],
                                 y=exits['executed_spread'],
                                 marker=dict(color='red'),
                                 mode='markers',
                                 marker_size=10,
                                 name='Sim Exit'))

        fig.add_trace(go.Scatter(
            x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Time'],
            y=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Entry Band'],
            line=dict(color="blue"),
            line_shape='hv',
            mode='lines',
            name='Entry Band'))

        fig.add_trace(go.Scatter(
            x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] <= 200), 'Time'],
            y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] <= 200), 'Exit Band'],
            line=dict(color="rgb(8,116,54)"),
            line_shape='hv',
            mode='lines',
            name='Exit Band'))

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
        if False:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df['Time'],
                y=df["price_spot_entry"],
                line=dict(color="green"),
                line_shape='hv',
                mode='lines',
                name='Entry price spot'))
            fig1.add_trace(go.Scatter(
                x=df['Time'],
                y=df["price_spot_exit"],
                line=dict(color="green"),
                line_shape='hv',
                mode='lines',
                name='Exit price spot'))
            fig1.add_trace(go.Scatter(
                x=df['Time'],
                y=df["price_swap_exit"],
                line=dict(color="red"),
                line_shape='hv',
                mode='lines',
                name='Entry price swap'))
            fig1.add_trace(go.Scatter(
                x=df['Time'],
                y=df["price_swap_entry"],
                line=dict(color="red"),
                line_shape='hv',
                mode='lines',
                name='Exit price spot'))
            fig1.add_trace(go.Scatter(
                x=simulated_cancellations_entry["Time posted"],
                y=simulated_cancellations_entry["price"],
                marker=dict(color="green"),
                marker_symbol='circle',
                marker_size=8,
                mode='markers',
                name="Entry posted limit order"
            ))
            fig1.add_trace(go.Scatter(
                x=simulated_cancellations_entry["Time cancelled"],
                y=simulated_cancellations_entry["price"],
                marker=dict(color="green"),
                mode='markers',
                marker_size=8,
                marker_symbol='x',
                name="Entry cancelled limit order"
            ))
            fig1.add_trace(go.Scatter(
                x=simulated_cancellations_exit["Time posted"],
                y=simulated_cancellations_exit["price"],
                marker=dict(color="red"),
                mode='markers',
                marker_size=8,
                marker_symbol='circle',
                name="Exit posted limit order"
            ))
            fig1.add_trace(go.Scatter(
                x=simulated_cancellations_exit["Time cancelled"],
                y=simulated_cancellations_exit["price"],
                marker=dict(color="red"),
                mode='markers',
                marker_size=8,
                marker_symbol='x',
                name="Exit cancelled limit order"
            ))
            fig1.update_layout(title='Posting and cancelling',
                               autosize=False,
                               height=1000,
                               xaxis_title='Date',
                               yaxis_title='Price')
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=simulated_cancellations_exit["timestamp_cancelled"] - simulated_cancellations_exit[
                    "timestamp_posted"],
                name="Time between posting and cancelling",
                xbins=go.XBins(size=500)
            ))
            fig2.update_layout(title='Posting and cancelling',
                               autosize=False,
                               height=1000,
                               xaxis_title='Millis',
                               yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)

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


if __name__ == '__main__':
    run_simulation()
