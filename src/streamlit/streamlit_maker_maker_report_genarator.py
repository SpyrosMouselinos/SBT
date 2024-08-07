import datetime
import numpy as np
import streamlit as st
import streamlit_permalink as stp
import pandas as pd
import hiplot as hip
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.queries.queries import Prices, Takers
from src.common.clients.backblaze_client import BackblazeClient

load_dotenv(find_dotenv())


def report_generator_maker_maker():
    date_range = st.date_input("Enter a period where the report is genarated",
                               [datetime.date.today() - datetime.timedelta(days=7), datetime.date.today()])

    t_start_search = int(
        datetime.datetime(year=date_range[0].year, month=date_range[0].month, day=date_range[0].day).timestamp() * 1000)
    t_end_search = int(
        datetime.datetime(year=date_range[1].year, month=date_range[1].month, day=date_range[1].day).timestamp() * 1000)

    st.write('The starting time in milliseconds', t_start_search)
    st.write('The ending time in milliseconds', t_end_search)
    st.text('Default time-range is 7 days')

    backblaze = BackblazeClient()
    backblaze.authorize()
    b2 = backblaze.get_b2_resource()
    file_dataframe = backblaze.get_simulations_by_name_and_date(bucket="equinoxai-trades-db",
                                                                b2=b2,
                                                                prefix="",
                                                                time_from=datetime.datetime.strptime(f"{date_range[0]}",
                                                                                                     '%Y-%m-%d'),
                                                                time_to=datetime.datetime.strptime(f"{date_range[1]}",
                                                                                                   '%Y-%m-%d')
                                                                )

    file_dataframe.rename(columns={'filename': 'file_name', 'time': 'timestamp'}, inplace=True)

    st.title('Maker Maker Simulations Report Genarator')
    file_dataframe = pd.DataFrame([x[1] for x in file_dataframe.iterrows() if "MMA" in x[1].file_name])

    check = st.checkbox('Click Here to view the entire data-base')
    st.write('State of the checkbox: ', check)

    if check:
        st.dataframe(file_dataframe)

    timestamp_MM = file_dataframe.loc[
        file_dataframe['file_name'].str.contains('Parameters_MMA', case=False).values, 'timestamp']

    time_list = []
    for idx in timestamp_MM.index:
        time_list.append(timestamp_MM[idx])
    time_MM = pd.Series(time_list)

    # InputData_MM=file_dataframe.loc[
    #     file_dataframe['file_name'].str.contains('InputData_MM', case=False).values, 'file_name']
    # st.dataframe(InputData_MM)
    Parameters_MM = file_dataframe.loc[
        file_dataframe['file_name'].str.contains('Parameters_MMA', case=False).values, 'file_name']
    # st.dataframe(Parameters_MM)
    # st.dataframe(Results_MM)

    result_MM_as_maker = file_dataframe.loc[
        file_dataframe['file_name'].str.contains('_as_maker', case=False, regex=True).values, 'file_name']

    result_MM_as_taker = file_dataframe.loc[
        file_dataframe['file_name'].str.contains('_as_taker', case=False, regex=True).values, 'file_name']
    # st.dataframe(Sim_MM)
    # st.dataframe(Position_duration_MM)
    band_values = file_dataframe.loc[
        file_dataframe['file_name'].str.contains('_band_values', case=False, regex=True).values, 'file_name']

    file_df = pd.concat([time_MM.reset_index(drop=True), Parameters_MM.reset_index(drop=True),
                         result_MM_as_maker.reset_index(drop=True), result_MM_as_taker.reset_index(drop=True),
                         band_values.reset_index(drop=True)], axis=1, ignore_index=True)
    file_df.rename(columns={0: 'time', 1: 'parameters', 2: 'results_as_maker', 3: 'results_as_taker', 4: 'band_values'},
                   inplace=True)
    # st.dataframe(file_df)
    # file_df.dropna(subset=['executions'], inplace=True)

    parameters_id = file_df['parameters'].str.split('_').apply(lambda x: int(x[-1]))

    simulated_executions_maker_ids = file_df['results_as_maker'].str.split('_').apply(lambda x: int(x[-3]))
    simulated_executions_maker_slpg_spot = file_df['results_as_maker'].str.split('_').apply(lambda x: float(x[-10]))
    simulated_executions_maker_slpg_swap = file_df['results_as_maker'].str.split('_').apply(lambda x: float(x[-7]))
    simulated_executions_maker_displacements = file_df['results_as_maker'].str.split('_').apply(lambda x: float(x[-5]))

    simulated_executions_taker_ids = file_df['results_as_taker'].str.split('_').apply(lambda x: float(x[-3]))

    band_values = file_df['band_values'].str.split('_').apply(lambda x: int(x[-3]) if type(x) == str else x)

    # position_id = file_df['position_duration'].str.split('_').apply(lambda x: 0 if '-' in x[-1] or len(x[-1]) > 7 or 'BTC' in x[-1] or 'ETH' in x[-1] else int(x[-1]))

    if parameters_id.all() == simulated_executions_maker_ids.all() and simulated_executions_maker_ids.all() == simulated_executions_taker_ids.all():  # and executions_id.all() == position_id.all():
        st.write("dataframe is correct, all id's match")
        file_df['file_id'] = parameters_id
        file_df['slpg_spot'] = simulated_executions_maker_slpg_spot
        file_df['slpg_swap'] = simulated_executions_maker_slpg_swap
        file_df['displacement'] = simulated_executions_maker_displacements

    else:
        st.write('dataframe has a mistake...')

    st.header('Data-base containing simulation results')
    # Enter the search criteria
    # cr_1, cr_2 = stp.columns(2)
    search_criterion1 = stp.text_input('Search Criterion 1', url_key='criterion1')
    search_criterion2 = stp.text_input('Search Criterion 2', url_key='criterion2')

    st.write(search_criterion1)

    search_result = file_df.loc[file_df['parameters'].str.contains(search_criterion1, case=False).values, :]
    search_result1 = search_result.loc[search_result['parameters'].str.contains(search_criterion2, case=False).values,
                     :]
    displayed_result = search_result1[
        ['time', 'parameters', 'slpg_spot', 'slpg_swap', 'displacement', 'file_id']].sort_values(by='time',
                                                                                                 ascending=False,
                                                                                                 ignore_index=True)
    displayed_result.reset_index(drop=True, inplace=True)

    st.dataframe(displayed_result.drop(displayed_result[displayed_result['file_id'] == 0].index))

    file_id_download = stp.number_input('Select the file ID in order to download the files for the report', value=-1,
                                        url_key='file_id')

    st.subheader('If you have selected a valid ID press the start button to see the report')
    if st.button("Start Report"):
        params_res = backblaze.download_simulation(
            file_df.loc[file_df.file_id == file_id_download, 'parameters'].values[0].replace("simulations/", ""))
        st.title('Parameters used in the Simulation')
        st.subheader('Period of Simulation')
        col_date_1, col_date_2, col_date_3 = st.columns(3)
        col_date_1.write(
            f'Starting date {datetime.datetime.fromtimestamp(params_res["t_start"] / 1000.0, tz=datetime.timezone.utc).strftime("%m-%d-%Y %H:%M:%S")}')
        col_date_2.write(
            f'Ending date {datetime.datetime.fromtimestamp(params_res["t_end"] / 1000.0, tz=datetime.timezone.utc).strftime("%m-%d-%Y %H:%M:%S")}')
        col_date_3.write(f"Strategy {params_res.loc[0, 'strategy']}")
        st.write('Time in milliseconds')
        st.dataframe(params_res[['t_start', 't_end']])
        st.subheader('Input Parameters')
        st.dataframe(params_res[['family', 'environment', 'strategy']])
        if 'instrument_spot' in params_res:
            st.dataframe(params_res[['exchange_spot', 'exchange_swap', 'instrument_spot', 'instrument_swap']])
        else:
            st.dataframe(params_res[['exchange_spot', 'exchange_swap', 'spot_instrument', 'swap_instrument']])
        st.dataframe(
            params_res[['taker_fee_spot', 'maker_fee_spot', "taker_fee_swap", "maker_fee_swap"]].style.format("{:.6}"))
        try:
            st.dataframe(params_res[['area_spread_threshold', 'funding_system']])

        except:
            st.dataframe(params_res[['area_spread_threshold']])

        st.dataframe(params_res[['latency_spot', 'latency_try_post_spot', 'latency_balance_spot', 'latency_cancel_spot',
                                 'latency_swap', 'latency_try_post_swap', 'latency_cancel_swap',
                                 'latency_balance_swap']])
        st.dataframe(params_res[['max_trade_volume', 'max_position']])
        st.dataframe(params_res[['taker_slippage', 'displacement']])

        if 'band' in parameters_id and params_res.loc[0, 'band'] == 'percentage_band':
            st.dataframe(params_res[['lookback', 'recomputation_time']])
            st.dataframe(params_res[['target_percentage_exit', 'target_percentage_entry']])
            st.dataframe(params_res[['exit_opportunity_source', 'entry_opportunity_source']])

        # Load data from backblaze
        simulated_executions_maker = backblaze.download_simulation(
            file_df.loc[file_df.file_id == file_id_download, 'results_as_maker'].values[0].replace("simulations/", ""))
        # st.dataframe(result_res)
        simulated_executions_taker = backblaze.download_simulation(
            file_df.loc[file_df.file_id == file_id_download, 'results_as_taker'].values[0].replace("simulations/", ""))

        band_values = backblaze.download_simulation(
            file_df.loc[file_df.file_id == file_id_download, 'band_values'].values[0].replace("simulations/", ""))

        # band_values = pd.merge_ordered(band_values[['Time', 'Central Band', 'timems']], model.df[['breakeven_band_exit_balancing_spot', 'breakeven_band_exit_balancing_swap', 'breakeven_band_entry_balancing_spot', 'breakeven_band_entry_balancing_swap']], on='timems')

        simulated_executions_maker = pd.merge_ordered(band_values[['Central Band', 'timems']],
                                                      simulated_executions_maker, on='timems')
        simulated_executions_maker['Central Band'].ffill(inplace=True)
        simulated_executions_maker['time_diff'] = simulated_executions_maker['timems'].diff()
        simulated_executions_maker = simulated_executions_maker.reset_index(drop=True)

        simulated_executions_taker = pd.merge_ordered(band_values[['Central Band', 'timems']],
                                                      simulated_executions_taker, on='timems')
        simulated_executions_taker['Central Band'].ffill(inplace=True)
        simulated_executions_taker['time_diff'] = simulated_executions_taker['timems'].diff()
        simulated_executions_taker = simulated_executions_taker.reset_index(drop=True)

        sim_ex_maker_entry_mask = simulated_executions_maker.side == 'entry'
        sim_ex_maker_exit_mask = simulated_executions_maker.side == 'exit'
        st.write(simulated_executions_taker)
        sim_ex_taker_entry_mask = simulated_executions_taker.side == 'entry'

        sim_ex_taker_exit_mask = simulated_executions_taker.side == 'exit'

        # simulated_executions_maker = simulated_executions_maker[~simulated_executions_maker["timestamp_spot_executed"].isnull()]
        # duration_pos_df = backblaze.download_simulation(file_df.loc[file_df.file_id == file_id_download, 'position_duration'].values[0])
        # st.dataframe(duration_pos_df)

        # parameters used in the simulation
        t_start = params_res.loc[0, 't_start']
        t_end = params_res.loc[0, 't_end']
        # band = params_res.loc[0, 'band']
        # lookback = params_res.loc[0, 'lookback']
        # recomputation_time = params_res.loc[0, 'recomputation_time']
        # target_percentage_entry = params_res.loc[0, 'target_percentage_entry']
        # target_percentage_exit = params_res.loc[0, 'target_percentage_exit']
        # entry_opportunity_source = params_res.loc[0, 'entry_opportunity_source']
        # exit_opportunity_source = params_res.loc[0, 'exit_opportunity_source']
        # environment = params_res.loc[0, 'environment']
        # strategy = params_res.loc[0, 'strategy']
        # exchange_spot = params_res.loc[0, 'exchange_spot']
        # exchange_swap = params_res.loc[0, 'exchange_swap']
        # spot_instrument = params_res.loc[0, 'spot_instrument']
        # swap_instrument = params_res.loc[0, 'swap_instrument']
        # maker_fee_spot = params_res.loc[0, 'maker_fee_spot']
        # taker_fee_spot = params_res.loc[0, 'taker_fee_spot']
        # maker_fee_swap = params_res.loc[0, 'maker_fee_swap']
        # taker_fee_swap = params_res.loc[0, 'taker_fee_swap']

        # Currently, unused variables in the report.
        # family = params_res.loc[0, 'family']
        # area_spread_threshold = params_res.loc[0, 'area_spread_threshold']
        # latency_spot = params_res.loc[0, 'latency_spot']
        # latency_swap = params_res.loc[0, 'latency_swap']
        # latency_try_post = params_res.loc[0, 'latency_try_post']
        # latency_cancel = params_res.loc[0, 'latency_cancel']
        # latency_spot_balance = params_res.loc[0, 'latency_spot_balance']
        # max_trade_volume = params_res.loc[0, 'max_trade_volume']
        # max_position = params_res.loc[0, 'max_position']
        # funding_system = params_res.loc[0, 'funding_system']

        # Display the results (descriptive statistics)

        # simulated_executions_maker['time_diff'] = simulated_executions_maker['timems'].diff()

        simulated_executions_taker = simulated_executions_taker[
            ~simulated_executions_taker["timestamp_spot_executed"].isnull()]
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

        # executions_origin = pd.DataFrame(list_executing, columns=['timems', 'current_state', 'side', 'previous_state'])

        # not_posted_df = pd.DataFrame(list_not_posted, columns=['timems', 'state', 'side', 'initial_spread'])

        # simulated_executions['exit_diff'] = -simulated_executions.loc[simulated_executions['side'] == 'exit', 'executed_spread'] + simulated_executions['Exit Band']
        # simulated_executions['entry_diff'] = simulated_executions.loc[simulated_executions['side'] == 'entry', 'executed_spread'] - simulated_executions['Entry Band']
        # simulated_executions = simulated_executions[(~simulated_executions['exit_diff'].isna()) | (~simulated_executions['entry_diff'].isna())]
        # simulated_executions.drop_duplicates(subset=['timems'], keep='last', inplace=True)

        # df_total = pd.merge_ordered(simulated_executions_maker, executions_origin, on='timems')

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
            name='Central Band')
        )

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


def plot_opportunity_points(t_start, t_end, opportunity_points, band_values):
    if len(opportunity_points) == 0:
        return
    # st.dataframe(opportunity_points)
    try:
        opportunity_points_pivot = opportunity_points.pivot(columns='side', values='opportunity')
    except:
        opportunity_points_pivot = opportunity_points.pivot(columns='type', values='opportunity')

    if len(opportunity_points_pivot) >= 2000:
        if t_end - t_start <= 1000 * 60 * 60 * 24 * 7:
            df1 = opportunity_points_pivot['entry_with_takers'].resample('5min').max()
            df2 = opportunity_points_pivot['exit_with_takers'].resample('5min').min()
            st.write('Opportunity points are aggregated to 5min interval')
        else:
            df1 = opportunity_points_pivot['entry_with_takers'].resample('10min').max()
            df2 = opportunity_points_pivot['exit_with_takers'].resample('10min').min()
            st.write('Opportunity points are aggregated to 10min interval')
        opportunity_points_pivot = pd.merge(df1, df2, left_index=True, right_index=True)

    # Plot opportunity points
    fig0 = make_subplots(rows=2, cols=1, subplot_titles=("Entry Opportunity Points", "Exit Opportunity Points"))
    fig0.append_trace(
        go.Scatter(x=opportunity_points_pivot[~opportunity_points_pivot['entry_with_takers'].isna()].index,
                   y=opportunity_points_pivot.loc[~opportunity_points_pivot['entry_with_takers'].isna(),
                   'entry_with_takers'],
                   marker=dict(color='blue'),
                   mode='markers',
                   name='Entry Opportunity with takers'), row=1, col=1)

    fig0.append_trace(go.Scatter(
        x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Time'],
        y=band_values.loc[
            (~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Entry Band'],
        line=dict(color="green"),
        line_shape='vh',
        mode='lines',
        name='Entry Band'), row=1, col=1)
    fig0.append_trace(go.Scatter(
        x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Time'],
        y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Exit Band'],
        line=dict(color="red"),
        line_shape='vh',
        mode='lines',
        name='Exit Band'), row=1, col=1)

    fig0.append_trace(
        go.Scatter(x=opportunity_points_pivot[~opportunity_points_pivot['exit_with_takers'].isna()].index,
                   y=opportunity_points_pivot.loc[~opportunity_points_pivot['exit_with_takers'].isna(),
                   'exit_with_takers'],
                   marker=dict(color='pink'),
                   mode='markers',
                   name='Exit Opportunity with takers'), row=2, col=1)

    fig0.append_trace(go.Scatter(
        x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Time'],
        y=band_values.loc[
            (~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 200), 'Entry Band'],
        line=dict(color="green"),
        line_shape='vh',
        mode='lines',
        name='Entry Band'), row=2, col=1)
    fig0.append_trace(go.Scatter(
        x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Time'],
        y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -200), 'Exit Band'],
        line=dict(color="red"),
        line_shape='vh',
        mode='lines',
        name='Exit Band'), row=2, col=1)

    fig0.update_layout(title='Opportunity Points',
                       template='plotly_dark',
                       autosize=False,
                       width=2000,
                       height=1000,
                       xaxis_title='Date',
                       yaxis_title='Spread in USD')
    return fig0


class BackfillOpportunityPointsLocal:

    def __init__(self, server_place='production', swap_symbol="XBTUSD", swap_market="BitMEX",
                 spot_symbol="BTC-PERPETUAL", spot_market="Deribit", spot_fee=0.0003, swap_fee=-0.0001):
        self.opp_points = None
        self.swapSymbol = swap_symbol
        self.spotSymbol = spot_symbol
        self.spotMarket = spot_market
        self.swapMarket = swap_market
        self.spotFee = spot_fee
        self.swapFee = swap_fee
        self.server = server_place

        self.influx_connection = InfluxConnection.getInstance()
        if self.server == 'production':
            self.swap_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        elif self.server == 'staging':
            self.swap_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        else:
            return

        if self.swapMarket == 'HuobiDMSwap':
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe, ['HuobiDM'],
                                              [self.swapSymbol])
        elif self.swapMarket == 'Okex':
            self.swap_takers_querier = Takers(self.influx_connection.staging_client_spotswap_dataframe,
                                              [self.swapMarket],
                                              [self.swapSymbol])
        else:
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe,
                                              [self.swapMarket], [self.swapSymbol])

    def get_taker_trades(self, t0, t1):
        return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)

    def backfill(self, t0, t1, latency=0):
        taker_trades = self.get_taker_trades(t0, t1)
        trades_buy = taker_trades[taker_trades['side'] == "Ask"]
        trades_sell = taker_trades[taker_trades['side'] == "Bid"]
        prices = self.swap_price_querier.query_data(t0, t1).get_data(t0, t1)
        spot_prices = self.spot_price_querier.query_data(t0, t1).get_data(t0, t1)

        prices_ask = prices[prices['side'] == 'Ask']
        prices_bid = prices[prices['side'] == 'Bid']
        prices_ask['diff'] = prices_ask['price'].diff().fillna(0)
        prices_bid['diff'] = prices_bid['price'].diff().fillna(0)
        spot_prices_ask = spot_prices[spot_prices['side'] == 'Ask']
        spot_prices_bid = spot_prices[spot_prices['side'] == 'Bid']

        points = []

        for ix, (timestamp, row) in enumerate(prices_ask.iterrows()):
            if row['diff'] <= 0:
                continue
            swap_price = prices_ask.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_buy['time'], row['time'], side="left") - 1)
            if trades_buy.iloc[sell_ix]['price'] != swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_ask['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_ask.iloc[spot_price_index]['price']
            spread = swap_price * (1 - self.swapFee) - (spot_price + spot_price * self.spotFee)

            points.append([int(row['timems']), spread, 'entry'])

        for ix, (timestamp, row) in enumerate(prices_bid.iterrows()):
            if row['diff'] >= 0:
                continue
            swap_price = prices_bid.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_sell['time'], row['time'], side="left") - 1)
            if trades_sell.iloc[sell_ix]['price'] != swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_bid['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_bid.iloc[spot_price_index]['price']
            spread = swap_price * (1 + self.swapFee) - (spot_price - spot_price * self.spotFee)

            points.append([int(row['timems']), spread, 'exit'])

        opp_points = pd.DataFrame(points, columns=['timems', 'value', 'side'])
        opp_points['entry_with_takers'] = opp_points.loc[opp_points.side == 'entry', 'value']
        opp_points['exit_with_takers'] = opp_points.loc[opp_points.side == 'exit', 'value']
        self.opp_points = opp_points
