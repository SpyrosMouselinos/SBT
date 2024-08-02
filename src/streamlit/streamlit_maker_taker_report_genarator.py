import datetime
import numpy as np
import streamlit as st
import streamlit_permalink as stp
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.queries.queries import Prices, Takers, get_percentage_band_values, get_opportunity_points_all
from src.common.utils.quanto_utils import bitmex_eth_prices, bitmex_btc_prices
from src.common.utils.utils import sharpe_sortino_ratio_fun
from src.scripts.data_backfilling.backfill_opportunities import BackfillOpportunityPoints
from src.simulations.simulation_codebase.local_funding.funding_aggregated_function_implementation import \
    funding_implementation
from src.simulations.simulation_codebase.pnl_computation_functions.pnl_computation import \
    compute_rolling_pnl
from src.common.clients.backblaze_client import BackblazeClient

load_dotenv(find_dotenv())


def get_id(string):
    if "_id_" in string:
        split_string = string.split("_")
        return split_string[split_string.index("id") + 1]
    return None


def report_generator_maker_taker():
    funding_system_list = ['Quanto_loss', 'Quanto_profit', 'Quanto_profit_BOX', 'Quanto_profit_exp',
                           'Quanto_both', 'Quanto_both_extended']

    disable_opp = st.sidebar.selectbox('Disable Opportunity Points creation', ('yes', 'no'))
    store_opp = st.sidebar.selectbox('Store the Opportunity points', ('no', 'yes'))
    market = st.sidebar.selectbox('Type of market', ('Futures', 'Spot'))
    backblaze = BackblazeClient()
    backblaze.authorize()
    b2 = backblaze.get_b2_resource()
    # url_key_stp = randint(10**8, 10**9)
    st.title('Maker Taker Simulations Report Generator')
    date_range = stp.date_input("Enter a period where the report is genarated",
                                [datetime.date.today() - datetime.timedelta(days=7),
                                 datetime.date.today() + datetime.timedelta(days=1)])
    t_start_search = int(
        datetime.datetime(year=date_range[0].year, month=date_range[0].month, day=date_range[0].day).timestamp() * 1000)
    t_end_search = int(
        datetime.datetime(year=date_range[1].year, month=date_range[1].month, day=date_range[1].day).timestamp() * 1000)
    # t_start_search = int(date_range[0].strftime("%s")) * 1000
    # t_end_search = int(date_range[1].strftime("%s")) * 1000
    st.write('The starting time in milliseconds', t_start_search)
    st.write('The ending time in milliseconds', t_end_search)
    st.text('Default time-range is 7 days')
    id_choice = st.selectbox('Do you already know the simulation ID?', ("Choose", 'Yes', 'No'))
    start_report_button = None

    if id_choice == "No":
        file_dataframe = backblaze.get_simulations_by_name_and_date(bucket="equinoxai-trades-db",
                                                                    b2=b2,
                                                                    prefix="",
                                                                    time_from=datetime.datetime.strptime(
                                                                        f"{date_range[0]}", '%Y-%m-%d'),
                                                                    time_to=datetime.datetime.strptime(
                                                                        f"{date_range[1]}", '%Y-%m-%d')
                                                                    )

        file_dataframe.rename(columns={'filename': 'file_name', 'time': 'timestamp'}, inplace=True)

        check = st.checkbox('Click Here to view the entire data-base')
        st.write('State of the checkbox: ', check)

        if check:
            st.dataframe(file_dataframe)

        time_MT = file_dataframe.loc[
            file_dataframe['file_name'].str.contains('Parameters_MT', case=False).values, 'timestamp']
        Parameters_MT = file_dataframe.loc[
            file_dataframe['file_name'].str.contains('Parameters_MT', case=False).values, 'file_name']
        # st.dataframe(Parameters_MT)
        Results_MT = file_dataframe.loc[
            file_dataframe['file_name'].str.contains('Results_MT', case=False).values, 'file_name']
        Results_MT = Results_MT.str.split('simulations/').apply(lambda x: x[1])
        # st.dataframe(Results_MT)
        Sim_MT = file_dataframe.loc[
            file_dataframe['file_name'].str.contains('Sim_MT', case=False, regex=True).values, 'file_name']
        # st.dataframe(Sim_MT)
        Sim_MT = Sim_MT.str.split('simulations/').apply(lambda x: x[1])
        Position_duration_MT = file_dataframe.loc[
            file_dataframe['file_name'].str.contains('Position_duration_MT', case=False).values, 'file_name']
        Position_duration_MT = Position_duration_MT.str.split('simulations/').apply(lambda x: x[1])
        # st.dataframe(Position_duration_MT)
        try:
            InputData_MT = file_dataframe.loc[
                file_dataframe['file_name'].str.contains('InputData_MT', case=False).values, 'file_name']
            InputData_MT = InputData_MT.str.split('simulations/').apply(lambda x: x[1])
        except:
            InputData_MT = ''
        # st.write(len(InputData_MT.index))

        file_df = pd.concat(
            [time_MT.reset_index(drop=True), Parameters_MT.reset_index(drop=True)], axis=1, ignore_index=True)
        file_df.rename(columns={0: 'time', 1: 'parameters'}, inplace=True)
        file_df.dropna(subset=['parameters'], inplace=True)
        file_df['parameters'] = file_df['parameters'].str.apply(lambda x: x[1])

        parameters_id = file_df['parameters'].str.split('_').apply(lambda x: get_id(x))

        file_df['file_id'] = parameters_id

        st.header('Data-base containing simulation results')
        # Enter the search criteria
        search_criterion1 = stp.text_input('Search Criterion 1', url_key='criterion1')
        search_criterion2 = stp.text_input('Search Criterion 2', url_key='criterion2')

        st.write(search_criterion1)

        search_result = file_df.loc[
                        file_df['parameters'].str.contains(search_criterion1, case=False).values, :]
        search_result1 = search_result.loc[
                         search_result['parameters'].str.contains(search_criterion2, case=False).values, :]
        displayed_result = search_result1[['time', 'parameters', 'file_id']].sort_values(by='time', ascending=False,
                                                                                         ignore_index=True)
        displayed_result.reset_index(drop=True, inplace=True)

        st.dataframe(displayed_result.drop(displayed_result[displayed_result['file_id'] == 0].index))
        # st.table(file_df)
        file_id_download = stp.number_input('Select the file ID in order to download the files for the report',
                                            value=-1,
                                            url_key='file_id')

        st.subheader('If you have selected a valid ID press the start button to see the report')
        start_report_button = st.button("Start Report")

    elif id_choice == "Yes":
        file_id_download = stp.text_input('Select the file ID in order to download the files for the report', value='',
                                          url_key='file_id')
        if file_id_download != '':
            file_dataframe = backblaze.get_simulations_by_id(bucket="equinoxai-trades-db", b2=b2, id_=file_id_download,
                                                             time_from=datetime.datetime.strptime(f"{date_range[0]}",
                                                                                                  '%Y-%m-%d'),
                                                             time_to=datetime.datetime.strptime(f"{date_range[1]}",
                                                                                                '%Y-%m-%d'))
            if len(file_dataframe) == 0:
                st.text('The run with the selected ID was not found in the selected period!')
            else:
                file_dataframe.rename(columns={'filename': 'file_name', 'time': 'timestamp'}, inplace=True)
                time_MT = file_dataframe.loc[
                    file_dataframe['file_name'].str.contains('Parameters_MT', case=False).values, 'timestamp']
                Parameters_MT = file_dataframe.loc[
                    file_dataframe['file_name'].str.contains('Parameters_MT', case=False).values, 'file_name']
                file_name = Parameters_MT.iloc[0].split("simulations/")[1]
                # st.dataframe(Parameters_MT)
                Results_MT = file_dataframe.loc[
                    file_dataframe['file_name'].str.contains('Results_MT', case=False).values, 'file_name']
                Results_MT = Results_MT.str.split('simulations/').apply(lambda x: x[1])
                # st.dataframe(Results_MT)
                Sim_MT = file_dataframe.loc[
                    file_dataframe['file_name'].str.contains('Sim_MT', case=False, regex=True).values, 'file_name']
                # st.dataframe(Sim_MT)
                Sim_MT = Sim_MT.str.split('simulations/').apply(lambda x: x[1])
                Position_duration_MT = file_dataframe.loc[
                    file_dataframe['file_name'].str.contains('Position_duration_MT', case=False).values, 'file_name']
                Position_duration_MT = Position_duration_MT.str.split('simulations/').apply(lambda x: x[1])

                try:
                    Rate_Limit_MT = file_dataframe.loc[
                        file_dataframe['file_name'].str.contains('Rate_Limit_MT', case=False).values, 'file_name']
                    Rate_Limit_MT = Rate_Limit_MT.str.split('simulations/').apply(lambda x: x[1])
                except:
                    Rate_Limit_MT = None

                # st.dataframe(Position_duration_MT)
                try:
                    InputData_MT = file_dataframe.loc[
                        file_dataframe['file_name'].str.contains('InputData_MT', case=False).values, 'file_name']
                    InputData_MT = InputData_MT.str.split('simulations/').apply(lambda x: x[1])
                except:
                    InputData_MT = ''
                # st.write(len(InputData_MT.index))

                file_df = pd.concat(
                    [time_MT.reset_index(drop=True), Parameters_MT.reset_index(drop=True)], axis=1, ignore_index=True)
                file_df.rename(columns={0: 'time', 1: 'parameters'}, inplace=True)
                file_df.dropna(subset=['parameters'], inplace=True)
                file_df['parameters'] = file_df['parameters'].str.split("simulations/").apply(lambda x: x[1])
                file_df['parameters'] = file_name
                start_report_button = st.radio("Start Report", [True, False], index=0)

    if start_report_button:
        if "+" in file_id_download:
            file_id_download = file_id_download.replace("+", " ")

        params_res = backblaze.download_simulation(
            file_df.loc[file_df['parameters'].str.contains(f'{file_id_download}'), 'parameters'].values[0])
        st.title('Parameters used in the Simulation')
        st.subheader('Period of Simulation')
        col_date_1, col_date_2, col_date_3 = st.columns(3)
        col_date_1.write(
            f'Starting date {datetime.datetime.fromtimestamp(params_res.loc[0, "t_start"] / 1000.0, tz=datetime.timezone.utc).strftime("%m-%d-%Y %H:%M:%S")}')
        col_date_2.write(
            f'Ending date {datetime.datetime.fromtimestamp(params_res.loc[0, "t_end"] / 1000.0, tz=datetime.timezone.utc).strftime("%m-%d-%Y %H:%M:%S")}')
        col_date_3.write(f"Strategy {params_res.loc[0, 'strategy']}")

        st.subheader('Ratio of moving the Bands')
        try:
            st.write(f'Ratio Entry = {params_res.loc[0, "ratio_entry_band_mov"]}')
            st.write(f'Ratio Exit = {params_res.loc[0, "ratio_exit_band_mov"]}')
            st.write(f'Rolling Time Window = {params_res.loc[0, "rolling_time_window"]}')
            st.write(f'Trailing Value = {params_res.loc[0, "trailing_value"]}')
            st.write(f'Minimum Value = {params_res.loc[0, "minimum_value"]}')
        except:
            st.write(f'Ratio = {None}')

        if params_res.loc[0, 'stop_trading']:
            st.subheader('Current and High R values')
            try:
                st.write(f'Current Ratio = {params_res.loc[0, "current_r"]}')
                st.write(f'High Ratio = {params_res.loc[0, "high_r"]}')
                st.write(f'Quanto Loss threshold = {-params_res.loc[0, "quanto_threshold"]}')
                st.write(f'Hours to Stop Trading = {params_res.loc[0, "hours_to_stop"]}')
            except:
                st.write(f'Current Ratio = {None}')
                st.write(f'High Ratio = {None}')
                st.write(f'Quanto Loss threshold = {None}')

        st.write('Time in milliseconds')
        st.dataframe(params_res.loc[0, ['t_start', 't_end']])
        st.subheader('Input Parameters')
        st.dataframe(params_res.loc[0, ['family', 'environment', 'strategy']])
        st.dataframe(params_res.loc[0, ['exchange_spot', 'exchange_swap', 'spot_instrument', 'swap_instrument']])
        # st.dataframe(params_res[['spot_fee', 'swap_fee']].style.format("{:.6}"))
        try:
            st.dataframe(params_res.loc[0, ['area_spread_threshold', 'funding_system']])

        except:
            st.dataframe(params_res.loc[0, ['area_spread_threshold']])

        st.dataframe(params_res.loc[0, ['latency_spot', 'latency_swap']])
        st.dataframe(params_res.loc[0, ['latency_try_post', 'latency_cancel', 'latency_spot_balance']])
        st.dataframe(params_res.loc[0, ['max_trade_volume', 'max_position']])
        try:
            st.dataframe(params_res.loc[0, ['minimum_value', 'trailing_value', 'disable_when_below']])
        except:
            st.write('Quanto Profit is not enabled or this parameters do not exist for this simulation')

        try:
            st.dataframe(params_res.loc[0, ['window_size', 'entry_delta_spread', 'exit_delta_spread']])
        except:
            st.write('No data for the parameters of the band')

        if params_res.loc[0, 'band'] == 'percentage_band':
            st.dataframe(params_res.loc[0, ['lookback', 'recomputation_time']])
            st.dataframe(params_res.loc[0, ['target_percentage_exit', 'target_percentage_entry']])
            st.dataframe(params_res.loc[0, ['exit_opportunity_source', 'entry_opportunity_source']])
        elif params_res.loc[0, 'band'] == 'custom_multi' or params_res.loc[0, 'band'] == 'custom_multi_symmetrical':
            st.dataframe(
                params_res.loc[0, ['window_size', 'entry_delta_spread', 'exit_delta_spread', 'band_funding_system']])
        try:
            funding_options = params_res.loc[0, 'funding_options']
        except:
            funding_options = None

        if funding_options is not None and not pd.isna(funding_options):
            st.dataframe(params_res.loc[0, ['band_funding_system', 'band_funding_system2', 'funding_options']])
            st.dataframe(params_res.loc[0, ['window_size2', 'entry_delta_spread2', 'exit_delta_spread2']])

        # Load data from backblaze
        result_res = backblaze.download_simulation(Results_MT[Results_MT.str.contains(f'{file_id_download}')].values[0])
        # st.dataframe(result_res)
        executions_res = backblaze.download_simulation(Sim_MT[Sim_MT.str.contains(f'{file_id_download}')].values[0])
        # st.dataframe(executions_res)
        duration_pos_df = backblaze.download_simulation(
            Position_duration_MT[Position_duration_MT.str.contains(f'{file_id_download}')].values[0])

        if Rate_Limit_MT is not None and len(Rate_Limit_MT) != 0:
            rate_limit_df = backblaze.download_simulation(
                Rate_Limit_MT[Rate_Limit_MT.str.contains(f'{file_id_download}')].values[0])
        else:
            rate_limit_df = None

        if params_res.get('price_box_basis_points', [None])[0] is not None:
            try:
                box_signal_df = duration_pos_df[(~duration_pos_df.counts.isna())][
                    ['timems', 'end_time', 'counts', 'signal']]
                duration_pos_df = duration_pos_df[
                    (~duration_pos_df.in_pos_entry.isna()) | (~duration_pos_df.in_pos_exit.isna()) | (
                        ~duration_pos_df.out_pos.isna())][
                    ['Time', 'timems', 'in_pos_entry', 'in_pos_exit', 'out_pos', 'traded_volume']]
                box_signal_df.reset_index(drop=True, inplace=True)
                duration_pos_df.reset_index(drop=True, inplace=True)
                price_box_upper_threshold = params_res.get('price_box_upper_threshold', [0])[0]
                price_box_lower_threshold = params_res.get('price_box_lower_threshold', 0)[0]
            except:
                box_signal_df = None

        # st.dataframe(duration_pos_df)
        simulated_executions = executions_res.dropna(subset=['Entry Band', 'Exit Band'])
        simulated_executions.rename(columns={'side_x': 'side'}, inplace=True)

        # parameters used in the simulation
        t_start = params_res.loc[0, 't_start']
        t_end = params_res.loc[0, 't_end']
        band = params_res.loc[0, 'band']
        lookback = params_res.loc[0, 'lookback']
        recomputation_time = params_res.loc[0, 'recomputation_time']
        target_percentage_entry = params_res.loc[0, 'target_percentage_entry']
        target_percentage_exit = params_res.loc[0, 'target_percentage_exit']
        entry_opportunity_source = params_res.loc[0, 'entry_opportunity_source']
        exit_opportunity_source = params_res.loc[0, 'exit_opportunity_source']
        environment = params_res.loc[0, 'environment']
        strategy = params_res.loc[0, 'strategy']
        exchange_spot = params_res.loc[0, 'exchange_spot']
        exchange_swap = params_res.loc[0, 'exchange_swap']
        spot_instrument = params_res.loc[0, 'spot_instrument']
        swap_instrument = params_res.loc[0, 'swap_instrument']
        spot_fee = params_res.loc[0, 'spot_fee']
        swap_fee = params_res.loc[0, 'swap_fee']

        # Currently, unused variables in the report.
        family = params_res.loc[0, 'family']
        # area_spread_threshold = params_res.loc[0, 'area_spread_threshold']
        # latency_spot = params_res.loc[0, 'latency_spot']
        # latency_swap = params_res.loc[0, 'latency_swap']
        # latency_try_post = params_res.loc[0, 'latency_try_post']
        # latency_cancel = params_res.loc[0, 'latency_cancel']
        # latency_spot_balance = params_res.loc[0, 'latency_spot_balance']
        # max_trade_volume = params_res.loc[0, 'max_trade_volume']
        max_position = params_res.loc[0, 'max_position']
        try:
            funding_system = params_res.loc[0, 'funding_system']
        except:
            funding_system = 'No'

        # Quanto Bands parameters
        try:
            minimum_value = params_res.loc[0, 'minimum_value']
            trailing_value = params_res.loc[0, 'trailing_value']
            disable_when_below = params_res.loc[0, 'disable_when_below']
        except:
            minimum_value = 0.0
            trailing_value = 0.0
            disable_when_below = 0.0
        funding_system_name = params_res.get("funding_system_name", default=[None])[0]

        # Display the results (descriptive statistics)
        col_text1, col_text2, col_text3 = st.columns(3)
        col_text1.subheader(f"Entry Execution Quality: {round(result_res.loc[0, 'Entry Execution Quality'], 2)}")
        col_text2.subheader(f'Exit Execution Quality: {round(result_res.loc[0, "Exit Execution Quality"], 2)}')
        col_text3.subheader(f'Successfully Cancelled: {round(result_res.loc[0, "Successfully Cancelled"], 2)}')
        col_text1.subheader(f"Total Traded Volume in this period: "
                            f"{round(result_res.loc[0, 'Total Traded Volume in this period'], 2)}")
        col_text2.subheader(f"Average Daily Traded Volume in this period:"
                            f" {round(result_res.loc[0, 'Average Daily Traded Volume in this period'], 2)}")
        col_text3.subheader(f"Total Executions: {len(executions_res.dropna(subset=['Entry Band']))}")

        col_text1.subheader(f"Average Entry Spread: {round(result_res.loc[0, 'Average Entry Spread'], 2)}")
        col_text2.subheader(f"Average Exit Spread: {round(result_res.loc[0, 'Average Exit Spread'], 2)}")
        col_text3.subheader(f"Average Fixed Spread: {round(result_res.loc[0, 'Avg Fixed Spread'], 2)}")

        # funding report
        funding_spot, funding_swap, funding_total, spot_funding, swap_funding = \
            funding_implementation(t0=t_start, t1=t_end,
                                   swap_exchange=exchange_swap,
                                   swap_symbol=swap_instrument,
                                   spot_exchange=exchange_spot,
                                   spot_symbol=spot_instrument,
                                   position_df=duration_pos_df,
                                   environment=environment)
        print('funding computation done')
        # st.write('spot_funding')
        # st.dataframe(spot_funding.head(1000))
        # st.dataframe(spot_funding.tail(1000))
        # st.write('swap_funding')
        # st.dataframe(swap_funding)
        # if exchange_spot == 'Deribit':
        #     spot_funding['Time'] = pd.to_datetime(spot_funding['timems'], unit='ms', utc=True)
        #     spot_funding.set_index('Time', inplace=True)
        #     spot_funding = spot_funding.resample('5T').sum()
        #     spot_funding.reset_index(inplace=True)
        #
        # if exchange_swap == 'Deribit':
        #     swap_funding['Time'] = pd.to_datetime(swap_funding['timems'], unit='ms', utc=True)
        #     swap_funding.set_index('Time', inplace=True)
        #     swap_funding = swap_funding.resample('5T').sum()
        #     swap_funding.reset_index(inplace=True)

        spot_funding.reset_index(drop=True, inplace=True)
        swap_funding.reset_index(drop=True, inplace=True)
        funding_df = pd.merge_ordered(spot_funding, swap_funding, on='timems', suffixes=['_spot', '_swap'])
        funding_df['cum_spot'] = funding_df['value_spot'].cumsum()
        funding_df['cum_swap'] = funding_df['value_swap'].cumsum()
        funding_df['total'] = funding_df['cum_spot'] + funding_df['cum_swap'].ffill()
        funding_df['Time'] = pd.to_datetime(funding_df['timems'], unit='ms', utc=True)
        funding_df_new = funding_df[['Time', 'cum_spot', 'cum_swap', 'total']].resample('5T', on='Time').last()
        # st.dataframe(funding_df.head(1000))
        # print('funding merge is done length of funding dataframe:', len(funding_df))
        # pnl chart data
        pnl_chart = compute_rolling_pnl(funding_df, simulated_executions, funding_system)
        print('rolling pnl is done, length of pnl chart:', len(pnl_chart))

        pnl_chart.index = pnl_chart['Time']
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

        pnl_download = pnl_chart['pnl_generated_new'].resample('1D').last()
        pnl_download1 = pnl_chart['pnl_generated_new'].resample('1D').max() - pnl_chart['pnl_generated_new'].resample(
            '1D').min()
        pnl_download2 = pnl_chart['pnl_generated_new'].resample('1D').last() - pnl_chart['pnl_generated_new'].resample(
            '1D').first()
        pnl_daily_df = pd.concat([pnl_download, pnl_download1, pnl_download2], axis=1,
                                 keys=['last', 'diff max-min', 'diff last-first'])

        # Sharpe Ratio computation
        num_of_days_in_period = (pnl_daily_df.index[-1] - pnl_daily_df.index[0]).days
        aum = int(2 * max_position / 9)
        if params_res.get('leverage', None) is not None:
            aum = float(2 * params_res['max_position'] / params_res.get('leverage'))
        st.subheader(
            f"Max daily drawdown {round(100 * np.array(drawdowns_d)[:, 1].astype(np.float64).max() / aum, 3)}%")
        st.subheader(
            f"Max weekly drawdown {round(100 * np.array(drawdowns_w)[:, 1].astype(np.float64).max() / aum, 3)}%")
        mean_daily_ror, std_daily_ror, sharpe_ratio, sortino_ratio = \
            sharpe_sortino_ratio_fun(df=pnl_daily_df, aum=aum,
                                     t_start=params_res.loc[0, 't_start'],
                                     t_end=params_res.loc[0, 't_end'])
        col_text1.subheader(f'Funding in Spot Market: {round(funding_spot, 2)}')
        col_text2.subheader(f'Funding in Swap Market: {round(funding_swap, 2)}')
        col_text3.subheader(f'Funding in Total: {round(funding_total, 2)}')
        st.write('If positive we get paid, if negative we pay this amount of USD')

        col_des1, col_des2, col_des3 = st.columns(3)
        try:
            col_des1.subheader(f"Total Traded Volume in Underlying Coin: "
                               f"{round(result_res.loc[0, 'Total Traded Volume in this Period in Coin Volume'], 4)}")

            col_des2.subheader(f"Average Daily Traded Volume in this period:"
                               f" {round(result_res.loc[0, 'Average Daily Traded Volume in this period in Coin'], 2)}")
            col_des3.subheader(
                f"Estimated PNL in USD: {round(result_res.loc[0, 'Estimated PNL with Funding'], 2)}")
        except:
            col_des1.subheader("Total Traded Volume in Underlying Coin: NO DATA")

            col_des2.subheader("Average Daily Traded Volume in this period: NO DATA")
            col_des3.subheader("Estimated PNL in USD: NO DATA")

        quanto1, quanto2 = st.columns(2)
        try:
            if funding_system in ['Quanto_loss', 'Quanto_both', 'Quanto_both_extended']:
                try:
                    qp = result_res.loc[0, 'quanto_profit']
                except:
                    qp = simulated_executions.loc[simulated_executions['side'] == 'exit', 'quanto_profit'].sum()
                quanto1.subheader(
                    f"Quanto Profit: {round(qp, 2)}")
                quanto2.subheader(
                    f"PNL Quanto Profit/Loss Included: {round(result_res.loc[0, 'Estimated PNL'] + funding_total + qp, 2)}")
            elif funding_system == 'Quanto_profit' or funding_system == 'Quanto_profit_BOX' or \
                    funding_system == 'Quanto_profit_exp':
                try:
                    qp = result_res.loc[0, 'quanto_profit']
                except:
                    qp = simulated_executions.loc[simulated_executions['side'] == 'entry', 'quanto_profit'].sum()
                quanto1.subheader(
                    f"Quanto Profit: {round(qp, 2)}")
                quanto2.subheader(
                    f"PNL Quanto Profit/Loss Included: {round(result_res.loc[0, 'Estimated PNL'] + funding_total + qp, 2)}")
        except:
            if funding_system in ['Quanto_loss', 'Quanto_profit', 'Quanto_profit_BOX', 'Quanto_profit_exp',
                                  'Quanto_both']:
                st.subheader(
                    f"PNL Quanto Profit/Loss Included: NO DATA")

        if market == 'Futures':
            st.subheader(f"Assets allocated: {int(2 * max_position / 11)} USD")
        elif market == 'Spot':
            st.subheader(f"Assets allocated: {int(max_position)} USD")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.subheader(f"Sharpe Ratio of the period {round(sharpe_ratio, 4)}")
        col_res2.subheader(f"Sortino Ratio of the period {round(sortino_ratio, 4)}")
        st.write("Sharpe Ratio = SQRT(365) * (Average of Daily Standard Return - Risk Free Ratio / 365) / "
                 "Standard Deviation of Daily Standard Return")

        try:
            ror_annualized = (result_res.loc[0, 'Estimated PNL'] + funding_total + qp) / aum * 365 / \
                             num_of_days_in_period * 100
        except:
            ror_annualized = (result_res.loc[0, 'Estimated PNL'] + funding_total) / aum * 365 / \
                             num_of_days_in_period * 100

        col_res3.subheader(f"ROR Annualized (%) {round(ror_annualized, 2)}%")
        # Create the execution plot
        if band == 'percentage_band':
            band_values = get_percentage_band_values(t0=t_start, t1=t_end,
                                                     lookback=lookback,
                                                     recomputation_time=recomputation_time,
                                                     target_percentage_exit=target_percentage_exit,
                                                     target_percentage_entry=target_percentage_entry,
                                                     entry_opportunity_source=entry_opportunity_source,
                                                     exit_opportunity_source=exit_opportunity_source,
                                                     spot_name=exchange_spot,
                                                     spot_instrument=f"hybrid_{spot_instrument}",
                                                     swap_instrument=f"hybrid_{swap_instrument}",
                                                     environment=environment)
            band_values.rename(columns={'entry_band': 'Entry Band', 'exit_band': 'Exit Band'}, inplace=True)
        else:
            try:
                band_values = backblaze.download_simulation(InputData_MT[InputData_MT.str.contains(f'{file_id_download}'
                                                                                                   )].values[0])
                band_values['Time'] = pd.to_datetime(band_values['timems'], unit='ms', utc=True)
                band_values.index = band_values.Time
            except:
                band_values = backblaze.download_simulation(InputData_MT[InputData_MT.str.contains(f'{file_id_download}'
                                                                                                   )].values[0])
                if "timems_x" in band_values.columns:
                    band_values.drop(columns=['timems_y'], inplace=True)
                    band_values.rename(columns={'timems_x': 'timems'}, inplace=True)
                band_values['Time'] = pd.to_datetime(band_values['timems'], unit='ms', utc=True)
                band_values.index = band_values.Time
                band_values = band_values.resample('5min').max()

        if funding_system in funding_system_list:
            exec_df = pd.merge_ordered(simulated_executions[['timems', 'side', 'executed_spread']],
                                       band_values[['timems', 'Entry Band', 'Exit Band', 'Entry Band with Quanto loss',
                                                    'Exit Band with Quanto loss']], on='timems')

            for col in ['Entry Band', 'Exit Band', 'Entry Band with Quanto loss', 'Exit Band with Quanto loss']:
                exec_df[col].ffill(inplace=True)
            exec_df.dropna(subset=['side'], inplace=True)
            exit_impact = len(exec_df[(exec_df['Exit Band'] < exec_df['Exit Band with Quanto loss']) &
                                      (exec_df['Exit Band'] < exec_df['executed_spread']) &
                                      (exec_df['side'] == 'exit')].index) / \
                          len(exec_df[exec_df['side'] == 'exit'].index)
            exit_impact_perc = round(exit_impact, 4) * 100

            if params_res.loc[0, 'ratio_entry_band_mov'] < 0:
                entry_impact = len(exec_df[(exec_df['Entry Band'] > exec_df['Entry Band with Quanto loss']) &
                                           (exec_df['Entry Band'] > exec_df['executed_spread']) &
                                           (exec_df['side'] == 'entry')].index) / \
                               len(exec_df[exec_df['side'] == 'entry'].index)
                entry_impact_perc = round(entry_impact, 4) * 100
            else:
                entry_impact = len(exec_df[(exec_df['Entry Band'] < exec_df['Entry Band with Quanto loss']) &
                                           (exec_df['Entry Band with Quanto loss'] < exec_df['executed_spread']) &
                                           (exec_df['side'] == 'entry')].index) / \
                               len(exec_df[exec_df['side'] == 'entry'].index)
                entry_impact_perc = round(entry_impact, 4) * 100
            impact_col1, impact_col2 = st.columns(2)
            impact_col1.subheader(f'Exit Band movement impact: {exit_impact_perc}%')
            impact_col2.subheader(f'Entry Band movement impact: {entry_impact_perc}%')

        # # Prices for ETHUSD strategies.
        if exchange_swap == 'BitMEX' and exchange_spot == 'Deribit' and swap_instrument == 'ETHUSD' and \
                spot_instrument == 'ETH-PERPETUAL':
            price_eth_t = bitmex_eth_prices(t0=t_start, t1=t_end, environment=environment, split_data=False)
            price_eth_t.set_index('Time', drop=True, inplace=True)
            price_btc_t = bitmex_btc_prices(t0=t_start, t1=t_end, environment=environment, split_data=False)
            price_btc_t.set_index('Time', drop=True, inplace=True)
            if t_end - t_start <= 1000 * 60 * 60 * 24 * 7:
                price_eth = price_eth_t['price_ask'].resample('5min').mean()
                price_btc = price_btc_t['price_ask'].resample('5min').mean()
                st.write('Price points are aggregated to 5min interval')
            else:
                price_eth = price_eth_t['price_ask'].resample('10min').mean()
                price_btc = price_btc_t['price_ask'].resample('10min').mean()
                st.write('Price points are aggregated to 10min interval')
            # st.dataframe(opportunity_points_pivot)
            prices_fig = go.Figure()
            prices_fig.add_trace(go.Scatter(x=price_eth.index,
                                            y=price_eth,
                                            marker=dict(color='orange'),
                                            line_shape='vh',
                                            mode='lines',
                                            name='BitMEX ETH Ask Price',
                                            opacity=0.5))
            prices_fig.add_trace(go.Scatter(x=price_btc.index,
                                            y=price_btc,
                                            marker=dict(color='blue'),
                                            line_shape='vh',
                                            mode='lines',
                                            name='BitMEX BTC Ask Price',
                                            opacity=0.5,
                                            yaxis="y2"
                                            )
                                 )
            prices_fig.add_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'entry', 'Time'],
                                            y=simulated_executions.loc[
                                                simulated_executions['side'] == 'entry', 'executed_spread'],
                                            marker=dict(color='green'),
                                            mode='markers',
                                            name='Sim Entry Exec',
                                            yaxis="y3"
                                            )
                                 )
            prices_fig.add_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'exit', 'Time'],
                                            y=simulated_executions.loc[
                                                simulated_executions['side'] == 'exit', 'executed_spread'],
                                            marker=dict(color='orange'),
                                            mode='markers',
                                            name='Sim Exit Exec',
                                            yaxis="y3"
                                            )
                                 )
            prices_fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 400), 'Time'],
                y=band_values.loc[
                    (~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 400), 'Entry Band'],
                line=dict(color="green"),
                opacity=0.7,
                line_shape='vh',
                mode='lines',
                name='Entry Band',
                yaxis="y3"
            )
            )
            prices_fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -400), 'Exit Band'],
                line=dict(color="red"),
                opacity=0.7,
                line_shape='vh',
                mode='lines',
                name='Exit Band',
                yaxis="y3"
            ))
            if funding_system in funding_system_list:
                prices_fig.add_trace(go.Scatter(
                    x=band_values.loc[(~band_values['Exit Band with Quanto loss'].isna()) &
                                      (band_values['Exit Band with Quanto loss'] >= -50) &
                                      (band_values['Exit Band with Quanto loss'] <= 50), 'Time'],
                    y=band_values.loc[(~band_values['Exit Band with Quanto loss'].isna()) &
                                      (band_values['Exit Band with Quanto loss'] >= -200) &
                                      (band_values['Exit Band with Quanto loss'] <= 50),
                    'Exit Band with Quanto loss'],
                    line=dict(color="red", dash='dash', width=0.5),
                    opacity=0.5,
                    line_shape='vh',
                    mode='lines',
                    name='Exit Band with Quanto Loss',
                    yaxis="y3"

                ))
                prices_fig.add_trace(go.Scatter(
                    x=band_values.loc[(~band_values['Entry Band with Quanto loss'].isna()) & (
                            band_values['Entry Band with Quanto loss'] >= -50) &
                                      (band_values['Exit Band with Quanto loss'] <= 50), 'Time'],
                    y=band_values.loc[(~band_values['Entry Band with Quanto loss'].isna()) & (
                            band_values['Entry Band with Quanto loss'] >= -50) &
                                      (band_values['Entry Band with Quanto loss'] <= 50),
                    'Entry Band with Quanto loss'],
                    line=dict(color="green", dash='dash', width=0.5),
                    opacity=0.5,
                    line_shape='vh',
                    mode='lines',
                    name='Entry Band with Quanto Loss',
                    yaxis="y3"
                ))

            prices_fig.update_layout(title='ETH-BTC Ask Prices',
                                     autosize=False,
                                     height=1000,
                                     xaxis_title='Date',
                                     yaxis=dict(
                                         title="USD",
                                         titlefont=dict(
                                             color="#ff7f0e"
                                         ),
                                         tickfont=dict(
                                             color="#ff7f0e"
                                         )
                                     ),
                                     yaxis2=dict(
                                         title="USD",
                                         titlefont=dict(
                                             color="#1f77b4"
                                         ),
                                         tickfont=dict(
                                             color="#1f77b4"
                                         ),
                                         anchor="free",
                                         overlaying="y",
                                         side="left",
                                         position=0.05
                                     ),
                                     yaxis3=dict(
                                         title="USD",
                                         anchor="x",
                                         overlaying="y",
                                         side="right",
                                     ),

                                     )
            st.plotly_chart(prices_fig, use_container_width=True)

        fig = go.Figure()
        try:
            # spread_type = 'executed_spread' if not params_res.use_id else 'executed_spread_bp'
            spread_type = 'executed_spread' if not params_res.use_bp.all() else 'executed_spread_bp'
        except:
            spread_type = 'executed_spread'
            st.write('No use_id found in dataframe')
        fig.add_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'entry', 'Time'],
                                 y=simulated_executions.loc[
                                     simulated_executions['side'] == 'entry', spread_type],
                                 marker=dict(color='green'),
                                 mode='markers',
                                 name='Sim Entry Exec'))
        fig.add_trace(go.Scatter(x=simulated_executions.loc[simulated_executions['side'] == 'exit', 'Time'],
                                 y=simulated_executions.loc[
                                     simulated_executions['side'] == 'exit', spread_type],
                                 marker=dict(color='orange'),
                                 mode='markers',
                                 name='Sim Exit Exec'))
        fig.add_trace(go.Scatter(
            x=band_values.loc[(~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 400), 'Time'],
            y=band_values.loc[
                (~band_values['Entry Band'].isna()) & (band_values['Entry Band'] <= 400), 'Entry Band'],
            line=dict(color="green"),
            opacity=0.7,
            line_shape='vh',
            mode='lines',
            name='Entry Band'))
        fig.add_trace(go.Scatter(
            x=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -400), 'Time'],
            y=band_values.loc[(~band_values['Exit Band'].isna()) & (band_values['Exit Band'] >= -400), 'Exit Band'],
            line=dict(color="red"),
            opacity=0.7,
            line_shape='vh',
            mode='lines',
            name='Exit Band'))
        if funding_system in funding_system_list:
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Exit Band with Quanto loss'].isna()) &
                                  (band_values['Exit Band with Quanto loss'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Exit Band with Quanto loss'].isna()) &
                                  (band_values['Exit Band with Quanto loss'] >= -400),
                'Exit Band with Quanto loss'],
                line=dict(color="red", dash='dash', width=0.5),
                opacity=0.5,
                line_shape='vh',
                mode='lines',
                name='Exit Band with Quanto Loss'))
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Entry Band with Quanto loss'].isna()) & (
                        band_values['Entry Band with Quanto loss'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Entry Band with Quanto loss'].isna()) & (
                        band_values['Entry Band with Quanto loss'] >= -400), 'Entry Band with Quanto loss'],
                line=dict(color="green", dash='dash', width=0.5),
                opacity=0.5,
                line_shape='vh',
                mode='lines',
                name='Entry Band with Quanto Loss'))
        elif funding_options is not None and not pd.isna(funding_options):
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Exit Band Exit to Zero'].isna()) &
                                  (band_values['Exit Band Exit to Zero'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Exit Band Exit to Zero'].isna()) &
                                  (band_values['Exit Band Exit to Zero'] >= -400), 'Exit Band Exit to Zero'],
                line=dict(color="red", dash='dash', width=0.5),
                opacity=1.0,
                line_shape='vh',
                mode='lines',
                name='Exit Band Exit to Zero'))
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Entry Band Enter to Zero'].isna()) & (
                        band_values['Entry Band Enter to Zero'] <= 400), 'Time'],
                y=band_values.loc[(~band_values['Entry Band Enter to Zero'].isna()) & (
                        band_values['Entry Band Enter to Zero'] <= 400), 'Entry Band Enter to Zero'],
                line=dict(color="green", dash='dash', width=0.5),
                opacity=1.0,
                line_shape='vh',
                mode='lines',
                name='Entry Band Enter to Zero'))

        if funding_system_name is not None and 'Exit Band with Funding adjustment' in band_values.columns:
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Exit Band with Funding adjustment'].isna()) &
                                  (band_values['Exit Band with Funding adjustment'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Exit Band with Funding adjustment'].isna()) &
                                  (band_values['Exit Band with Funding adjustment'] >= -400),
                'Exit Band with Funding adjustment'],
                line=dict(color="red", dash='dash', width=0.5),
                opacity=0.5,
                line_shape='hv',
                mode='lines',
                name='Exit Band with Funding Adjustment'))
            fig.add_trace(go.Scatter(
                x=band_values.loc[(~band_values['Entry Band with Funding adjustment'].isna()) & (
                        band_values['Entry Band with Funding adjustment'] >= -400), 'Time'],
                y=band_values.loc[(~band_values['Entry Band with Funding adjustment'].isna()) & (
                        band_values[
                            'Entry Band with Funding adjustment'] >= -400), 'Entry Band with Funding adjustment'],
                line=dict(color="green", dash='dash', width=0.5),
                opacity=0.5,
                line_shape='hv',
                mode='lines',
                name='Entry Band with Funding Adjustment'))

        fig.update_layout(title='Simulated Executions',
                          autosize=False,
                          height=1000,
                          xaxis_title='Date',
                          yaxis_title='Spread in USD')
        st.plotly_chart(fig, use_container_width=True)
        if params_res.get('price_box_basis_points', [None])[0] is not None:
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=box_signal_df['end_time'], y=box_signal_df[f"signal"],
                                         mode='markers+lines', marker_color="green",
                                         name=f'Box TP signal'))

                fig.add_hline(y=price_box_lower_threshold, line_dash="dot", line_width=2)
                fig.add_hline(y=price_box_upper_threshold, line_dash="dot", line_width=2)
                fig.update_yaxes(range=[-0.1, 4.6])
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.write('no box mechanism found')

        # plot quanto profit / loss
        if funding_system in funding_system_list:
            try:
                fig_quanto = go.Figure()
                fig_quanto.add_trace(go.Scatter(
                    x=band_values['Time'],
                    y=band_values['quanto_profit'],
                    marker=dict(color='green'),
                    mode='lines',
                    line_shape='vh',
                    name='Quanto Profit'))

                fig_quanto.update_layout(title='Quanto Profit / Loss',
                                         autosize=False,
                                         height=1000,
                                         xaxis_title='Date',
                                         yaxis_title='USD/ETH')
                st.plotly_chart(fig_quanto, use_container_width=True)
            except:
                st.write("No data found for Quanto Profit")

        fig_funding = make_subplots(rows=5, cols=1, subplot_titles=("Funding", "PNL over time", "Position Duration",
                                                                    "PNL Max-Min", "PNL Last-First"), shared_xaxes=True)
        fig_funding.add_trace(go.Scatter(
            x=funding_df_new.index,
            y=funding_df_new['cum_spot'].ffill(),
            marker=dict(color='green'),
            mode='lines',
            line_shape='vh',
            name='Spot Funding'), row=1, col=1)
        fig_funding.add_trace(go.Scatter(
            x=funding_df_new.index,
            y=funding_df_new['cum_swap'].ffill(),
            marker=dict(color='blue'),
            mode='lines',
            line_shape='vh',
            name='Swap Funding'), row=1, col=1)
        fig_funding.add_trace(go.Scatter(
            x=funding_df_new.index,
            y=funding_df_new['total'].ffill(),
            marker=dict(color='red'),
            mode='lines',
            line_shape='vh',
            name='Total Funding'), row=1, col=1)

        pnl_chart_new = pnl_chart[['Time', 'pnl_generated_new']].resample('5T', on='Time').last()

        fig_funding.add_trace(go.Scatter(
            x=pnl_chart_new.index[:-1],
            y=pnl_chart_new.iloc[:-1, 0].fillna(method='ffill'),
            marker=dict(color='red'),
            mode='lines',
            line_shape='vh',
            name='PNL'), row=2, col=1)
        fig_funding.add_trace(go.Scatter(x=duration_pos_df['Time'],
                                         y=duration_pos_df['out_pos'],
                                         opacity=0.5,
                                         connectgaps=False,
                                         line=dict(color="blue"),
                                         mode='lines',
                                         name='Out 0f Position'), row=3, col=1)
        fig_funding.add_trace(go.Scatter(x=duration_pos_df['Time'],
                                         y=duration_pos_df['in_pos_entry'],
                                         line=dict(color="green"),
                                         mode='lines',
                                         connectgaps=False,
                                         name='In Position Enter'), row=3, col=1)
        fig_funding.add_trace(go.Scatter(x=duration_pos_df['Time'],
                                         y=duration_pos_df['in_pos_exit'],
                                         line=dict(color="gold"),
                                         mode='lines',
                                         connectgaps=False,
                                         name='In Position Exit'), row=3, col=1)

        if swap_instrument == 'XBTUSD' or swap_instrument == 'ETHUSD':
            ix_time = t_start + 1000 * 60 * 60 * 4
            # st.write(ix_time)
            while ix_time <= t_end:
                time_ix = datetime.datetime.fromtimestamp(ix_time / 1000.0, tz=datetime.timezone.utc)
                fig_funding.add_vline(x=time_ix, line_color='green', row=3, col=1)
                ix_time = ix_time + 1000 * 60 * 60 * 8
                # st.write(ix_time)

        fig_funding.add_trace(go.Scatter(
            x=pnl_daily_df.index,
            y=pnl_daily_df['diff max-min'],
            marker=dict(color='black', size=10, symbol='diamond'),
            mode='lines+markers',
            line_shape='spline',
            name='PNL Daily Difference'), row=4, col=1)

        fig_funding.add_trace(go.Scatter(
            x=pnl_daily_df.index,
            y=pnl_daily_df['diff last-first'],
            marker=dict(color='brown', size=10, symbol='diamond'),
            mode='lines+markers',
            line_shape='spline',
            name='PNL Daily Difference'), row=5, col=1)

        fig_funding.update_layout(title='Funding',
                                  autosize=False,
                                  height=1500,
                                  xaxis_title='Date',
                                  yaxis_title='USD',
                                  xaxis_showticklabels=True,
                                  xaxis2_showticklabels=True,
                                  xaxis3_showticklabels=True,
                                  xaxis4_showticklabels=True,
                                  xaxis5_showticklabels=True
                                  )
        st.plotly_chart(fig_funding, use_container_width=True)

        st.download_button(
            "Press to Download PNL Data",
            pnl_daily_df.to_csv(index=True).encode('UTF-8'),
            f"daily_pnl_{file_id_download}.csv",
            key='download-csv'
        )

        # rate limit figure
        if rate_limit_df is not None:
            fig_rate_limit = rate_limit_analysis(t_start, t_end, rate_limit_df, fig.data)
            st.plotly_chart(fig_rate_limit, use_container_width=True)
            hist_rate_limit = rate_limit_histogram(rate_limit_df)
            st.plotly_chart(hist_rate_limit, use_container_width=True)

        if disable_opp == 'no':
            # Create Opportunity points.
            try:
                opportunity_points = get_opportunity_points_all(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                                spot=f'hybrid_{spot_instrument}',
                                                                swap=f'hybrid_{swap_instrument}',
                                                                environment=environment)
            except:
                with st.spinner('Wait for opportunity points to be created ...'):
                    if store_opp == 'no':
                        app = BackfillOpportunityPointsLocal(server_place=environment, swap_symbol=swap_instrument,
                                                             swap_market=exchange_swap, spot_symbol=spot_instrument,
                                                             spot_market=exchange_spot, spot_fee=spot_fee,
                                                             swap_fee=swap_fee)

                        app.backfill(t0=t_start, t1=t_end, latency=0)
                        opportunity_points = app.opp_points
                    else:
                        app = BackfillOpportunityPoints(server_place=environment, swap_symbol=swap_instrument,
                                                        swap_market=exchange_swap, spot_symbol=spot_instrument,
                                                        spot_market=exchange_spot, spot_fee=spot_fee,
                                                        swap_fee=swap_fee)

                        app.backfill(t0=t_start, t1=t_end, latency=0)
                        opportunity_points = get_opportunity_points_all(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                                        spot=spot_instrument, swap=swap_instrument,
                                                                        environment=environment)
                st.success('Done!')

            fig0 = plot_opportunity_points(t_start, t_end, opportunity_points, band_values)
            st.plotly_chart(fig0, use_container_width=True)

        if st.radio('Show Data', [True, False], index=1):
            st.write('Executions results')
            st.dataframe(executions_res)
            st.write('simulated executions dataframe')
            st.dataframe(simulated_executions)
            st.write('Position')
            st.dataframe(duration_pos_df)
            st.write("bands")
            st.dataframe(band_values)


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
    fig0 = make_subplots(rows=2, cols=1, subplot_titles=("Entry Opportunity Points", "Exit Opportunity Points"),
                         shared_xaxes=True)
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
                       yaxis_title='Spread in USD',
                       xaxis_showticklabels=True,
                       xaxis2_showticklabels=True
                       )
    return fig0


if __name__ == '__main__':
    report_generator_maker_taker()


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
        trades_buy = taker_trades[taker_trades['side'] == "Bid"]
        trades_sell = taker_trades[taker_trades['side'] == "Ask"]
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
            if trades_buy.iloc[sell_ix]['price'] < swap_price:
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
            if trades_sell.iloc[sell_ix]['price'] > swap_price:
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


# a=report_generator_maker_taker()

def rate_limit_analysis(t_start, t_end, df, fig):
    price_eth_t = bitmex_eth_prices(t0=t_start, t1=t_end, environment='staging', split_data=False)
    price_eth_t.set_index('Time', drop=True, inplace=True)
    price_btc_t = bitmex_btc_prices(t0=t_start, t1=t_end, environment='staging', split_data=False)
    price_btc_t.set_index('Time', drop=True, inplace=True)
    if t_end - t_start <= 1000 * 60 * 60 * 24 * 7:
        price_eth = price_eth_t['price_ask'].resample('5min').mean()
        price_btc = price_btc_t['price_ask'].resample('5min').mean()
    else:
        price_eth = price_eth_t['price_ask'].resample('10min').mean()
        price_btc = price_btc_t['price_ask'].resample('10min').mean()

    df['Time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    mask = df[df.condition_state == False].index

    figure = make_subplots(rows=3, cols=1, subplot_titles=("Trading", "Rate Limit"), shared_xaxes=True,
                           row_heights=[0.45, 0.15, 0.4])

    for idx in range(len(fig)):
        figure.add_trace(fig[idx], row=1, col=1)

    figure.add_trace(go.Scatter(x=df.Time,
                                y=df.counter,
                                mode='lines',
                                name='Try Post Count',
                                line=dict(color='blue')
                                ),
                     row=2, col=1
                     )

    figure.add_trace(go.Scatter(x=df.loc[mask, 'Time'],
                                y=[1] * len(df.loc[mask, 'Time']),
                                mode='markers',
                                marker=dict(color='red', size=5),
                                name='RL Cond=True'
                                ),
                     row=2, col=1
                     )

    figure.add_hline(y=int(df.minute_limit.unique()),
                     line_dash="dot",
                     line_color="red",
                     annotation_text='Minute Limit',
                     annotation_position='bottom right',
                     row=2, col=1
                     )

    figure.add_hline(y=int(df.second_limit.unique()),
                     line_dash="dot",
                     line_color="red",
                     annotation_text='Second Limit',
                     annotation_position='bottom right',
                     row=2, col=1
                     )

    figure.add_trace(go.Scatter(x=price_btc.index,
                                y=price_btc,
                                marker=dict(color='blue'),
                                line_shape='vh',
                                mode='lines',
                                opacity=0.5,
                                name='BTC Price'
                                ),
                     row=3, col=1
                     )

    figure.update_layout(title='Rate Limit Analysis',
                         autosize=False,
                         height=1500,
                         xaxis_title='Date',
                         xaxis2_title='Date',
                         xaxis3_title='Date',
                         yaxis_title='USD',
                         yaxis2_title='USD',
                         yaxis3_title='Counts from clear to try_to_post',
                         xaxis_showticklabels=True,
                         xaxis2_showticklabels=True,
                         xaxis3_showticklabels=True
                         )

    return figure


def rate_limit_histogram(df):
    hist = make_subplots(rows=2, cols=1, subplot_titles=("Rate Limit Counter", "Rate Limit Activated"))
    hist.add_trace(go.Histogram(x=df.loc[df.counter != 0, 'counter']), row=1, col=1)
    hist.add_trace(go.Histogram(x=df.condition_state, histnorm='probability'), row=2, col=1)
    return hist
