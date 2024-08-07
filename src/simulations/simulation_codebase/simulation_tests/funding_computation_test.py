import datetime
import pandas as pd
from src.common.clients.backblaze_client import BackblazeClient
import ssl

from src.common.queries.funding_queries import funding_implementation
from src.simulations.simulation_codebase.pnl_computation_functions.pnl_computation import compute_rolling_pnl

ssl._create_default_https_context = ssl._create_unverified_context


def download_simulation_results_from_backbalaze(file_id_download: str = None, date_range: list = None):
    """
     @brief Download simulation results from backbalaze. This will be used to download the simulation results from equinoxai
     @param file_id_download File ID to download.
     @param date_range List of dates to download simulation results
    """
    backblaze = BackblazeClient()
    backblaze.authorize()
    b2 = backblaze.get_b2_resource()

    file_dataframe = backblaze.get_simulations_by_id(bucket="equinoxai-trades-db", b2=b2, id_=file_id_download,
                                                     time_from=datetime.datetime.strptime(f"{date_range[0]}",
                                                                                          '%Y-%m-%d'),
                                                     time_to=datetime.datetime.strptime(f"{date_range[1]}", '%Y-%m-%d'))
    # This function will take a file_dataframe and return a dataframe with the selected time and time as a dataframe.
    if len(file_dataframe) == 0:
        print('The run with the selected ID was not found in the selected period!')
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
            InputData_MT = file_dataframe.loc[
                file_dataframe['file_name'].str.contains('InputData_MT', case=False).values, 'file_name']
            InputData_MT = InputData_MT.str.split('simulations/').apply(lambda x: x[1])
        except:
            InputData_MT = ''

        file_df = pd.concat(
            [time_MT.reset_index(drop=True), Parameters_MT.reset_index(drop=True)], axis=1, ignore_index=True)
        file_df.rename(columns={0: 'time', 1: 'parameters'}, inplace=True)
        file_df.dropna(subset=['parameters'], inplace=True)
        file_df['parameters'] = file_df['parameters'].str.split("simulations/").apply(lambda x: x[1])
        file_df['parameters'] = file_name
        # load parameters
        if "+" in file_id_download:
            file_id_download = file_id_download.replace("+", " ")
        params_res = backblaze.download_simulation(
            file_df.loc[file_df['parameters'].str.contains(f'{file_id_download}'), 'parameters'].values[0])
        # Load data from backblaze
        result_res = backblaze.download_simulation(Results_MT[Results_MT.str.contains(f'{file_id_download}')].values[0])
        # st.dataframe(result_res)
        executions_res = backblaze.download_simulation(Sim_MT[Sim_MT.str.contains(f'{file_id_download}')].values[0])
        # st.dataframe(executions_res)
        duration_pos_df = backblaze.download_simulation(
            Position_duration_MT[Position_duration_MT.str.contains(f'{file_id_download}')].values[0])
        simulated_executions = executions_res.dropna(subset=['Entry Band', 'Exit Band'])
        simulated_executions.rename(columns={'side_x': 'side'}, inplace=True)

        try:
            funding_system = params_res.loc[0, 'funding_system']
        except:
            funding_system = 'No'

        funding_spot, funding_swap, funding_total, spot_funding, swap_funding = \
            funding_implementation(t_start=params_res.loc[0, 't_start'], t_end=params_res.loc[0, 't_end'],
                                   swap_exchange=params_res.loc[0, 'exchange_swap'],
                                   swap_symbol=params_res.loc[0, 'swap_instrument'],
                                   spot_exchange=params_res.loc[0, 'exchange_spot'],
                                   spot_symbol=params_res.loc[0, 'spot_instrument'],
                                   position_df=duration_pos_df,
                                   environment=environment)
        print('funding computation done')
        spot_funding.reset_index(drop=True, inplace=True)
        swap_funding.reset_index(drop=True, inplace=True)
        funding_df = pd.merge_ordered(spot_funding, swap_funding, on='timems', suffixes=['_spot', '_swap'])
        funding_df['cum_spot'] = funding_df['value_spot'].cumsum()
        funding_df['cum_swap'] = funding_df['value_swap'].cumsum()
        funding_df['total'] = funding_df['cum_spot'] + funding_df['cum_swap'].ffill()
        funding_df['Time'] = pd.to_datetime(funding_df['timems'], unit='ms', utc=True)
        pnl_chart = compute_rolling_pnl(funding_df, simulated_executions, funding_system)
        print('rolling pnl is done, length of pnl chart:', len(pnl_chart))

        pnl_chart.index = pnl_chart['Time']
        pnl_chart['timems'] = pnl_chart.index.astype(int) // 10 ** 6

        pnl_chart.loc[(pnl_chart['timems'] >= 1712423297430) & (pnl_chart['timems'] <= 1713805109512), :].to_csv(
            'pnl_chart1.csv', index=False)
        funding_df.loc[(funding_df['timems'] >= 1712423297430) & (funding_df['timems'] <= 1713805109512), :].to_csv(
            'funding_df1.csv', index=False)
        duration_pos_df.loc[(duration_pos_df['timems'] >= 1712423297430) & (duration_pos_df['timems'] <= 1713805109512),
        :].to_csv('position_df1.csv', index=False)
        executions_res.loc[(executions_res['timems'] >= 1712423297430) & (executions_res['timems'] <= 1713805109512),
        :].to_csv('execution_results1.csv', index=False)
        a = 3


if __name__ == '__main__':
    # t0 = 1716835522885
    t0 = 1709506800000
    # t1 = 1716875433016
    t1 = 1709852399000
    swap_exchange = 'BitMEX'
    swap_symbol = 'XBTUSD'
    spot_exchange = 'Deribit'
    spot_symbol = 'BTC-PERPETUAL'
    environment = 'staging'
    max_position = 9.047719 * 69019
    position_df = pd.DataFrame(
        {'in_pos_entry': [0, 0], 'in_pos_exit': [0, 0], 'traded_volume': [max_position, max_position],
         'timems': [t0, t1],
         'Time': [0, 0]})
    # periodical_funding_fun(t0, t1, swap_exchange, swap_symbol, position_df, environment)
    # funding_spot, funding_swap, funding_total, spot_df, swap_df = funding_implementation(t0, t1, swap_exchange,
    #                                                                                      swap_symbol, spot_exchange,
    #                                                                                      spot_symbol, position_df,
    #                                                                                      environment)
    download_simulation_results_from_backbalaze(file_id_download='6_2024+B60GPX8lVcKgGM7bzFjWGDst',
                                                date_range=['2024-05-28', '2024-06-05'])
