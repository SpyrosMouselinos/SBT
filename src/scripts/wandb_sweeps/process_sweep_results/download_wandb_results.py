########################################################################################################################
# script to combine training simulations with their confirmations and produce a single csv file
########################################################################################################################
import argparse
import pandas as pd
import wandb
import os
import numpy as np
from src.common.constants.constants import WANDB_ENTITY
from dotenv import find_dotenv, load_dotenv
from pytictoc import TicToc
from tqdm import tqdm
from src.common.utils.utils import parse_args
import datetime
load_dotenv(find_dotenv())


class AutomateParameterSelection:
    def __init__(self, project_name: str ='taker_maker_simulations_2024_1') -> None:
        self.project_name = project_name

    def download_sweep_results(self, sweep_id: str = 'jd1a03uf'):
        """Download and compile results from a specified sweep in Weights & Biases.

        This function connects to the Weights & Biases API to retrieve the
        results of a sweep identified by the provided sweep ID. It logs in using
        the specified credentials and fetches the runs associated with the
        sweep. The results are compiled into a DataFrame that includes
        configuration parameters and summary metrics for each run. If no runs
        are found with the initial sorting criteria, it attempts to fetch the
        runs again using an alternative sorting method. The final DataFrame is
        returned after removing any duplicate columns.

        Args:
            sweep_id (str): The ID of the sweep to download results from. Defaults to 'jd1a03uf'.

        Returns:
            pandas.DataFrame: A DataFrame containing the compiled results of the sweep, including
            configuration parameters and summary metrics.
        """

        t = TicToc()
        t.tic()
        wandb.login(host=os.getenv("WANDB_HOST"), key="local-c079a1f81a639c9546d4e0a7790074d341572ef7")

        api = wandb.Api({'entity': WANDB_ENTITY, 'project':  f'{self.project_name}', 'sweep': f'{sweep_id}'}, timeout=300)
        runs = api.runs(f"{WANDB_ENTITY}/{self.project_name}",
                        filters={"sweep": sweep_id},
                        order="-summary_metrics.Estimated PNL with Funding"
                        )
        if len(runs) == 0:
            api = wandb.Api({'entity': WANDB_ENTITY, 'project': f'{self.project_name}', 'sweep': f'{sweep_id}'},
                            timeout=300)
            runs = api.runs(
                f"{WANDB_ENTITY}/{self.project_name}",
                filters={"sweep": sweep_id},
                order="-summary_metrics.Estimated PNL with Quanto_profit"
            )

        print(f'project_name : {self.project_name}')
        print(f'sweep_id: {sweep_id}')
        print(f'num of runs: {len(runs)}')
        for idx, run in tqdm(enumerate(runs)):
            if idx == 0:
                config_dict = run.config
                params = pd.DataFrame(config_dict, index=[0])
                run.summary._json_dict.update(config_dict)
                sweep_results_df = pd.DataFrame(run.summary._json_dict, index=[0])
                sweep_results_df = sweep_results_df.reindex(sorted(sweep_results_df.columns), axis=1)

            if len(dict(run.summary)) > 1 and idx != 0:
                config_dict = run.config
                params_df = pd.DataFrame(config_dict, index=[0])
                params = pd.concat([params, params_df], ignore_index=True)
                run.summary._json_dict.update(config_dict)
                df = pd.DataFrame(run.summary._json_dict, index=[0])
                df = df.reindex(sorted(df.columns), axis=1)
                sweep_results_df = pd.concat([sweep_results_df, df], ignore_index=True)

        f_df = pd.concat([params, sweep_results_df], axis=1)
        t.toc()
        df = f_df.loc[:, ~f_df.columns.duplicated(keep='last')]
        return df.replace("NaN", np.nan)

    def add_date_column_df(self, df):
        """Add a formatted date column to a DataFrame.

        This function takes a DataFrame and adds a new column named 'Date'. The
        'Date' column is constructed by formatting the start and end dates from
        the existing 'date_start' and 'date_end' columns. The format used is
        'MM-YYYY to MM-YYYY'. This allows for a clear representation of the date
        range directly within the DataFrame, making it easier to analyze and
        visualize date-related data.

        Args:
            df (pandas.DataFrame): The input DataFrame containing 'date_start'

        Returns:
            pandas.DataFrame: The original DataFrame with an additional 'Date'
            column.
        """

        df['Date'] = df.loc[0, 'date_start'].split("-")[1] + '-' + df.loc[0, 'date_start'].split("-")[0] + ' to ' \
                     + df.loc[0, 'date_end'].split("-")[1] + '-' + df.loc[0, 'date_end'].split("-")[0]
        return df

    def filter_data(self, df, global_filter: dict = {"Average_Distance": [1.5, "s"], "max_drawdown_d": [1.7, "s"],
                                                     "max_drawup_d": [1.7, "s"],
                                                     "Std_daily_ROR": [0.01, "s"], "ROR Annualized": [30, "g"],
                                                     "Sharpe Ratio": [4, "g"]}):
        """Filter a DataFrame based on specified global criteria.

        This method applies a series of filters to the provided DataFrame based
        on the conditions defined in the `global_filter` dictionary. Each key in
        the dictionary corresponds to a column in the DataFrame, and the
        associated value is a list where the first element is the threshold for
        filtering and the second element indicates whether to filter for values
        less than or equal to (denoted by "s") or greater than or equal to
        (denoted by "g") the threshold. The method returns a filtered DataFrame
        containing only the rows that meet all specified criteria.

        Args:
            df (pd.DataFrame): The DataFrame to be filtered.
            global_filter (dict?): A dictionary of filtering criteria where keys are column names
                and values are lists containing a threshold and a condition.
                Defaults to a predefined set of filters.

        Returns:
            pd.DataFrame: A DataFrame containing only the rows that satisfy all filter conditions.
        """

        variables = list(global_filter.keys())
        temp = pd.Series(True, index=df.index)
        for k, v in global_filter.items():
            if v[1] == "s":
                temp = temp & (df[k] <= v[0])
            else:
                temp = temp & (df[k] >= v[0])
        return df[temp]

    def combine_results_to_single_df(self, sweep_id_confirm: list = [], sweep_id_training: str = None):
        """Combine multiple sweep results into a single DataFrame.

        This method retrieves sweep results based on the provided sweep IDs. It
        first checks for an existing CSV file containing results; if it exists,
        it loads the data from that file. If the file does not exist, it
        downloads the results for the training sweep ID and saves them to the
        CSV file. The method then adds a date column to the initial DataFrame
        and iterates through the list of confirmed sweep IDs, downloading and
        appending each corresponding result to the DataFrame. Finally, it
        returns a single DataFrame containing all combined results.

        Args:
            sweep_id_confirm (list): A list of confirmed sweep IDs to retrieve results for.
            sweep_id_training (str): The training sweep ID to download initial results.

        Returns:
            pandas.DataFrame: A DataFrame containing combined results from all specified sweeps.
        """

        existing_file_name = '/home/kpap/Downloads/results_temp_ethusd_1.csv'
        if os.path.exists(existing_file_name):
            df_init = pd.read_csv(existing_file_name)
        else:
            df_init = self.download_sweep_results(sweep_id=sweep_id_training)
            try:
                df_init.to_csv(existing_file_name, index=False)
            except:
                pass
        df_init = self.add_date_column_df(df_init)
        # df_1 = self.filter_data(df_init)
        df_1 = df_init

        for id in sweep_id_confirm:
            df = self.download_sweep_results(sweep_id=id)
            df = self.add_date_column_df(df)
            # df = self.filter_data(df)
            df_1 = pd.concat([df_1, df], ignore_index=True)

        return df_1.reset_index(drop=True)

    def combined_results_df(self, df):
        return None

    def xbtusd_params_list(self):
        """Retrieve a list of parameters for the XBT/USD trading strategy.

        This function returns a predefined list of parameter names that are used
        in the XBT/USD trading strategy. These parameters are essential for
        configuring various aspects of the trading algorithm, including window
        sizes and delta spreads.

        Returns:
            list: A list of parameter names relevant to the XBT/USD trading strategy.
        """

        return ['window_size', 'exit_delta_spread', 'entry_delta_spread',
                'band_funding_system', 'window_size2', 'exit_delta_spread2', 'entry_delta_spread2',
                'band_funding_system2', 'funding_window', 'funding_options']

    def ethusd_params_list(self):
        """Retrieve a list of parameters related to ETH/USD trading.

        This function returns a predefined list of parameter names that are
        relevant for configuring trading strategies or algorithms involving the
        ETH/USD trading pair. The parameters include various settings such as
        window sizes, delta spreads, and thresholds that can be adjusted for
        optimal trading performance.

        Returns:
            list: A list of strings representing the parameter names.
        """

        return ['window_size', 'exit_delta_spread', 'entry_delta_spread',
                'current_r', 'high_r', 'quanto_threshold',
                'hours_to_stop', 'ratio_entry_band_mov_ind',
                'rolling_time_window_size', 'band_funding_system', "funding_window", 'max_trade_volume']

    def generic_funding_swap_params_list(self):
        """Retrieve a list of generic funding swap parameters.

        This function returns a predefined list of parameter names related to
        funding swaps. These parameters are used in various financial
        calculations and configurations, providing essential information for
        managing funding swaps effectively.

        Returns:
            list: A list of strings representing the names of generic funding swap
            parameters.
        """

        return ['fastWeightSwap0', 'fastWeightSwap1', 'fastWeightSwap2',
                'slowWeightSwap0', 'slowWeightSwap1', 'slowWeightSwap2',
                'hoursBeforeSwap0', 'hoursBeforeSwap1', 'hoursBeforeSwap2',
                'slow_funding_window', 'funding_periods_lookback']

    def generic_funding_spot_params_list(self):
        """Retrieve a list of generic funding spot parameters.

        This function returns a predefined list of parameter names that are used
        for generic funding spots. These parameters are likely used in financial
        calculations or configurations related to funding spots.

        Returns:
            list: A list of strings representing the generic funding spot parameters.
        """

        return ['fastWeightSpot0', 'fastWeightSpot1', 'fastWeightSpot2',
                'slowWeightSpot0', 'slowWeightSpot1', 'slowWeightSpot2',
                'hoursBeforeSpot0', 'hoursBeforeSpot1', 'hoursBeforeSpot2',
                ]

class AutomateParameterSelectionEthusd(AutomateParameterSelection):

    def __init__(self, project_name: str ='taker_maker_simulations_2023_2') -> None:
        super().__init__(project_name)

    def combined_results_df(self, df):
        """Combine and transform results from a DataFrame.

        This function processes the input DataFrame `df` to create a new
        DataFrame that consolidates various financial metrics based on specific
        parameters. It checks for the presence of certain columns in the input
        DataFrame and adjusts the parameters accordingly. The function
        identifies duplicate rows based on the specified parameters and
        organizes the data into a new format that is easier to analyze. The
        resulting DataFrame includes metrics such as estimated profit and loss,
        Sharpe ratio, and other relevant financial indicators.

        Args:
            df (pd.DataFrame): The input DataFrame containing financial data.

        Returns:
            pd.DataFrame: A new DataFrame with combined and organized results.
        """

        param_list = self.ethusd_params_list()

        if "num_of_points_to_lookback_entry" in df.columns :
            param_list = param_list + ["num_of_points_to_lookback_entry"]

        if "num_of_points_to_lookback_exit" in df.columns:
            param_list = param_list + ["num_of_points_to_lookback_exit"]

        df = df[['Date', 'link', 'Estimated PNL with Quanto_profit', 'Funding in Total', 'Quanto Profit',
                 'Sharpe Ratio',
                 "Average_Distance", "max_drawdown_d", "max_drawup_d", "Std_daily_ROR", 'ROR Annualized'] + param_list]
        mask = df[param_list].duplicated(keep=False) == True

        df_duplicate_rows = df[mask].sort_values(by=param_list + ['Date'])
        df_duplicate_rows.reset_index(drop=True, inplace=True)

        col_names_list = []
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"ROR {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Sharpe {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"PNL {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"QL {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"FUND {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"LINK {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Average_Distance {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawdown_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawup_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Std_daily_ROR {date}")


        col_names_list = col_names_list + param_list
        df_final = pd.DataFrame(columns=col_names_list, index=range(10000))
        ix = 0
        for idx in df_duplicate_rows.index[1:]:
            if all(df_duplicate_rows[param_list].iloc[idx - 1] ==
                   df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx - 1, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx - 1, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']

            elif any(df_duplicate_rows[param_list].iloc[idx - 1] !=
                     df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx - 1, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx - 1, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx - 1, 'window_size']
                df_final.loc[ix, 'current_r'] = df_duplicate_rows.loc[idx - 1, 'current_r']
                df_final.loc[ix, 'high_r'] = df_duplicate_rows.loc[idx - 1, 'high_r']
                df_final.loc[ix, 'quanto_threshold'] = df_duplicate_rows.loc[idx - 1, 'quanto_threshold']
                df_final.loc[ix, 'hours_to_stop'] = df_duplicate_rows.loc[idx - 1, 'hours_to_stop']
                df_final.loc[ix, 'ratio_entry_band_mov_ind'] = df_duplicate_rows.loc[
                    idx - 1, 'ratio_entry_band_mov_ind']
                df_final.loc[ix, 'rolling_time_window_size'] = df_duplicate_rows.loc[
                    idx - 1, 'rolling_time_window_size']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx - 1, 'band_funding_system']
                df_final.loc[ix, 'funding_window'] = df_duplicate_rows.loc[idx - 1, 'funding_window']
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_exit']

                ix += 1
            if idx == df_duplicate_rows.index[-1]:
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[idx, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx, 'window_size']
                df_final.loc[ix, 'current_r'] = df_duplicate_rows.loc[idx, 'current_r']
                df_final.loc[ix, 'high_r'] = df_duplicate_rows.loc[idx, 'high_r']
                df_final.loc[ix, 'quanto_threshold'] = df_duplicate_rows.loc[idx, 'quanto_threshold']
                df_final.loc[ix, 'hours_to_stop'] = df_duplicate_rows.loc[idx, 'hours_to_stop']
                df_final.loc[ix, 'ratio_entry_band_mov_ind'] = df_duplicate_rows.loc[idx, 'ratio_entry_band_mov_ind']
                df_final.loc[ix, 'rolling_time_window_size'] = df_duplicate_rows.loc[idx, 'rolling_time_window_size']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx, 'band_funding_system']
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_exit']
        return df_final.dropna(how='all')


class AutomateParameterSelectionXbtusd(AutomateParameterSelection):

    def __init__(self, project_name: str ='taker_maker_simulations_2023_2') -> None:
        super().__init__(project_name)

    def combined_results_df(self, df):
        """Combine and process results from a DataFrame.

        This method takes a DataFrame containing various financial metrics and
        processes it to create a new DataFrame that consolidates results based
        on specific parameters. It checks for the presence of certain columns in
        the input DataFrame and adjusts the parameters accordingly. The function
        identifies duplicate rows based on the specified parameters and
        organizes the data into a final DataFrame with relevant metrics for each
        unique date.

        Args:
            df (pd.DataFrame): A DataFrame containing financial data with columns such as 'Date',
                'Estimated PNL with Funding', 'Funding in Total', 'Sharpe Ratio', and
                other
                relevant metrics.

        Returns:
            pd.DataFrame: A new DataFrame containing combined results, with columns for each
                unique date and relevant metrics.
        """

        param_list = self.xbtusd_params_list()

        additional_swap_params = self.generic_funding_swap_params_list()
        additional_spot_params = self.generic_funding_swap_params_list()

        if "num_of_points_to_lookback_entry" in df.columns:
            param_list = param_list + ["num_of_points_to_lookback_entry"]

        if "num_of_points_to_lookback_exit" in df.columns:
            param_list = param_list + ["num_of_points_to_lookback_exit"]

        if 'hoursBeforeSwap0' in df.columns:
            param_list = param_list + additional_swap_params

        df = df[['Date', 'link', 'Estimated PNL with Funding', 'Funding in Total', 'Sharpe Ratio',
                 "Average_Distance", "max_drawdown_d", "max_drawup_d", "Std_daily_ROR", 'ROR Annualized'] + param_list]


        mask = df[param_list].duplicated(keep=False) == True
        df_duplicate_rows = df[mask].sort_values(by=param_list + ['Date'])

        df_duplicate_rows.reset_index(drop=True, inplace=True)
        col_names_list = []
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"ROR {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Sharpe {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"PNL {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"FUND {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"LINK {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Average_Distance {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawdown_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawup_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Std_daily_ROR {date}")



        col_names_list = col_names_list + param_list
        df_final = pd.DataFrame(columns=col_names_list, index=range(10000))

        ix = 0
        for idx in df_duplicate_rows.index[1:]:
            if all(df_duplicate_rows[param_list].iloc[idx - 1] ==
                   df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx - 1, 'Estimated PNL with Funding']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']

            elif any(df_duplicate_rows[param_list].iloc[idx - 1] !=
                     df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx - 1, 'Estimated PNL with Funding']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx - 1, 'window_size']
                df_final.loc[ix, 'entry_delta_spread2'] = df_duplicate_rows.loc[idx - 1, 'entry_delta_spread2']
                df_final.loc[ix, 'exit_delta_spread2'] = df_duplicate_rows.loc[idx - 1, 'exit_delta_spread2']
                df_final.loc[ix, 'window_size2'] = df_duplicate_rows.loc[idx - 1, 'window_size2']
                df_final.loc[ix, 'funding_window'] = df_duplicate_rows.loc[idx - 1, 'funding_window']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx - 1, 'band_funding_system']
                df_final.loc[ix, 'band_funding_system2'] = df_duplicate_rows.loc[idx - 1, 'band_funding_system2']
                df_final.loc[ix, 'funding_options'] = df_duplicate_rows.loc[idx-1, 'funding_options']
                if 'hoursBeforeSwap0' in df.columns:
                    df_final.loc[ix, additional_swap_params] = df_duplicate_rows.loc[idx - 1, additional_swap_params]
                if 'hoursBeforeSpot0' in df.columns:
                    df_final.loc[ix, additional_spot_params] = df_duplicate_rows.loc[idx - 1, additional_spot_params]
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_exit']

                ix += 1
            if idx == df_duplicate_rows.index[-1]:
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx, 'Estimated PNL with Funding']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx, 'ROR Annualized']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[idx, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx, 'window_size']
                df_final.loc[ix, 'entry_delta_spread2'] = df_duplicate_rows.loc[idx, 'entry_delta_spread2']
                df_final.loc[ix, 'exit_delta_spread2'] = df_duplicate_rows.loc[idx, 'exit_delta_spread2']
                df_final.loc[ix, 'window_size2'] = df_duplicate_rows.loc[idx, 'window_size2']
                df_final.loc[ix, 'funding_window'] = df_duplicate_rows.loc[idx, 'funding_window']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx, 'band_funding_system']
                df_final.loc[ix, 'band_funding_system2'] = df_duplicate_rows.loc[idx, 'band_funding_system2']
                df_final.loc[ix, 'funding_options'] = df_duplicate_rows.loc[idx, 'funding_options']
                if 'hoursBeforeSwap0' in df.columns:
                    df_final.loc[ix, additional_swap_params] = df_duplicate_rows.loc[idx, additional_swap_params]
                if 'hoursBeforeSpot0' in df.columns:
                    df_final.loc[ix, additional_spot_params] = df_duplicate_rows.loc[idx, additional_spot_params]
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_exit']
        if 'use_same_values_generic_funding' in df.columns:
            if df['use_same_values_generic_funding'].iloc[0] in [True, 'True', 'true', 'TRUE']:
                df_final[additional_spot_params] = df_final[additional_swap_params]
        return df_final.dropna(how='all')

def is_int(k):
    """Check if a value can be converted to an integer.

    This function attempts to convert the given value to an integer. If the
    conversion is successful, it returns True. If the conversion fails due
    to a ValueError or TypeError, it catches the exception and returns
    False.

    Args:
        k: The value to be checked for integer conversion.

    Returns:
        bool: True if the value can be converted to an integer, False otherwise.
    """

    try:
        int(k)
        return True
    except:
        return False

class AutomateParameterSelectionEthusdMultiperiod(AutomateParameterSelection):

    columns_ordering = ["ROR", "Sharpe", "max_drawdown_d", "link","LINK"]
    def __init__(self, project_name: str ='taker_maker_simulations_2023_2') -> None:
        self.project_name = project_name

    def combine_results_to_single_df(self, sweep_id_confirm: list = []):
        """Combine multiple sweep results into a single DataFrame.

        This method takes a list of sweep IDs, downloads the corresponding sweep
        results for each ID, adds a date column to each DataFrame, and
        concatenates them into a single DataFrame. The resulting DataFrame is
        reset to have a clean index. This is useful for aggregating results from
        multiple sweeps into a unified format for further analysis or reporting.

        Args:
            sweep_id_confirm (list): A list of sweep IDs to download results for.

        Returns:
            pandas.DataFrame: A single DataFrame containing the combined results
            from all specified sweep IDs.
        """

        df_1 = pd.DataFrame()
        for id in sweep_id_confirm:
            df = self.download_sweep_results(sweep_id=id)
            df = self.add_date_column_df(df)
            # df = self.filter_data(df)
            df_1 = pd.concat([df_1, df], ignore_index=True)

        return df_1.reset_index(drop=True)

    def combined_results_df(self, df):
        """Combine and process results from a DataFrame.

        This function takes a DataFrame containing financial data and processes
        it to create a new DataFrame that summarizes various metrics such as
        Estimated PNL, Sharpe Ratio, and others. It handles duplicate entries
        based on specified parameters and organizes the results into a
        structured format. The function also dynamically adjusts the columns
        based on the presence of specific parameters in the input DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame containing financial data with columns such as 'Date',
                'Estimated PNL with Quanto_profit', 'Sharpe Ratio', and others.

        Returns:
            pd.DataFrame: A new DataFrame containing combined results with metrics organized by
                date and additional parameters if present.
        """

        param_list = self.ethusd_params_list()

        if "num_of_points_to_lookback_entry" in df.columns:
            param_list = param_list + ["num_of_points_to_lookback_entry"]

        if "num_of_points_to_lookback_exit" in df.columns:
            param_list = param_list + ["num_of_points_to_lookback_exit"]

        df = df[['Date', 'link', 'Estimated PNL with Quanto_profit', 'Funding in Total', 'Quanto Profit',
                 'Sharpe Ratio',"Average_Distance", "max_drawdown_d", "max_drawup_d", "Std_daily_ROR",
                 'ROR Annualized'+ param_list]]

        mask = df[param_list].duplicated(keep=False) == True

        df_duplicate_rows = df[mask].sort_values(by=param_list + ['Date'])
        df_duplicate_rows.reset_index(drop=True, inplace=True)

        col_names_list = []
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"ROR {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Sharpe {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"PNL {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"QL {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"FUND {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"LINK {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Average_Distance {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawdown_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"max_drawup_d {date}")
        for num, date in enumerate(df_duplicate_rows['Date'].unique()):
            col_names_list.append(f"Std_daily_ROR {date}")


        col_names_list = col_names_list + param_list
        df_final = pd.DataFrame(columns=col_names_list, index=range(10000))
        ix = 0
        for idx in df_duplicate_rows.index[1:]:
            if all(df_duplicate_rows[param_list].iloc[idx - 1] ==
                   df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx - 1, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']

            elif any(df_duplicate_rows[param_list].iloc[idx - 1] !=
                     df_duplicate_rows[param_list].iloc[idx]):
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx - 1, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx - 1, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx - 1, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx - 1, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[
                            idx - 1, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx - 1, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx - 1, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx - 1, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx - 1, 'window_size']
                df_final.loc[ix, 'current_r'] = df_duplicate_rows.loc[idx - 1, 'current_r']
                df_final.loc[ix, 'high_r'] = df_duplicate_rows.loc[idx - 1, 'high_r']
                df_final.loc[ix, 'quanto_threshold'] = df_duplicate_rows.loc[idx - 1, 'quanto_threshold']
                df_final.loc[ix, 'hours_to_stop'] = df_duplicate_rows.loc[idx - 1, 'hours_to_stop']
                df_final.loc[ix, 'ratio_entry_band_mov_ind'] = df_duplicate_rows.loc[
                    idx - 1, 'ratio_entry_band_mov_ind']
                df_final.loc[ix, 'rolling_time_window_size'] = df_duplicate_rows.loc[
                    idx - 1, 'rolling_time_window_size']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx - 1, 'band_funding_system']
                df_final.loc[ix, 'funding_window'] = df_duplicate_rows.loc[idx - 1, 'funding_window']
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx - 1, 'num_of_points_to_lookback_exit']

                ix += 1
            if idx == df_duplicate_rows.index[-1]:
                for num, date in enumerate(df_duplicate_rows['Date'].unique()):
                    if df_duplicate_rows.loc[idx - 1, 'Date'] == date:
                        df_final.loc[ix, f'PNL {date}'] = df_duplicate_rows.loc[idx, 'Estimated PNL with Quanto_profit']
                        df_final.loc[ix, f'Sharpe {date}'] = df_duplicate_rows.loc[idx, 'Sharpe Ratio']
                        df_final.loc[ix, f'ROR {date}'] = df_duplicate_rows.loc[idx, 'ROR Annualized']
                        df_final.loc[ix, f'QL {date}'] = df_duplicate_rows.loc[idx, 'Quanto Profit']
                        df_final.loc[ix, f'FUND {date}'] = df_duplicate_rows.loc[idx, 'Funding in Total']
                        df_final.loc[ix, f'LINK {date}'] = df_duplicate_rows.loc[idx, 'link']
                        df_final.loc[ix, f'Average_Distance {date}'] = df_duplicate_rows.loc[idx, 'Average_Distance']
                        df_final.loc[ix, f'max_drawdown_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawdown_d']
                        df_final.loc[ix, f'max_drawup_d {date}'] = df_duplicate_rows.loc[idx, 'max_drawup_d']
                        df_final.loc[ix, f'Std_daily_ROR {date}'] = df_duplicate_rows.loc[idx, 'Std_daily_ROR']
                df_final.loc[ix, 'entry_delta_spread'] = df_duplicate_rows.loc[idx, 'entry_delta_spread']
                df_final.loc[ix, 'exit_delta_spread'] = df_duplicate_rows.loc[idx, 'exit_delta_spread']
                df_final.loc[ix, 'window_size'] = df_duplicate_rows.loc[idx, 'window_size']
                df_final.loc[ix, 'current_r'] = df_duplicate_rows.loc[idx, 'current_r']
                df_final.loc[ix, 'high_r'] = df_duplicate_rows.loc[idx, 'high_r']
                df_final.loc[ix, 'quanto_threshold'] = df_duplicate_rows.loc[idx, 'quanto_threshold']
                df_final.loc[ix, 'hours_to_stop'] = df_duplicate_rows.loc[idx, 'hours_to_stop']
                df_final.loc[ix, 'ratio_entry_band_mov_ind'] = df_duplicate_rows.loc[idx, 'ratio_entry_band_mov_ind']
                df_final.loc[ix, 'rolling_time_window_size'] = df_duplicate_rows.loc[idx, 'rolling_time_window_size']
                df_final.loc[ix, 'band_funding_system'] = df_duplicate_rows.loc[idx, 'band_funding_system']
                df_final.loc[ix, 'funding_window'] = df_duplicate_rows.loc[idx, 'funding_window']
                if "num_of_points_to_lookback_entry" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_entry'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_entry']
                if "num_of_points_to_lookback_exit" in param_list:
                    df_final.loc[ix, 'num_of_points_to_lookback_exit'] = df_duplicate_rows.loc[
                        idx, 'num_of_points_to_lookback_exit']
        return df_final.dropna(how='all')

    def combined_results_df_multi(self, df_multi, df_conf):
        """Combine multiple DataFrames based on specific parameters.

        This function takes two DataFrames, `df_multi` and `df_conf`, and
        combines them based on matching parameter values. It renames certain
        columns in `df_conf` for clarity and filters the columns from both
        DataFrames to include only those relevant for the analysis. The function
        then iterates through the rows of `df_conf`, finding corresponding rows
        in `df_multi` based on a specified tolerance for parameter differences.
        The results are compiled into a new DataFrame that includes the relevant
        columns from both input DataFrames, while also ensuring that rows with
        all NaN values are removed from the final output.

        Args:
            df_multi (pd.DataFrame): The first DataFrame containing multiple results.
            df_conf (pd.DataFrame): The second DataFrame containing configuration results.

        Returns:
            pd.DataFrame: A new DataFrame containing combined results from both input DataFrames,
                filtered to remove rows where all values are NaN.
        """

        # df_conf = self.filter_data(df_conf, global_filter=None)
        params_col = self.ethusd_params_list()
        results = []
        try:
            df_conf.rename(columns={"ROR Annualized": "ROR Annualized_conf", "Sharpe Ratio": "Sharpe Ratio_conf",
                                   "max_drawdown_d": "max_drawdown_d_conf", "link": "link_conf"}, inplace=True)
        except:
            pass
        temp_cols_multi = list(df_multi.columns)
        temp_cols_conf = [col for col in df_conf.columns if (('ROR' in col) or ('Sharpe' in col) or
                                                             ('max_drawdown_d' in col) or ('LINK' in col))
                          and ('Std' not in col)]

        columns = []
        for col in temp_cols_multi:
            if 'link' in col or "max_drawdown_d" in col or "ROR Annualized" in col or "Sharpe Ratio" in col:
                columns.append(col)
        columns += params_col + ["band_funding_system"]
        result_cols = columns + temp_cols_conf
        for j in range(len(df_conf)):
            row = df_conf.iloc[j]
            index_multi = np.where(np.abs((df_multi[params_col].to_numpy() -
                                           row[params_col].to_numpy()).astype(np.float64).sum(axis=1)) < 0.00000001)[0]
            results.append(df_multi[columns].iloc[int(index_multi)].tolist() + row[temp_cols_conf].to_list())
        df_result = pd.DataFrame(data=results, columns=result_cols)
        columns_result = []
        for col in self.columns_ordering:
            columns_result += [s for s in df_result.columns if col in s]
        for col in df_result.columns:
            if col not in columns_result:
                columns_result.append(col)
        return df_result[columns_result].dropna(how='all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--project_name', default='ethusd_confirmation_11_06_2024', type=str)
    parser.add_argument('--symbol', default='ETHUSD', type=str)
    parser.add_argument('--sweep_id_training', default="0igvi5zr", type=str)
    parser.add_argument('--sweep_id_confirm', default="xbb7ux6s", type=str)
    args = parse_args(parser)
    sweep_id_training = args.sweep_id_training
    try:
        sweep_id_confirm = args.sweep_id_confirm.split(",")
    except:
        pass
    if args.symbol == "ETHUSD_Multiperiod":
        data_processing = AutomateParameterSelectionEthusdMultiperiod(project_name=args.project_name)
        df = data_processing.download_sweep_results(sweep_id_training)
        # df.to_csv('/home/kpap/Downloads/results_temp.csv', index=False)
        # df = pd.read_csv("/home/kpap/Downloads/results_temp.csv")
        # df_conf = data_processing.combine_results_to_single_df(sweep_id_confirm=["s8tkarqx", "lfat6ypc", "jqlz310m"])
        # df_conf = data_processing.combined_results_df(df=df_conf)

        filter1 = {"max_drawdown_d": [1.7, "s"],
                   "max_drawdown_d_p1": [1.7, "s"],
                   "max_drawdown_d_p2": [1.7, "s"],
                   "max_drawdown_d_p3": [1.7, "s"],
                   "max_drawdown_d_p4": [1.7, "s"],
                   "max_drawdown_d_p5": [1.7, "s"],
                   "max_drawdown_d_p6": [1.7, "s"],
                   "ROR Annualized": [25, "l"],
                   "ROR Annualized_p1": [25, "l"],
                   "ROR Annualized_p2": [25, "l"],
                   "ROR Annualized_p3": [25, "l"],
                   "ROR Annualized_p4": [25, "l"],
                   "ROR Annualized_p5": [25, "l"],
                   "ROR Annualized_p6": [25, "l"],
                   "Sharpe Ratio": [3, "l"],
                   "Sharpe Ratio_p1": [3, "l"],
                   "Sharpe Ratio_p2": [3, "l"],
                   "Sharpe Ratio_p3": [3, "l"],
                   "Sharpe Ratio_p4": [3, "l"],
                   "Sharpe Ratio_p5": [3, "l"],
                   "Sharpe Ratio_p6": [3, "l"]
                   }
        filter2 = {"max_drawdown_d": [1.8, "s"],
                   "max_drawdown_d_p1": [1.8, "s"],
                   "max_drawdown_d_p2": [1.8, "s"],
                   "max_drawdown_d_p3": [1.8, "s"],
                   "max_drawdown_d_p4": [1.8, "s"],
                   "max_drawdown_d_p5": [1.8, "s"],
                   "max_drawdown_d_p6": [1.8, "s"],
                   "ROR Annualized": [0, "l"],
                   "ROR Annualized_p1": [0, "l"],
                   "ROR Annualized_p2": [0, "l"],
                   "ROR Annualized_p3": [0, "l"],
                   "ROR Annualized_p4": [0, "l"],
                   "ROR Annualized_p5": [0, "l"],
                   "ROR Annualized_p6": [25, "l"],
                   "Sharpe Ratio": [2, "l"],
                   "Sharpe Ratio_p1": [2, "l"],
                   "Sharpe Ratio_p2": [2, "l"],
                   "Sharpe Ratio_p3": [2, "l"],
                   "Sharpe Ratio_p4": [2, "l"],
                   "Sharpe Ratio_p5": [2, "l"],
                   "Sharpe Ratio_p6": [4, "l"]
                   }

        filtered_df = data_processing.filter_data(df, global_filter=filter2)
        print(len(filtered_df))
        filtered_df.to_csv(f"/home/kpap/Downloads/ethusd_multiperiod_confirmation_{datetime.date.today()}.csv", index=False)
        # data_processing.combined_results_df_multi(df, df_conf).to_csv(f"/home/kpap/Downloads/ethusd_multiperiod_confirmation_{datetime.date.today()}.csv", index=False)
    elif args.symbol == 'XBTUSD':
        data_processing = AutomateParameterSelectionXbtusd(project_name=args.project_name)
        df = data_processing.combine_results_to_single_df(sweep_id_confirm=sweep_id_confirm,
                                                          sweep_id_training=sweep_id_training)
        df1 = data_processing.combined_results_df(df)
        # df1.to_csv(f'/home/enea/Downloads/Results_XBTUSD_{datetime.date.today()}.csv', index=False)
        df1.to_csv(f'/home/kpap/Downloads/Results_XBTUSD_{datetime.date.today()}.csv', index=False)

        df_filtered = df1[abs(df1['window_size'] - df1['window_size2']) <= 400].sort_values(by=['window_size', "window_size2"], ignore_index=True)
        df_filtered.dropna(how='all').to_csv(f'/home/kpap/Downloads/Results_XBTUSD_filtered_{datetime.date.today()}.csv', index=False)
        # a=3
    elif args.symbol == 'ETHUSD':
        data_processing = AutomateParameterSelectionEthusd(project_name=args.project_name)
        df = data_processing.combine_results_to_single_df(sweep_id_confirm=sweep_id_confirm,
                                                          sweep_id_training=sweep_id_training)
        df1 = data_processing.combined_results_df(df)

        df1.to_csv(f'/Users/konstantinospapastamatiou/Downloads/Results_ETHUSD_{datetime.date.today()}.csv', index=False)
    else:
        data_processing = AutomateParameterSelection(project_name=args.project_name)
        df = data_processing.download_sweep_results(sweep_id='8ftsmh3a')
        df.to_csv(f'/home/kpap/Downloads/Deribit_BitMEX_ETHUSD_short_go_long_multiperiod_optimize_latest_pnl_with_drawdown_19_09_to_29_11_2023.csv', index=False)