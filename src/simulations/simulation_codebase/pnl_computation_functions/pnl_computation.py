import pandas as pd
import numpy as np
import numba


def compute_rolling_pnl(funding, executions, funding_system):
    """
     @brief Compute rolling PNL for each execution. We need to compute the rolling pricelevel and summation of the spreads for each execution in order to make the summation easier.
     @param funding DataFrame with funding information. Must have'side'and'entry'columns
     @param executions DataFrame with executions in the format returned by compute_pro
     @param funding_system
    """
    entry_executions = executions[executions['side'] == 'entry']
    entry_executions.reset_index(drop=True, inplace=True)
    entry_executions['entry_spread_volume'] = entry_executions['executed_spread'] * entry_executions['traded_volume']
    entry_executions['cum_entry_spread_volume'] = entry_executions['entry_spread_volume'].cumsum()
    entry_executions['cum_volume_over_price'] = entry_executions['volume_over_price'].cumsum()
    entry_executions['entry_avg_fixed_spread'] = entry_executions['cum_entry_spread_volume'] / entry_executions[
        'traded_volume'].cumsum()
    exit_executions = executions[executions['side'] == 'exit']
    exit_executions.reset_index(drop=True, inplace=True)
    exit_executions['exit_spread_volume'] = exit_executions['executed_spread'] * exit_executions['traded_volume']
    exit_executions['cum_exit_spread_volume'] = exit_executions['exit_spread_volume'].cumsum()
    exit_executions['cum_volume_over_price'] = exit_executions['volume_over_price'].cumsum()
    exit_executions['exit_avg_fixed_spread'] = exit_executions['cum_exit_spread_volume'] / exit_executions[
        'traded_volume'].cumsum()
    if funding_system == 'No' or funding_system == None:
        entry_executions['quanto_profit'] = 0
        exit_executions['quanto_profit'] = 0

    if funding_system == 'Quanto_both':
        executions_df = pd.merge_ordered(entry_executions[['timems', 'cum_volume_over_price', 'entry_avg_fixed_spread',
                                                           'side', 'trade', 'quanto_profit']],
                                         exit_executions[['timems', 'cum_volume_over_price', 'exit_avg_fixed_spread',
                                                          'side', 'trade', 'quanto_profit']],
                                         on=['timems', 'side', 'trade'], suffixes=['_entry', '_exit'])
    else:
        executions_df = pd.merge_ordered(entry_executions[['timems', 'cum_volume_over_price', 'entry_avg_fixed_spread',
                                                           'side', 'quanto_profit']],
                                         exit_executions[['timems', 'cum_volume_over_price', 'exit_avg_fixed_spread',
                                                          'side', 'quanto_profit']],
                                         on=['timems', 'side'], suffixes=['_entry', '_exit'])

    if funding_system == 'Quanto_loss':
        executions_df['quanto_profit'] = executions_df['quanto_profit_exit'].fillna(0)
    elif funding_system == 'Quanto_profit' or funding_system == 'Quanto_profit_exp':
        executions_df['quanto_profit'] = executions_df['quanto_profit_entry'].fillna(0)
    elif funding_system == 'Quanto_both':
        qp_df = executions.loc[((executions.side == 'entry') & (executions.trade == 'long')) | (
                    (executions.side == 'exit') & (executions.trade == 'short')), ['timems', 'quanto_profit']]
        executions_df = pd.merge_ordered(executions_df, qp_df, on='timems')

    if 'quanto_profit' in executions_df.columns:
        executions_df['quanto_profit'].fillna(0, inplace=True)
    else:
        executions_df['quanto_profit'] = 0

    # executions_df['side'] = executions_df['side_entry'].fillna(executions_df['side_exit'])
    executions_df['cum_quanto_profit'] = executions_df['quanto_profit'].cumsum()
    data_df = pd.merge_ordered(funding[['timems', 'total']], executions_df, on='timems')
    data_df['total'].ffill(inplace=True)
    data_df['total'].fillna(0, inplace=True)
    # data_df.dropna(subset=['side'], inplace=True)
    data_df['cum_volume_over_price_entry'].ffill(inplace=True)
    data_df['cum_volume_over_price_entry'].fillna(0, inplace=True)
    data_df['cum_volume_over_price_exit'].ffill(inplace=True)
    data_df['cum_volume_over_price_exit'].fillna(0, inplace=True)
    data_df['min_volume_over_price'] = data_df[['cum_volume_over_price_entry', 'cum_volume_over_price_exit']].min(
        axis=1)
    data_df['entry_avg_fixed_spread'].ffill(inplace=True)
    data_df['exit_avg_fixed_spread'].ffill(inplace=True)
    data_df['avg_fixed_spread'] = data_df['entry_avg_fixed_spread'].fillna(0) - data_df['exit_avg_fixed_spread'].fillna(
        0)
    data_df['pnl_generated'] = data_df['avg_fixed_spread'] * data_df['min_volume_over_price'] + data_df['total'] + \
                               data_df['cum_quanto_profit']
    data_df['Time'] = pd.to_datetime(data_df['timems'], unit='ms', utc=True)
    data_df['Total_diff'] = data_df['total'].diff()
    data_df['pnl_generated_new'] = 0
    # print('length of data_df', len(data_df))
    # print('compute_rolling_pnl prior to for loop')
    data_array = data_df[['pnl_generated_new', 'pnl_generated', 'Total_diff']].to_numpy()
    # print('data_array shape', data_array.shape)
    data_df['pnl_generated_new'] = pnl_generated_new_column(data_array)
    # print('compute_rolling_pnl end of for loop')
    df = data_df[['Time', 'pnl_generated_new']].drop_duplicates(subset=['Time'], keep='first')
    return df.iloc[:-1]


@numba.jit(nopython=True)
def pnl_generated_new_column(data_array):
    # print('entering pnl_generated_new_column')
    for idx in range(1, len(data_array) - 1):
        if np.isnan(data_array[idx, 1]):
            data_array[idx, 0] = data_array[idx - 1, 0] + data_array[idx, 2]
        else:
            data_array[idx, 0] = data_array[idx, 1]
    # print('exiting pnl_generated_new_column')
    return data_array[:, 0]
