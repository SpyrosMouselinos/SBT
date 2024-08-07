from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.utils.utils import parse_args, values_to_list
import argparse
import numpy as np

from src.simulations.simulation_codebase.core_code.base_new import TraderExpectedExecutions
from src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function import \
    simulation_trader, upload_to_backblaze, \
    upload_to_wandb
from dotenv import find_dotenv, load_dotenv
import pandas as pd

load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description='')
# time
parser.add_argument('--t_start_period', default=None, type=str)
parser.add_argument('--t_end_period', default=None, type=str)
parser.add_argument('--t_start', default=1715979600000, type=int)
parser.add_argument('--t_end', default=1716325200000, type=int)
# exchanges
parser.add_argument('--exchange_spot', default="Deribit", type=str)
parser.add_argument('--exchange_swap', default="BitMEX", type=str)
parser.add_argument('--swap_instrument', default="BTCUSD", type=str)
parser.add_argument('--spot_instrument', default="BTC-PERPETUAL", type=str)
# band parameters
parser.add_argument('--window_size', default=None, type=int)
parser.add_argument('--entry_delta_spread', default=None, type=float)
parser.add_argument('--exit_delta_spread', default=None, type=float)
parser.add_argument('--area_spread_threshold', default=0.0, type=float)
parser.add_argument('--band_funding_system', default=None, type=str)
parser.add_argument('--funding_window', default=90, type=int)
parser.add_argument('--funding_periods_lookback', default=0, type=int)
parser.add_argument('--slow_funding_window', default=0, type=int)
parser.add_argument('--max_position', default=275000, type=int)
parser.add_argument('--max_trade_volume', default=3000, type=int)
parser.add_argument('--spot_fee', default=None, type=float)
parser.add_argument('--swap_fee', default=None, type=float)
# second band parameters
parser.add_argument('--window_size2', default=None, type=int)
parser.add_argument('--entry_delta_spread2', default=None, type=float)
parser.add_argument('--exit_delta_spread2', default=None, type=float)
parser.add_argument('--band_funding_system2', default=None, type=str)
# funding options (used in quanto only)
parser.add_argument('--funding_options', default=None, type=str)
# depth parameters
parser.add_argument('--constant_depth', default=0, type=float)
parser.add_argument('--swap_market_tick_size', default=0.5, type=float)
# pair name
parser.add_argument("--pair_name", default="BTC-PERPETUAL~XBTUSD")
# new parameters for the ratio in funding
parser.add_argument("--funding_system_name", default="", type=str)

parser.add_argument("--funding_ratios_swap_to_zero_entry", default=None, type=float)
parser.add_argument("--funding_ratios_swap_to_zero_exit", default=None, type=float)
parser.add_argument("--funding_ratios_swap_entry", default=None, type=float)
parser.add_argument("--funding_ratios_swap_exit", default=None, type=float)
parser.add_argument("--funding_ratios_spot_to_zero_entry", default=None, type=float)
parser.add_argument("--funding_ratios_spot_to_zero_exit", default=None, type=float)
parser.add_argument("--funding_ratios_spot_entry", default=None, type=float)
parser.add_argument("--funding_ratios_spot_exit", default=None, type=float)
parser.add_argument("--moving_average_window", default=None, type=int)
# parameters for funding_continuous_weight_concept funding system
parser.add_argument("--use_same_values_generic_funding", default='false', type=str)
parser.add_argument("--use_same_slowSwap_slowSpot_generic_funding", default='false', type=str)

for x in range(5):
    parser.add_argument(f"--hoursBeforeSwap{x}", default=np.nan, type=int)
    parser.add_argument(f"--slowWeightSwap{x}", default=np.nan, type=float)
    parser.add_argument(f"--fastWeightSwap{x}", default=np.nan, type=float)
    parser.add_argument(f"--hoursBeforeSpot{x}", default=np.nan, type=int)
    parser.add_argument(f"--slowWeightSpot{x}", default=np.nan, type=float)
    parser.add_argument(f"--fastWeightSpot{x}", default=np.nan, type=float)

parser.add_argument("--record_rate_limit", default="False", type=str)

parser.add_argument("--fee_addition", default=0.0, type=float)

parser.add_argument("--train_multiple_periods", default="false", type=str)
parser.add_argument("--use_last_two_periods", default="false", type=str)

# paremeters for local minimum and maximum
parser.add_argument('--use_local_min_max', default="False", type=str)
parser.add_argument('--use_same_num_rolling_points', default="False", type=str)
parser.add_argument('--use_extended_filter', default="False", type=str)
parser.add_argument('--num_of_points_to_lookback_entry', default=None, type=int)
parser.add_argument('--num_of_points_to_lookback_exit', default=None, type=int)

parser.add_argument('--filter_only_training', default="False", type=str)

parser.add_argument('--use_same_window_size', default="False", type=str)

parser.add_argument('--adjustment_entry_band', default=0, type=float)
parser.add_argument('--adjustment_exit_band', default=0, type=float)

parser.add_argument('--adjust_pnl_automatically', default="True", type=str)
parser.add_argument('--maximum_quality', default=1000000, type=float)

params = vars(parse_args(parser))

# t_end = params['t_start'] + 1000 * 60 * 60 * 24 * 7 # 27-08-2022 00:00:00 UTC
family = "deribit_xbtusd"
environment = 'staging'

# time variables
params['ratio_entry_band_mov'] = 0.0
params['stop_trading'] = False

# fees
swap_fee, spot_fee = exchange_fees(params['exchange_swap'], params['swap_instrument'],
                                   params['exchange_spot'], params['spot_instrument'])
if params['spot_fee'] is None:
    params['spot_fee'] = spot_fee
if params['swap_fee'] is None:
    params['swap_fee'] = swap_fee

params['spot_fee'] = params['spot_fee'] + params['fee_addition']
params['swap_fee'] = params['swap_fee'] + params['fee_addition']

params['band'] = 'bogdan_bands'
params['lookback'] = None
params['recomputation_time'] = None
params['target_percentage_exit'] = None
params['target_percentage_entry'] = None
params['entry_opportunity_source'] = None
params['exit_opportunity_source'] = None
params['generate_percentage_bands'] = False

# family and environment
params['family'] = 'Other'
params['environment'] = environment
params['strategy'] = ''

# latancies
ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(params['exchange_swap'], params['exchange_spot'])
params['latency_spot'] = ws_spot
params['latency_swap'] = ws_swap
params['latency_try_post'] = api_swap
params['latency_cancel'] = api_swap
params['latency_spot_balance'] = api_spot

# trade volume
# params['max_trade_volume'] = max_trade_volume
# params['max_position'] = max_position

params['volatility'] = None

params['ratio_entry_band_mov'] = 1.0
params['stop_trading'] = False

# quanto bands values
params['funding_system'] = 'No'
params['minimum_distance'] = None
params['minimum_value'] = None
params['trailing_value'] = None
params['disable_when_below'] = 0.0

params['force_band_creation'] = True
params['move_bogdan_band'] = 'No'

if params["record_rate_limit"] in ["True", "true", "TRUE"]:
    params["record_rate_limit"] = True
else:
    params["record_rate_limit"] = False

if params['use_local_min_max'] in ["True", "true", "TRUE"]:
    params['use_local_min_max'] = True
else:
    params['use_local_min_max'] = False

if params['use_same_num_rolling_points'] in ["True", "true", "TRUE"]:
    params['use_same_num_rolling_points'] = True
else:
    params['use_same_num_rolling_points'] = False

if params['use_same_num_rolling_points']:
    params['num_of_points_to_lookback_exit'] = params['num_of_points_to_lookback_entry']

# generic funding system input parameters
if params['use_same_values_generic_funding'] in ["True", "true", "TRUE"]:
    params['use_same_values_generic_funding'] = True
else:
    params['use_same_values_generic_funding'] = False

if params['use_extended_filter'] in ["True", "true", "TRUE"]:
    params['use_extended_filter'] = True
else:
    params['use_extended_filter'] = False

if params['use_same_slowSwap_slowSpot_generic_funding'] in ["True", "true", "TRUE"]:
    params['use_same_slowSwap_slowSpot_generic_funding'] = True
else:
    params['use_same_slowSwap_slowSpot_generic_funding'] = False

if params['use_same_slowSwap_slowSpot_generic_funding']:
    for x in range(5):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']
        params[f'slowWeightSpot{x}'] = params[f'slowWeightSwap{x}']
else:
    for x in range(5):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']

if params['use_same_values_generic_funding']:
    for x in range(5):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']
        params[f'slowWeightSpot{x}'] = params[f'slowWeightSwap{x}']
        params[f'fastWeightSpot{x}'] = params[f'fastWeightSwap{x}']
try:
    params = values_to_list(params, ['hoursBeforeSwap', 'slowWeightSwap', 'fastWeightSwap',
                                     'hoursBeforeSpot', 'slowWeightSpot', 'fastWeightSpot'])
except:
    pass

# new multiperiod parameters
if params['use_last_two_periods'] in ['True', 'TRUE', 'true']:
    use_last_two_periods = True
else:
    use_last_two_periods = False

if params['filter_only_training'] in ['True', 'TRUE', 'true']:
    filter_only_training = True
else:
    filter_only_training = False

if params['use_same_window_size'] in ['True', 'TRUE', 'true']:
    params['window_size2'] = params['window_size']

if params['adjust_pnl_automatically'] in ['True', 'TRUE', 'true']:
    params['adjust_pnl_automatically'] = True
else:
    params['adjust_pnl_automatically'] = False


def extended_filter(df, condition):
    """Filter a DataFrame based on specified conditions.

    This function evaluates the 'Estimated PNL' and 'Funding in Total'
    columns of the provided DataFrame. If the condition is True, it checks
    if either 'Estimated PNL' or 'Funding in Total' is less than zero. If
    the condition is False, it only checks if 'Estimated PNL' is less than
    zero. The function returns a boolean result based on these evaluations.

    Args:
        df (pandas.DataFrame): The DataFrame containing the financial data
            with 'Estimated PNL' and 'Funding in Total' columns.
        condition (bool): A boolean value that determines which condition to apply.

    Returns:
        bool: True if the specified conditions are met; otherwise, False.
    """

    if condition:
        return df['Estimated PNL'] < 0 or df['Funding in Total'] < 0
    else:
        return df['Estimated PNL'] < 0


if params['train_multiple_periods'] in ["True", "true", "TRUE"]:
    wandb_to_upload = {}
    t_starts = list(map(int, params['t_start_period'].split('~')))
    t_ends = list(map(int, params['t_end_period'].split('~')))
    pnl_daily_list = []
    ror_total_list = []
    # initialising max drawdown and has_drawdown_triggered variables
    max_allowed_drawdown = 2
    has_drawdown_triggered = False
    sr = 0

    for idx, t_selected in enumerate(zip(t_starts, t_ends)):
        params['t_start'] = t_selected[0]
        params['t_end'] = t_selected[1]
        params['period'] = f'period_{idx}'
        print(f'params={params}')
        (params_df, simulation_describe, df_total, duration_pos_df, file_id, wandb_summary, band_values, model,
         daily_pnl, rate_limit) = simulation_trader(params)
        band_columns = [x for x in list(band_values.columns) if " Band" in x]
        if params_df['funding_system'] in TraderExpectedExecutions.funding_system_list_fun():
            band_columns = ["timems"] + band_columns + ['quanto_profit']
        else:
            band_columns = ["timems"] + band_columns

        if params['swap_instrument'] in ['ETH-PERPETUAL', 'ETHUSD']:
            bands_df = model.df[band_columns].drop_duplicates(subset=band_columns, keep='last')
        else:
            bands_df = band_values[band_columns].drop_duplicates(subset=band_columns, keep='last')
        upload_to_backblaze(params_df, simulation_describe, df_total, duration_pos_df, params, file_id, bands_df,
                            rate_limit)
        pnl_daily_list.append(daily_pnl)
        ror_total_list.append(wandb_summary['ROR Annualized'])
        print(
            f"#############################MAX DRAWDOWN: {wandb_summary['max_drawdown_d']}. Estimated PNL {wandb_summary['Estimated PNL with Funding']}###########################")
        if (wandb_summary['max_drawdown_d'] > max_allowed_drawdown or
            extended_filter(wandb_summary, params['use_extended_filter'])) and not filter_only_training:
            has_drawdown_triggered = True
        if idx == 0:
            sr = wandb_summary['Sharpe Ratio']
            wandb_to_upload = wandb_summary
            wandb_to_upload['PNL Total'] = wandb_summary['Estimated PNL with Funding']
            if params['adjust_pnl_automatically']:
                condition = (wandb_to_upload['Estimated PNL adj'] < 0) or (wandb_to_upload['Funding in Total'] < 0)
            else:
                condition = wandb_to_upload[f'Estimated PNL_p{idx}'] < 0
            if filter_only_training and (wandb_to_upload[f'max_drawdown_d'] > max_allowed_drawdown or condition):
                print("################################ DRAWDOWN TRIGGERED ###########################################")
                wandb_to_upload[f'Sharpe Ratio'] = -10
                wandb_to_upload[f'ROR Annualized'] = -100
                wandb_to_upload[f'ROR Annualized adj'] = -100
                wandb_to_upload[f'Funding in Total'] = - 10000
                wandb_to_upload[f'Estimated PNL'] = - 10000
                wandb_to_upload[f'Estimated PNL with Funding'] = - 10000
                if params['adjust_pnl_automatically']:
                    wandb_to_upload[f'Estimated PNL adj'] = - 10000
                    wandb_to_upload[f'ROR Annualized adj'] = -100
                print(
                    "##################### condition is true: the simulation is stopping ############################")
                break
        else:
            wandb_to_upload[f'date_start_p{idx}'] = wandb_summary['date_start']
            wandb_to_upload[f'date_end_p{idx}'] = wandb_summary['date_end']
            wandb_to_upload[f'Sharpe Ratio_p{idx}'] = wandb_summary['Sharpe Ratio']
            wandb_to_upload[f'Sortino Ratio_p{idx}'] = wandb_summary['Sortino Ratio']
            wandb_to_upload[f'ROR Annualized_p{idx}'] = wandb_summary['ROR Annualized']
            wandb_to_upload[f'ROR Annualized adj_p{idx}'] = wandb_summary['ROR Annualized adj']
            wandb_to_upload[f'link_p{idx}'] = wandb_summary['link']
            wandb_to_upload[f'Funding in Spot Market_p{idx}'] = wandb_summary['Funding in Spot Market']
            wandb_to_upload[f'Funding in Swap Market_p{idx}'] = wandb_summary['Funding in Swap Market']
            wandb_to_upload[f'Funding in Total_p{idx}'] = wandb_summary['Funding in Total']
            wandb_to_upload[f'Estimated PNL real_p{idx}'] = wandb_summary['Estimated PNL']
            wandb_to_upload[f'Estimated PNL_p{idx}'] = wandb_summary['Estimated PNL']
            wandb_to_upload[f'Estimated PNL adj_p{idx}'] = wandb_summary['Estimated PNL adj']
            wandb_to_upload[f'Estimated PNL with Funding_p{idx}'] = wandb_summary['Estimated PNL with Funding']
            wandb_to_upload[f'Coin Volume in entry side_p{idx}'] = wandb_summary['Coin Volume in entry side']
            wandb_to_upload[f'Coin Volume in exit side_p{idx}'] = wandb_summary['Coin Volume in exit side']
            wandb_to_upload['PNL Total'] = wandb_to_upload['PNL Total'] + \
                                           wandb_summary['Estimated PNL with Funding']
            sr = sr + wandb_summary['Sharpe Ratio']
            wandb_to_upload[f'max_drawdown_d_p{idx}'] = wandb_summary['max_drawdown_d']
    pnl_daily = pd.concat(pnl_daily_list)
    if has_drawdown_triggered:
        print(
            "########################################## DRAWDOWN TRIGGERED ################################################")
        if use_last_two_periods:
            wandb_to_upload[f'Sharpe Ratio'] = -10
            wandb_to_upload[f'ROR Annualized'] = -100
            wandb_to_upload[f'ROR Annualized adj'] = -100
            wandb_to_upload[f'Estimated PNL with Funding'] = - 10000
            wandb_to_upload[f'Funding in Total'] = - 10000
            wandb_to_upload['Estimated PNL last two periods'] = -10000
        wandb_to_upload[f'Sharpe Ratio'] = -10
        wandb_to_upload[f'ROR Annualized'] = -100
        wandb_to_upload[f'Funding in Total'] = - 10000
        wandb_to_upload[f'Estimated PNL with Funding'] = - 10000

    upload_to_wandb(params_df, wandb_to_upload)
else:
    if (params['t_start_period'] is not None) or (params['t_end_period'] is not None):
        params['t_start'] = int(params['t_start_period'])
        params['t_end'] = int(params['t_end_period'])
    (params_df, simulation_describe, df_total, duration_pos_df, file_id, wandb_summary, band_values, model,
     daily_pnl, rate_limit) = simulation_trader(params)
    band_columns = [x for x in list(band_values.columns) if " Band" in x]
    band_columns = ["timems"] + band_columns
    if params['swap_instrument'] in ['ETH-PERPETUAL', 'ETHUSD']:
        bands_df = model.df[band_columns].drop_duplicates(subset=band_columns, keep='last')
    else:
        bands_df = band_values[band_columns].drop_duplicates(subset=band_columns, keep='last')
    # print(bands_df)
    upload_to_backblaze(params_df, simulation_describe, df_total, duration_pos_df, params, file_id, bands_df,
                        rate_limit)
    upload_to_wandb(params_df, wandb_summary)
