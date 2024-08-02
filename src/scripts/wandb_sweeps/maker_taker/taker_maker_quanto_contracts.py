import pandas as pd
from src.common.utils.utils import parse_args, values_to_list
import argparse
import numpy as np
from src.simulations.simulation_codebase.core_code.base_new import TraderExpectedExecutions

from src.simulations.simulation_codebase.latencies_fees.latencies_fees import exchange_fees, set_latencies_auto
from src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function import simulation_trader, \
    upload_to_backblaze, \
    upload_to_wandb
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description='')
# define the start and end timestamps of a period seperated by "~"
# First t_start_period value corresponds  to first t_end_period value, second value to second value and so on.


parser.add_argument('--t_start_period', default=None, type=str)
parser.add_argument('--t_end_period', default=None, type=str)
parser.add_argument('--t_start', default=None, type=int)
parser.add_argument('--t_end', default=None, type=int)

# parameters for the strategy
parser.add_argument('--exchange_spot', default='Deribit', type=str)
parser.add_argument('--exchange_swap', default='BitMEX', type=str)
parser.add_argument('--spot_instrument', default='ETH-PERPETUAL', type=str)
parser.add_argument('--swap_instrument', default='ETHUSD', type=str)
parser.add_argument('--spot_fee', default=None, type=float)
parser.add_argument('--swap_fee', default=None, type=float)
parser.add_argument('--max_trade_volume', default=3000, type=int)
parser.add_argument('--max_position', default=275000, type=int)
parser.add_argument('--environment', default='staging', type=str)
parser.add_argument('--family', default='deribit_eth', type=str)
parser.add_argument('--strategy', default='', type=str)

# parameters for band creation
parser.add_argument('--train_multiple_periods', default='true', type=str)
parser.add_argument('--period', default=None, type=str)
parser.add_argument('--window_size', default=None, type=int)
parser.add_argument('--entry_delta_spread', default=None, type=float)
parser.add_argument('--exit_delta_spread', default=None, type=float)
parser.add_argument('--band_funding_system', default=None, type=str)
parser.add_argument('--funding_system', default='Quanto_both', type=str)
parser.add_argument('--funding_window', default=90, type=int)
parser.add_argument('--funding_periods_lookback', default=0, type=int)
parser.add_argument('--slow_funding_window', default=0, type=int)
parser.add_argument('--move_bogdan_band', default='No', type=str)
parser.add_argument('--ratio_entry_band_mov', default=1.0, type=float)
# parser.add_argument('--stop_trading', default=False, type=bool)

# variables to switch from current to high R values
parser.add_argument('--current_r', default=None, type=float)
parser.add_argument('--high_r', default=None, type=float)
parser.add_argument('--quanto_threshold', default=None, type=float)
parser.add_argument('--hours_to_stop', default=None, type=int)
parser.add_argument('--minimum_distance', default=0.4, type=float)

parser.add_argument('--ratio_entry_band_mov_ind', default=0.0, type=float)
parser.add_argument('--rolling_time_window_size', default=0.0, type=int)

# parameters for when the position is long in ETHUSD sort-go-long
parser.add_argument('--ratio_entry_band_mov_long', default=None, type=float)
parser.add_argument('--ratio_exit_band_mov_ind_long', default=None, type=float)
parser.add_argument('--rolling_time_window_size_long', default=None, type=int)

parser.add_argument('--constant_depth', default=0, type=float)
parser.add_argument('--swap_market_tick_size', default=0.05, type=float)
parser.add_argument("--pair_name", default="ETH-PERPETUAL~ETHUSD")

parser.add_argument('--use_last_two_periods', default='False', type=str)

# parameters for funding_continuous_weight_concept funding system
parser.add_argument("--use_same_values_generic_funding", default='true', type=str)
parser.add_argument("--use_same_slowSwap_slowSpot_generic_funding", default='true', type=str)

for x in range(10):
    parser.add_argument(f"--hoursBeforeSwap{x}", default=np.nan, type=int)
    parser.add_argument(f"--slowWeightSwap{x}", default=np.nan, type=float)
    parser.add_argument(f"--fastWeightSwap{x}", default=np.nan, type=float)
    parser.add_argument(f"--hoursBeforeSpot{x}", default=np.nan, type=int)
    parser.add_argument(f"--slowWeightSpot{x}", default=np.nan, type=float)
    parser.add_argument(f"--fastWeightSpot{x}", default=np.nan, type=float)

parser.add_argument('--filter_only_training', default="False", type=str)
parser.add_argument('--adjustment_entry_band', default=None, type=float)
parser.add_argument('--adjustment_exit_band', default=None, type=float)

parser.add_argument('--adjust_pnl_automatically', default="False", type=str)
parser.add_argument('--maximum_quality', default=1000000, type=float)

params = vars(parse_args(parser))

# stop trading
params['stop_trading'] = True
# revert to low R after 8 hours
params['high_to_current'] = True

if params['use_last_two_periods'] in ['True', 'TRUE', 'true']:
    use_last_two_periods = True
else:
    use_last_two_periods = False

# t_end = params['t_start'] + 1000 * 60 * 60 * 24 * 7 # 27-08-2022 00:00:00 UTC

# time variables

# params['t_end'] = t_end

swap_fee, spot_fee = exchange_fees(params['exchange_swap'], params['swap_instrument'], params['exchange_spot'],
                                   params['spot_instrument'])

if params['swap_fee'] is None:
    params['swap_fee'] = swap_fee
if params['spot_fee'] is None:
    params['spot_fee'] = spot_fee

params['band'] = 'bogdan_bands'
params['lookback'] = None
params['recomputation_time'] = None
params['target_percentage_exit'] = None
params['target_percentage_entry'] = None
params['entry_opportunity_source'] = None
params['exit_opportunity_source'] = None
params['generate_percentage_bands'] = False

# spread thresholds
params['area_spread_threshold'] = 0.0

# latancies
ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(params['exchange_swap'], params['exchange_spot'])
params['latency_spot'] = ws_spot
params['latency_swap'] = ws_swap
params['latency_try_post'] = api_swap
params['latency_cancel'] = api_swap
params['latency_spot_balance'] = api_spot

params['volatility'] = None

# quanto bands values
params['minimum_distance'] = 0.4
params['minimum_value'] = None
params['trailing_value'] = None
params['disable_when_below'] = None
params['ratio_exit_band_mov'] = 0.0
# Create a new band every time
params['force_band_creation'] = True

# generic funding system input parameters
if params['use_same_values_generic_funding'] in ["True", "true", "TRUE"]:
    params['use_same_values_generic_funding'] = True
else:
    params['use_same_values_generic_funding'] = False

if params['use_same_slowSwap_slowSpot_generic_funding'] in ["True", "true", "TRUE"]:
    params['use_same_slowSwap_slowSpot_generic_funding'] = True
else:
    params['use_same_slowSwap_slowSpot_generic_funding'] = False

if params['use_same_slowSwap_slowSpot_generic_funding']:
    for x in range(10):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']
        params[f'slowWeightSpot{x}'] = params[f'slowWeightSwap{x}']
else:
    for x in range(10):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']

if params['use_same_values_generic_funding']:
    for x in range(10):
        params[f'hoursBeforeSpot{x}'] = params[f'hoursBeforeSwap{x}']
        params[f'slowWeightSpot{x}'] = params[f'slowWeightSwap{x}']
        params[f'fastWeightSpot{x}'] = params[f'fastWeightSwap{x}']

params = values_to_list(params, ['hoursBeforeSwap', 'slowWeightSwap', 'fastWeightSwap',
                                 'hoursBeforeSpot', 'slowWeightSpot', 'fastWeightSpot'])

if params['filter_only_training'] in ['True', 'TRUE', 'true']:
    filter_only_training = True
else:
    filter_only_training = False

if params['adjust_pnl_automatically'] in ['True', 'TRUE', 'true']:
    params['adjust_pnl_automatically'] = True
else:
    params['adjust_pnl_automatically'] = False

if params['train_multiple_periods'] in ['True', 'TRUE', 'true']:
    wandb_to_upload = {}
    t_starts = list(map(int, params['t_start_period'].split('~')))
    t_ends = list(map(int, params['t_end_period'].split('~')))
    #  t_start and t_end for debugging
    # t_starts = [1660521600000, 1665496000000]
    # t_ends = [1660921600000, 1665896000000]
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
        band_columns = [x for x in list(model.df.columns) if " Band" in x]
        if params_df['funding_system'] in TraderExpectedExecutions.funding_system_list_fun():
            band_columns = ["timems"] + band_columns + ['quanto_profit']
        else:
            band_columns = ["timems"] + band_columns
        bands_df = model.df[band_columns].drop_duplicates(subset=band_columns, keep='last')
        upload_to_backblaze(params_df, simulation_describe, df_total, duration_pos_df, params, file_id, bands_df,
                            rate_limit)
        pnl_daily_list.append(daily_pnl)
        ror_total_list.append(wandb_summary['ROR Annualized'])
        print(
            f"#############################MAX DRAWDOWN: {wandb_summary['max_drawdown_d']}. Estimated PNL {wandb_summary['Estimated PNL with Realized Quanto_profit']}###########################")
        if (wandb_summary['max_drawdown_d'] > max_allowed_drawdown or
            wandb_summary['Estimated PNL with Realized Quanto_profit'] < 0) and not filter_only_training:
            has_drawdown_triggered = True
        if idx == 0:
            sr = wandb_summary['Sharpe Ratio']
            wandb_to_upload = wandb_summary
            wandb_to_upload['PNL Total'] = wandb_summary['Estimated PNL with Realized Quanto_profit']
            if filter_only_training and (wandb_to_upload[f'max_drawdown_d'] > max_allowed_drawdown or
                                         wandb_to_upload[f'Estimated PNL with Realized Quanto_profit'] < 0):
                print(
                    "##################################### DRAWDOWN TRIGGERED ##############################################")
                wandb_to_upload[f'Sharpe Ratio'] = -10
                wandb_to_upload[f'ROR Annualized'] = -100
                wandb_to_upload[f'Estimated PNL with Realized Quanto_profit'] = - 10000
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
            wandb_to_upload[f'Estimated PNL_p{idx}'] = wandb_summary['Estimated PNL']
            try:
                wandb_to_upload[f'Estimated PNL adj_p{idx}'] = wandb_summary['Estimated PNL αdj']
            except:
                wandb_to_upload[f'Estimated PNL adj_p{idx}'] = wandb_summary['Estimated PNL']
            wandb_to_upload[f'Estimated PNL with Funding_p{idx}'] = wandb_summary['Estimated PNL with Funding']
            wandb_to_upload[f'Quanto Profit_p{idx}'] = wandb_summary['Quanto Profit']
            wandb_to_upload[f'Quanto Profit Unrealised_p{idx}'] = wandb_summary['Quanto Profit Unrealised']
            wandb_to_upload[f'Estimated PNL with Quanto_profit_p{idx}'] = wandb_summary[
                'Estimated PNL with Quanto_profit']
            wandb_to_upload[f'Estimated PNL with Realized Quanto_profit_p{idx}'] = \
                wandb_summary['Estimated PNL with Realized Quanto_profit']
            wandb_to_upload[f'Coin Volume in entry side_p{idx}'] = wandb_summary['Coin Volume in entry side']
            wandb_to_upload[f'Coin Volume in exit side_p{idx}'] = wandb_summary['Coin Volume in exit side']
            wandb_to_upload['PNL Total'] = wandb_to_upload['PNL Total'] + \
                                           wandb_summary['Estimated PNL with Realized Quanto_profit']
            sr = sr + wandb_summary['Sharpe Ratio']
            wandb_to_upload[f'max_drawdown_d_p{idx}'] = wandb_summary['max_drawdown_d']

    pnl_daily = pd.concat(pnl_daily_list)
    if has_drawdown_triggered:
        print("########################################## DRAWDOWN TRIGGERED #########################################")
        if use_last_two_periods:
            wandb_to_upload[f'Sharpe Ratio'] = -10
            wandb_to_upload[f'ROR Annualized'] = -100
            wandb_to_upload[f'Estimated PNL with Realized Quanto_profit'] = - 10000
            wandb_to_upload['Estimated PNL last two periods'] = -10000
        wandb_to_upload[f'Sharpe Ratio'] = -10
        wandb_to_upload[f'ROR Annualized'] = -100
        wandb_to_upload[f'Estimated PNL with Realized Quanto_profit'] = - 10000

    upload_to_wandb(params_df, wandb_to_upload)
else:
    (params_df, simulation_describe, df_total, duration_pos_df, file_id, wandb_summary, band_values, model,
     daily_pnl, rate_limit) = simulation_trader(params)
    band_columns = [x for x in list(model.df.columns) if " Band" in x]
    if params_df['funding_system'] in TraderExpectedExecutions.funding_system_list_fun():
        band_columns = ["timems"] + band_columns + ['quanto_profit']
    else:
        band_columns = ["timems"] + band_columns
    bands_df = model.df[band_columns].drop_duplicates(subset=band_columns, keep='last')
    upload_to_backblaze(params_df, simulation_describe, df_total, duration_pos_df, params, file_id, bands_df,
                        rate_limit)
    upload_to_wandb(params_df, wandb_summary)
