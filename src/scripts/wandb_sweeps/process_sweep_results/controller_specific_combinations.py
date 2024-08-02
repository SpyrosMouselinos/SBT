import argparse
import os
from dotenv import find_dotenv, load_dotenv
import wandb
from human_id import generate_id
from src.common.constants.constants import WANDB_ENTITY, TAKER_MAKER_PROJECT_2023
from src.common.utils.utils import parse_args
from src.scripts.wandb_sweeps.process_sweep_results.mapping_simulations_real_results import \
    mapping_real_params_to_simulation_params, params_for_xbtusd, params_for_ethusd_short_go_long, \
    params_ethusd_combined_results, params_xbtusd_combined_results, params_for_btc_deribit_maker, \
    params_btc_deribit_maker_combined_results
from src.simulations.simulations_management import sweep_rerun_simulations
import pandas as pd

load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sweep_id', default=None, type=str)
parser.add_argument('--source_sweep_id', default=None, type=str)
parser.add_argument('--select_num_simulations', default=10, type=int)
parser.add_argument('--custom_filter', default=None, type=str)
parser.add_argument('--project_name', type=str, default='taker_maker_simulations_2024')
parser.add_argument('--symbol', type=str, default=None)
parser.add_argument('--file_name', type=str, default=None)
parser.add_argument('--band_funding_system', type=str, default=None)
parser.add_argument('--band_funding_system2', type=str, default=None)
params = vars(parse_args(parser))

schedule_list = []


def schedule(controller, run):
    """Schedule a run with the given controller.

    This function initiates a scheduling action by ensuring that the
    controller is started. It generates a unique schedule ID and logs the
    action with the parameters from the run configuration. The schedule
    information is then appended to a global schedule list, which is also
    updated in the controller's internal state.

    Args:
        controller (Controller): The controller instance responsible for managing
            the scheduling process.
        run (Run): An instance containing the configuration for the run to be scheduled.

    Returns:
        None: This function does not return a value.
    """

    global schedule_list
    controller._start_if_not_started()
    schedule_id = generate_id()
    param_list = [
        "%s=%s" % (k, v.get("value")) for k, v in sorted(run.config.items())
    ]
    controller._log_actions.append(("schedule", ",".join(param_list)))
    schedule_list.append({"id": schedule_id, "data": {"args": run.config}})
    controller._controller["schedule"] = schedule_list


class Combination():
    config = {}

    def __init__(self, config, sweep):
        self.config = config
        params = sweep.sweep_config['parameters']
        for key in params.keys():
            param = params[key]
            if param['distribution'] == 'constant':
                self.config[key] = {'value': param['value']}


def start(sweep_id, source_sweep_id, select_num_simulations, custom_filter=params['custom_filter'],
          project_name: str = params['project_name'], symbol: str = params['symbol'],
          file_name: str = params['file_name'],
          band_funding_system: str = params["band_funding_system"],
          band_funding_system2: str = params["band_funding_system"]):
    """Start a sweep for simulations based on specified parameters.

    This function initiates a sweep using the Weights and Biases (wandb)
    library. It connects to the wandb server, retrieves the specified sweep,
    and runs simulations based on the provided filters and parameters.
    Depending on the `custom_filter` value, it can rerun simulations from a
    source sweep or generate combinations of parameters for different
    symbols. The function also handles various cases for production and
    simulation results, ensuring that the appropriate combinations are
    processed.

    Args:
        sweep_id (str): The ID of the sweep to start.
        source_sweep_id (str or None): The ID of the source sweep to rerun simulations from, if applicable.
        select_num_simulations (int): The number of simulations to select from the source sweep.
        custom_filter (str?): A filter to customize the simulation parameters. Defaults to
            params['custom_filter'].
        project_name (str?): The name of the project in wandb. Defaults to params['project_name'].
        symbol (str?): The trading symbol for which to run simulations. Defaults to
            params['symbol'].
        file_name (str?): The name of the file containing relevant data. Defaults to
            params['file_name'].
        band_funding_system (str?): The first band funding system parameter. Defaults to
            params["band_funding_system"].
        band_funding_system2 (str?): The second band funding system parameter. Defaults to
            params["band_funding_system"].
    """

    wandb.login(host=os.getenv("WANDB_HOST"))
    try:
        sweep = wandb.controller(sweep_id, project=project_name, entity=WANDB_ENTITY)
    except:
        sweep = wandb.controller(sweep_id, project=TAKER_MAKER_PROJECT_2023, entity=WANDB_ENTITY)

    if source_sweep_id is not None:
        lst = sweep_rerun_simulations(sweep_id=source_sweep_id,
                                      select_num_simulations=select_num_simulations,
                                      custom_filter=custom_filter, project_name=project_name)
        combinations_loop(param_lst=lst, sweep=sweep)

    elif params['custom_filter'] == 'short':
        combinations_loop(param_lst=combinations2(), sweep=sweep)
    elif params['custom_filter'] == 'production':
        if symbol == "XBTUSD":
            combination_list = xbtusd_production_params(file_name=file_name, band_funding_system=band_funding_system,
                                                        band_funding_system2=band_funding_system2)
        elif symbol == "ETHUSD":
            combination_list = ethusd_production_params(file_name=file_name, band_funding_system=band_funding_system)
        elif symbol == "BTC":
            combination_list = btc_production_params(file_name=file_name)
        combinations_loop(param_lst=combination_list, sweep=sweep)
    elif params['custom_filter'] == 'simulation_results':
        if symbol == "XBTUSD":
            combination_list = xbtusd_confirmation_combined(file_name=file_name)
        elif symbol == "ETHUSD":
            combination_list = ethusd_confirmation_combined(file_name=file_name)
        elif symbol == "BTC":
            combination_list = btc_confirmation_combined(file_name=file_name)
        combinations_loop(param_lst=combination_list, sweep=sweep)
    elif params['custom_filter'] == 'xbtusd_own':
        combinations(param_lst=xbtusd_combinations_own(), sweep=sweep)
    elif params['custom_filter'] == 'xbtusd_nickel':
        combinations_loop(param_lst=xbtusd_combinations_nickel(), sweep=sweep)
    else:
        combinations_loop(param_lst=combinations(), sweep=sweep)
    sweep._sweep_object_sync_to_backend()


def combinations_loop(param_lst=None, sweep=None):
    """Schedule combinations of parameters using a loop.

    This function iterates over a list of parameter combinations, creates a
    Combination object for each combination, and schedules it. After
    scheduling, it prints the status of the sweep and logs the configuration
    of each scheduled combination. Finally, it provides feedback on the
    completion of the selection process and the length of the input list.

    Args:
        param_lst (list): A list of parameter combinations to be scheduled.
        sweep (object): An object responsible for managing the scheduling process.

    Returns:
        None: This function does not return a value.
    """

    for combination in param_lst:
        c = Combination(combination, sweep)
        schedule(sweep, c)
        sweep.print_status()
        print(f"Scheduled {c.config}")
    print('selection is done')
    print(f"length of result list {len(param_lst)}")


def combinations():
    """Generate a list of configuration combinations for a trading strategy.

    This function returns a list of dictionaries, where each dictionary
    represents a unique combination of parameters for a trading strategy.
    The parameters include various metrics such as window size, delta
    spreads, entry and exit ratios, and other relevant settings that
    influence the behavior of the strategy. Each combination is designed to
    explore different scenarios in trading to optimize performance based on
    historical data.

    Returns:
        list: A list of dictionaries, each containing configuration parameters
        for a trading strategy.
    """

    return [
        dict(window_size={"value": 3093}, exit_delta_spread={"value": 1.9973}, entry_delta_spread={"value": 3.6078},
             ratio_entry_band_mov={"value": -1.0981}, minimum_value={"value": 1.1595},
             ratio_exit_band_mov={"value": 1.2942}, rolling_time_window_size={"value": 322},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.9961}),
        dict(window_size={"value": 3694}, exit_delta_spread={"value": 1.5183}, entry_delta_spread={"value": 3.7594},
             ratio_entry_band_mov={"value": -3.2302}, minimum_value={"value": 1.0224},
             ratio_exit_band_mov={"value": 1.0982}, rolling_time_window_size={"value": 249},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.7871}),
        dict(window_size={"value": 3304}, exit_delta_spread={"value": 1.6709}, entry_delta_spread={"value": 4.5993},
             ratio_entry_band_mov={"value": -0.779}, minimum_value={"value": 1.6985},
             ratio_exit_band_mov={"value": 1.1070}, rolling_time_window_size={"value": 250},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.1419}),
        dict(window_size={"value": 2945}, exit_delta_spread={"value": 1.8941}, entry_delta_spread={"value": 2.9565},
             trailing_value={"value": 2.6767}, ratio_exit_band_mov={"value": 1.2987},
             ratio_entry_band_mov={"value": -2.6146}, minimum_value={"value": 1.6414},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 235}),
        dict(window_size={"value": 2649}, exit_delta_spread={"value": 1.9550}, entry_delta_spread={"value": 3.0095},
             trailing_value={"value": 2.3651}, ratio_exit_band_mov={"value": 1.2913},
             ratio_entry_band_mov={"value": -3.6159}, minimum_value={"value": 1.8363},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 244}),
        dict(window_size={"value": 3248}, exit_delta_spread={"value": 1.9377}, entry_delta_spread={"value": 4.4354},
             trailing_value={"value": 2.3495}, ratio_exit_band_mov={"value": 1.3214},
             ratio_entry_band_mov={"value": -3.3787}, minimum_value={"value": 1.6129},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 239}),
        dict(window_size={"value": 2947}, exit_delta_spread={"value": 1.8170}, entry_delta_spread={"value": 4.1308},
             trailing_value={"value": 1.7495}, ratio_exit_band_mov={"value": 1.1514},
             ratio_entry_band_mov={"value": -2.9354}, minimum_value={"value": 1.7052},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 237}),
        dict(window_size={"value": 3093}, exit_delta_spread={"value": 1.9973}, entry_delta_spread={"value": 3.6078},
             ratio_entry_band_mov={"value": -1.0981}, minimum_value={"value": 1.1595},
             ratio_exit_band_mov={"value": 1.2942}, rolling_time_window_size={"value": 322},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.9961}),
        dict(window_size={"value": 3694}, exit_delta_spread={"value": 1.5183}, entry_delta_spread={"value": 3.7594},
             ratio_entry_band_mov={"value": -3.2302}, minimum_value={"value": 1.0224},
             ratio_exit_band_mov={"value": 1.0982}, rolling_time_window_size={"value": 249},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.7871}),
        dict(window_size={"value": 3304}, exit_delta_spread={"value": 1.6709}, entry_delta_spread={"value": 4.5993},
             ratio_entry_band_mov={"value": -0.779}, minimum_value={"value": 1.6985},
             ratio_exit_band_mov={"value": 1.1070}, rolling_time_window_size={"value": 250},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             trailing_value={"value": 1.1419}),
        dict(window_size={"value": 2945}, exit_delta_spread={"value": 1.8941}, entry_delta_spread={"value": 2.9565},
             trailing_value={"value": 2.6767}, ratio_exit_band_mov={"value": 1.2987},
             ratio_entry_band_mov={"value": -2.6146}, minimum_value={"value": 1.6414},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 235}),
        dict(window_size={"value": 2649}, exit_delta_spread={"value": 1.9550}, entry_delta_spread={"value": 3.0095},
             trailing_value={"value": 2.3651}, ratio_exit_band_mov={"value": 1.2913},
             ratio_entry_band_mov={"value": -3.6159}, minimum_value={"value": 1.8363},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 244}),
        dict(window_size={"value": 3248}, exit_delta_spread={"value": 1.9377}, entry_delta_spread={"value": 4.4354},
             trailing_value={"value": 2.3495}, ratio_exit_band_mov={"value": 1.3214},
             ratio_entry_band_mov={"value": -3.3787}, minimum_value={"value": 1.6129},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 239}),
        dict(window_size={"value": 2947}, exit_delta_spread={"value": 1.8170}, entry_delta_spread={"value": 4.1308},
             trailing_value={"value": 1.7495}, ratio_exit_band_mov={"value": 1.1514},
             ratio_entry_band_mov={"value": -2.9354}, minimum_value={"value": 1.7052},
             band_funding_system={"value": 'funding_adjusted_band_swap_spot_with_drop'},
             move_exit_above_entry={"value": 'False'},
             rolling_time_window_size={"value": 237})
    ]


def combinations2():
    """Generate a list of configuration dictionaries for trading parameters.

    This function returns a list containing a single dictionary that holds
    various trading parameters such as window size, delta spreads, current
    and high R values, and other relevant metrics. These parameters can be
    used in trading algorithms or simulations to configure the behavior of
    the trading strategy.

    Returns:
        list: A list containing a dictionary with trading configuration parameters.
    """

    return [
        dict(window_size={"value": 2840}, exit_delta_spread={"value": 0.6353}, entry_delta_spread={"value": 1.46538},
             current_r={"value": 0.88807}, high_r={"value": 2.01372}, ratio_entry_band_mov_ind={"value": 6.8615},
             rolling_time_window_size={"value": 275},
             hours_to_stop={"value": 31}, quanto_threshold={"value": 0.2897},
             band_funding_system={"value": 'funding_adjusted_exit_band_with_drop'})

    ]


def xbtusd_combinations_own():
    """Generate a list of XBT/USD combinations with specific parameters.

    This function creates a list of dictionaries, each representing a unique
    combination of parameters related to XBT/USD trading. Each dictionary
    contains values for window sizes, exit and entry delta spreads, and
    funding options. The combinations are designed for a specific funding
    system scenario, which is consistent across all entries.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
        following keys:
            - window_size (dict): A dictionary with a 'value' key indicating
            the size of the window.
            - exit_delta_spread (dict): A dictionary with a 'value' key
            indicating the exit delta spread.
            - entry_delta_spread (dict): A dictionary with a 'value' key
            indicating the entry delta spread.
            - window_size2 (dict): A second window size dictionary.
            - exit_delta_spread2 (dict): A second exit delta spread dictionary.
            - entry_delta_spread2 (dict): A second entry delta spread dictionary.
            - band_funding_system (dict): A dictionary indicating the funding
            system used.
            - band_funding_system2 (dict): A second funding system dictionary.
            - funding_options (dict): A dictionary indicating the funding option.
    """

    return [
        dict(window_size={"value": 2988}, exit_delta_spread={"value": 1.5194}, entry_delta_spread={"value": 1.8445},
             window_size2={"value": 2988}, exit_delta_spread2={"value": 1.5194}, entry_delta_spread2={"value": 1.8445},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3192}, exit_delta_spread={"value": 1.5139}, entry_delta_spread={"value": 1.7137},
             window_size2={"value": 3192}, exit_delta_spread2={"value": 1.5139}, entry_delta_spread2={"value": 1.7137},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3390}, exit_delta_spread={"value": 1.527}, entry_delta_spread={"value": 1.9051},
             window_size2={"value": 3390}, exit_delta_spread2={"value": 1.527}, entry_delta_spread2={"value": 1.9051},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3568}, exit_delta_spread={"value": 1.5628}, entry_delta_spread={"value": 1.8217},
             window_size2={"value": 3568}, exit_delta_spread2={"value": 1.5628}, entry_delta_spread2={"value": 1.8217},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3780}, exit_delta_spread={"value": 1.5264}, entry_delta_spread={"value": 1.7884},
             window_size2={"value": 3780}, exit_delta_spread2={"value": 1.5264}, entry_delta_spread2={"value": 1.7884},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3997}, exit_delta_spread={"value": 1.5158}, entry_delta_spread={"value": 2.0108},
             window_size2={"value": 3997}, exit_delta_spread2={"value": 1.5158}, entry_delta_spread2={"value": 2.0108},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 2984}, exit_delta_spread={"value": 1.5029}, entry_delta_spread={"value": 1.9319},
             window_size2={"value": 2984}, exit_delta_spread2={"value": 1.5029}, entry_delta_spread2={"value": 1.9319},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3183}, exit_delta_spread={"value": 1.5361}, entry_delta_spread={"value": 1.7523},
             window_size2={"value": 3183}, exit_delta_spread2={"value": 1.5361}, entry_delta_spread2={"value": 1.7523},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3383}, exit_delta_spread={"value": 1.5064}, entry_delta_spread={"value": 1.7902},
             window_size2={"value": 3383}, exit_delta_spread2={"value": 1.5064}, entry_delta_spread2={"value": 1.7902},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3544}, exit_delta_spread={"value": 1.5426}, entry_delta_spread={"value": 1.8392},
             window_size2={"value": 3544}, exit_delta_spread2={"value": 1.5426}, entry_delta_spread2={"value": 1.8392},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3770}, exit_delta_spread={"value": 1.5057}, entry_delta_spread={"value": 1.8392},
             window_size2={"value": 3770}, exit_delta_spread2={"value": 1.5057}, entry_delta_spread2={"value": 1.8392},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3991}, exit_delta_spread={"value": 1.542}, entry_delta_spread={"value": 1.8739},
             window_size2={"value": 3991}, exit_delta_spread2={"value": 1.542}, entry_delta_spread2={"value": 1.8739},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 2983}, exit_delta_spread={"value": 1.5278}, entry_delta_spread={"value": 1.8589},
             window_size2={"value": 2983}, exit_delta_spread2={"value": 1.5278}, entry_delta_spread2={"value": 1.8589},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3161}, exit_delta_spread={"value": 1.5017}, entry_delta_spread={"value": 1.811},
             window_size2={"value": 3161}, exit_delta_spread2={"value": 1.5017}, entry_delta_spread2={"value": 1.811},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3380}, exit_delta_spread={"value": 1.5084}, entry_delta_spread={"value": 1.9174},
             window_size2={"value": 3380}, exit_delta_spread2={"value": 1.5084}, entry_delta_spread2={"value": 1.9174},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3519}, exit_delta_spread={"value": 1.5037}, entry_delta_spread={"value": 1.7757},
             window_size2={"value": 3519}, exit_delta_spread2={"value": 1.5037}, entry_delta_spread2={"value": 1.7757},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3755}, exit_delta_spread={"value": 1.5049}, entry_delta_spread={"value": 1.798},
             window_size2={"value": 3755}, exit_delta_spread2={"value": 1.5049}, entry_delta_spread2={"value": 1.798},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3990}, exit_delta_spread={"value": 1.6427}, entry_delta_spread={"value": 1.8984},
             window_size2={"value": 3990}, exit_delta_spread2={"value": 1.6427}, entry_delta_spread2={"value": 1.8984},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 2982}, exit_delta_spread={"value": 1.5367}, entry_delta_spread={"value": 1.8482},
             window_size2={"value": 2982}, exit_delta_spread2={"value": 1.5367}, entry_delta_spread2={"value": 1.8482},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3136}, exit_delta_spread={"value": 1.5032}, entry_delta_spread={"value": 1.7846},
             window_size2={"value": 3136}, exit_delta_spread2={"value": 1.5032}, entry_delta_spread2={"value": 1.7846},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3377}, exit_delta_spread={"value": 1.5407}, entry_delta_spread={"value": 1.9142},
             window_size2={"value": 3377}, exit_delta_spread2={"value": 1.5407}, entry_delta_spread2={"value": 1.9142},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3507}, exit_delta_spread={"value": 1.5451}, entry_delta_spread={"value": 1.7336},
             window_size2={"value": 3507}, exit_delta_spread2={"value": 1.5451}, entry_delta_spread2={"value": 1.7336},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3711}, exit_delta_spread={"value": 1.506}, entry_delta_spread={"value": 1.7103},
             window_size2={"value": 3711}, exit_delta_spread2={"value": 1.506}, entry_delta_spread2={"value": 1.7103},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3987}, exit_delta_spread={"value": 1.5022}, entry_delta_spread={"value": 1.8096},
             window_size2={"value": 3987}, exit_delta_spread2={"value": 1.5022}, entry_delta_spread2={"value": 1.8096},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 2980}, exit_delta_spread={"value": 1.5235}, entry_delta_spread={"value": 1.6268},
             window_size2={"value": 2980}, exit_delta_spread2={"value": 1.5235}, entry_delta_spread2={"value": 1.6268},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3116}, exit_delta_spread={"value": 1.5109}, entry_delta_spread={"value": 1.8118},
             window_size2={"value": 3116}, exit_delta_spread2={"value": 1.5109}, entry_delta_spread2={"value": 1.8118},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3370}, exit_delta_spread={"value": 1.5329}, entry_delta_spread={"value": 1.8527},
             window_size2={"value": 3370}, exit_delta_spread2={"value": 1.5329}, entry_delta_spread2={"value": 1.8527},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3472}, exit_delta_spread={"value": 1.5301}, entry_delta_spread={"value": 1.821},
             window_size2={"value": 3472}, exit_delta_spread2={"value": 1.5301}, entry_delta_spread2={"value": 1.821},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3679}, exit_delta_spread={"value": 1.5185}, entry_delta_spread={"value": 1.9123},
             window_size2={"value": 3679}, exit_delta_spread2={"value": 1.5185}, entry_delta_spread2={"value": 1.9123},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3986}, exit_delta_spread={"value": 1.5348}, entry_delta_spread={"value": 1.8264},
             window_size2={"value": 3986}, exit_delta_spread2={"value": 1.5348}, entry_delta_spread2={"value": 1.8264},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 2975}, exit_delta_spread={"value": 1.5538}, entry_delta_spread={"value": 1.6933},
             window_size2={"value": 2975}, exit_delta_spread2={"value": 1.5538}, entry_delta_spread2={"value": 1.6933},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3107}, exit_delta_spread={"value": 1.5162}, entry_delta_spread={"value": 1.7583},
             window_size2={"value": 3107}, exit_delta_spread2={"value": 1.5162}, entry_delta_spread2={"value": 1.7583},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3344}, exit_delta_spread={"value": 1.5018}, entry_delta_spread={"value": 1.862},
             window_size2={"value": 3344}, exit_delta_spread2={"value": 1.5018}, entry_delta_spread2={"value": 1.862},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3455}, exit_delta_spread={"value": 1.5365}, entry_delta_spread={"value": 1.8615},
             window_size2={"value": 3455}, exit_delta_spread2={"value": 1.5365}, entry_delta_spread2={"value": 1.8615},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3667}, exit_delta_spread={"value": 1.5534}, entry_delta_spread={"value": 1.8783},
             window_size2={"value": 3667}, exit_delta_spread2={"value": 1.5534}, entry_delta_spread2={"value": 1.8783},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}),
        dict(window_size={"value": 3979}, exit_delta_spread={"value": 1.6089}, entry_delta_spread={"value": 1.8319},
             window_size2={"value": 3979}, exit_delta_spread2={"value": 1.6089}, entry_delta_spread2={"value": 1.8319},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'})
    ]


def xbtusd_combinations_nickel():
    """Generate a list of trading combinations for XBT/USD.

    This function creates a comprehensive list of dictionaries, where each
    dictionary represents a unique combination of trading parameters for the
    XBT/USD trading pair. The parameters include various window sizes, delta
    spreads for both entry and exit, funding options, and limits on maximum
    position and trade volume. These combinations are essential for traders
    looking to simulate or analyze different trading strategies in the
    XBT/USD market.

    Returns:
        list: A list of dictionaries, each containing trading parameters for
        XBT/USD combinations.
    """

    return [
        dict(window_size={"value": 2969}, exit_delta_spread={"value": 1.5617}, entry_delta_spread={"value": 1.8762},
             window_size2={"value": 2969}, exit_delta_spread2={"value": 1.5617}, entry_delta_spread2={"value": 1.8762},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3090}, exit_delta_spread={"value": 1.51}, entry_delta_spread={"value": 1.8492},
             window_size2={"value": 3090}, exit_delta_spread2={"value": 1.51}, entry_delta_spread2={"value": 1.8492},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3307}, exit_delta_spread={"value": 1.5196}, entry_delta_spread={"value": 1.8492},
             window_size2={"value": 3307}, exit_delta_spread2={"value": 1.5196}, entry_delta_spread2={"value": 1.8492},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3431}, exit_delta_spread={"value": 1.5226}, entry_delta_spread={"value": 1.8489},
             window_size2={"value": 3431}, exit_delta_spread2={"value": 1.5226}, entry_delta_spread2={"value": 1.8489},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3641}, exit_delta_spread={"value": 1.5013}, entry_delta_spread={"value": 1.8663},
             window_size2={"value": 3641}, exit_delta_spread2={"value": 1.5013}, entry_delta_spread2={"value": 1.8663},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3853}, exit_delta_spread={"value": 1.5731}, entry_delta_spread={"value": 1.9328},
             window_size2={"value": 3853}, exit_delta_spread2={"value": 1.5731}, entry_delta_spread2={"value": 1.9328},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 2967}, exit_delta_spread={"value": 1.5522}, entry_delta_spread={"value": 1.748},
             window_size2={"value": 2967}, exit_delta_spread2={"value": 1.5522}, entry_delta_spread2={"value": 1.748},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3068}, exit_delta_spread={"value": 1.5072}, entry_delta_spread={"value": 1.7744},
             window_size2={"value": 3068}, exit_delta_spread2={"value": 1.5072}, entry_delta_spread2={"value": 1.7744},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3264}, exit_delta_spread={"value": 1.502}, entry_delta_spread={"value": 1.7159},
             window_size2={"value": 3264}, exit_delta_spread2={"value": 1.502}, entry_delta_spread2={"value": 1.7159},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3420}, exit_delta_spread={"value": 1.5054}, entry_delta_spread={"value": 1.8241},
             window_size2={"value": 3420}, exit_delta_spread2={"value": 1.5054}, entry_delta_spread2={"value": 1.8241},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3632}, exit_delta_spread={"value": 1.5313}, entry_delta_spread={"value": 1.9511},
             window_size2={"value": 3632}, exit_delta_spread2={"value": 1.5313}, entry_delta_spread2={"value": 1.9511},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3841}, exit_delta_spread={"value": 1.5198}, entry_delta_spread={"value": 1.9511},
             window_size2={"value": 3841}, exit_delta_spread2={"value": 1.5198}, entry_delta_spread2={"value": 1.9511},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 2960}, exit_delta_spread={"value": 1.5276}, entry_delta_spread={"value": 1.7091},
             window_size2={"value": 2960}, exit_delta_spread2={"value": 1.5276}, entry_delta_spread2={"value": 1.7091},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3033}, exit_delta_spread={"value": 1.5106}, entry_delta_spread={"value": 1.81},
             window_size2={"value": 3033}, exit_delta_spread2={"value": 1.5106}, entry_delta_spread2={"value": 1.81},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3239}, exit_delta_spread={"value": 1.5034}, entry_delta_spread={"value": 1.9092},
             window_size2={"value": 3239}, exit_delta_spread2={"value": 1.5034}, entry_delta_spread2={"value": 1.9092},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3412}, exit_delta_spread={"value": 1.542}, entry_delta_spread={"value": 1.9043},
             window_size2={"value": 3412}, exit_delta_spread2={"value": 1.542}, entry_delta_spread2={"value": 1.9043},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3620}, exit_delta_spread={"value": 1.5457}, entry_delta_spread={"value": 1.7011},
             window_size2={"value": 3620}, exit_delta_spread2={"value": 1.5457}, entry_delta_spread2={"value": 1.7011},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3831}, exit_delta_spread={"value": 1.5346}, entry_delta_spread={"value": 2.0022},
             window_size2={"value": 3831}, exit_delta_spread2={"value": 1.5346}, entry_delta_spread2={"value": 2.0022},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 2951}, exit_delta_spread={"value": 1.5187}, entry_delta_spread={"value": 1.7164},
             window_size2={"value": 2951}, exit_delta_spread2={"value": 1.5187}, entry_delta_spread2={"value": 1.7164},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3003}, exit_delta_spread={"value": 1.541}, entry_delta_spread={"value": 1.7556},
             window_size2={"value": 3003}, exit_delta_spread2={"value": 1.541}, entry_delta_spread2={"value": 1.7556},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3217}, exit_delta_spread={"value": 1.542}, entry_delta_spread={"value": 1.9334},
             window_size2={"value": 3217}, exit_delta_spread2={"value": 1.542}, entry_delta_spread2={"value": 1.9334},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3405}, exit_delta_spread={"value": 1.515}, entry_delta_spread={"value": 1.9197},
             window_size2={"value": 3405}, exit_delta_spread2={"value": 1.515}, entry_delta_spread2={"value": 1.9197},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3613}, exit_delta_spread={"value": 1.5176}, entry_delta_spread={"value": 1.6702},
             window_size2={"value": 3613}, exit_delta_spread2={"value": 1.5176}, entry_delta_spread2={"value": 1.6702},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000}),
        dict(window_size={"value": 3813}, exit_delta_spread={"value": 1.5192}, entry_delta_spread={"value": 1.9827},
             window_size2={"value": 3813}, exit_delta_spread2={"value": 1.5192}, entry_delta_spread2={"value": 1.9827},
             band_funding_system={"value": 'funding_both_sides_no_netting_worst_case'},
             band_funding_system2={"value": 'funding_both_sides_no_netting_worst_case'},
             funding_options={"value": 'option1'}, max_position={"value": 2000000}, max_trade_volume={"value": 10000})
    ]


def xbtusd_production_params(file_name: str = None, band_funding_system: str = None, band_funding_system2: str = None):
    """Retrieve and process production parameters for XBTUSD.

    This function reads a CSV file containing production parameters for
    XBTUSD, maps the real parameters to simulation parameters, and then
    returns the processed parameters. The function takes optional arguments
    to specify different band funding systems that can be used in the
    processing. The resulting DataFrame contains the relevant parameters
    needed for further analysis or simulation.

    Args:
        file_name (str): The name of the CSV file containing production parameters.
        band_funding_system (str?): The first band funding system to use.
        band_funding_system2 (str?): The second band funding system to use.

    Returns:
        DataFrame: A DataFrame containing the processed parameters for XBTUSD.
    """

    df = pd.read_csv(f"/home/equinoxai/production_parameters/XBTUSD/{file_name}")
    df = mapping_real_params_to_simulation_params(df)
    return params_for_xbtusd(df, band_funding_system=band_funding_system, band_funding_system2=band_funding_system2)


def btc_production_params(file_name: str = None, band_funding_system: str = None, band_funding_system2: str = None):
    """Retrieve Bitcoin production parameters from a CSV file.

    This function reads a specified CSV file that contains production
    parameters for Bitcoin. It maps the real parameters to simulation
    parameters and retrieves the relevant parameters specifically for the
    Deribit maker, based on the provided funding systems. This is
    particularly useful for analysts and developers who need structured data
    for simulating and analyzing various Bitcoin production scenarios.

    Args:
        file_name (str): The name of the CSV file containing production parameters.
        band_funding_system (str): The first band funding system to be used.
        band_funding_system2 (str): The second band funding system to be used.

    Returns:
        DataFrame: A DataFrame containing the mapped production parameters for Bitcoin.
    """

    df = pd.read_csv(f"/home/equinoxai/production_parameters/XBTUSD/{file_name}")
    df = mapping_real_params_to_simulation_params(df)
    return params_for_btc_deribit_maker(df, band_funding_system=band_funding_system,
                                        band_funding_system2=band_funding_system2)


def ethusd_production_params(file_name: str = None, band_funding_system: str = None):
    """Retrieve and process production parameters for ETH/USD.

    This function reads a CSV file containing production parameters for
    ETH/USD, maps the real parameters to simulation parameters, and then
    retrieves the relevant parameters based on the specified funding system.
    The function is designed to facilitate the analysis of ETH/USD
    production data by providing a structured output.

    Args:
        file_name (str?): The name of the CSV file containing production parameters.
        band_funding_system (str?): The funding system to be used for parameter retrieval.

    Returns:
        DataFrame: The processed parameters for ETH/USD based on the input data and funding
            system.
    """

    df = pd.read_csv(f"/home/equinoxai/production_parameters/ETHUSD/{file_name}")
    df = mapping_real_params_to_simulation_params(df)
    return params_for_ethusd_short_go_long(df, band_funding_system=band_funding_system)


def ethusd_confirmation_combined(file_name):
    """Process ETH/USD confirmation data from a CSV file.

    This function reads a CSV file containing ETH/USD simulation parameters
    and processes the data using the `params_ethusd_combined_results`
    function. It expects the file to be located in a specific directory
    structure and returns the results of the processing.

    Args:
        file_name (str): The name of the CSV file to be read.

    Returns:
        The result of processing the DataFrame with
            `params_ethusd_combined_results`.
    """

    df = pd.read_csv(f"/home/equinoxai/simulation_params/ETHUSD/{file_name}")
    return params_ethusd_combined_results(df)


def xbtusd_confirmation_combined(file_name):
    """Process XBTUSD confirmation data from a CSV file.

    This function reads a CSV file containing XBTUSD simulation parameters
    and processes the data using the `params_xbtusd_combined_results`
    function. The file path is constructed using a predefined directory
    structure. It is expected that the CSV file contains relevant data for
    further analysis.

    Args:
        file_name (str): The name of the CSV file to be processed.

    Returns:
        The result of the `params_xbtusd_combined_results` function, which is
        expected to handle the DataFrame created from the CSV file.
    """

    df = pd.read_csv(f"/home/equinoxai/simulation_params/XBTUSD/{file_name}")
    return params_xbtusd_combined_results(df)


def btc_confirmation_combined(file_name):
    """Process Bitcoin confirmation data from a CSV file.

    This function reads a CSV file containing Bitcoin confirmation data and
    processes it using the `params_btc_deribit_maker_combined_results`
    function. The file is expected to be located in the specified directory.
    The function returns the results of the processing.

    Args:
        file_name (str): The name of the CSV file containing Bitcoin confirmation data.

    Returns:
        The result of processing the data from the CSV file.
    """

    df = pd.read_csv(f"/home/equinoxai/simulation_params/XBTUSD/{file_name}")
    return params_btc_deribit_maker_combined_results(df)


if __name__ == '__main__':
    start(params['sweep_id'], params['source_sweep_id'], params['select_num_simulations'], params['custom_filter'],
          params['project_name'], params['symbol'], params['file_name'])
