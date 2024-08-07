import argparse
import numpy as np

from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.utils.utils import parse_args


class GenericParser:
    """
    @brief A generic argument parser for configuring trading strategies.

    This class provides a common interface for parsing command-line arguments related to trading strategies. It includes
    methods for adding common arguments, processing parsed arguments, and setting additional parameters based on the
    input.

    @param description A description for the argument parser. Default is an empty string.
    """

    def __init__(self, description=''):
        """
        @brief Initializes the GenericParser with a description and sets up common and specific arguments.

        This constructor initializes an argument parser, adds common arguments related to trading configurations, and
        allows subclasses to add specific arguments as needed. It processes the parsed arguments and sets additional
        parameters.

        @param description A description for the argument parser. Default is an empty string.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.add_common_arguments()
        self.add_specific_arguments()
        self.params = vars(parse_args(self.parser))
        self.post_process_arguments()
        self.set_additional_params()

    def add_common_arguments(self):
        """
        @brief Adds common arguments to the parser.

        This method defines a set of common command-line arguments that are typically used across various trading
        strategies. These include time-related arguments, exchange configurations, strategy parameters, band creation
        parameters, and more.

        @note This method should be called during the initialization of the parser to ensure common arguments are
        available.
        """
        # Common time arguments
        self.parser.add_argument('--t_start_period', default=None, type=str)
        self.parser.add_argument('--t_end_period', default=None, type=str)
        self.parser.add_argument('--t_start', default=None, type=int)
        self.parser.add_argument('--t_end', default=None, type=int)

        # Common parameters for the strategy
        self.parser.add_argument('--exchange_spot', default='Deribit', type=str)
        self.parser.add_argument('--exchange_swap', default='BitMEX', type=str)
        self.parser.add_argument('--spot_instrument', default='ETH-PERPETUAL', type=str)
        self.parser.add_argument('--swap_instrument', default='ETHUSD', type=str)
        self.parser.add_argument('--spot_fee', default=None, type=float)
        self.parser.add_argument('--swap_fee', default=None, type=float)
        self.parser.add_argument('--max_trade_volume', default=3000, type=int)
        self.parser.add_argument('--max_position', default=275000, type=int)
        self.parser.add_argument('--environment', default='staging', type=str)
        self.parser.add_argument('--family', default='deribit_eth', type=str)
        self.parser.add_argument('--strategy', default='', type=str)

        # Common parameters for band creation
        self.parser.add_argument('--train_multiple_periods', default='true', type=str)
        self.parser.add_argument('--period', default=None, type=str)
        self.parser.add_argument('--window_size', default=None, type=int)
        self.parser.add_argument('--entry_delta_spread', default=None, type=float)
        self.parser.add_argument('--exit_delta_spread', default=None, type=float)
        self.parser.add_argument('--band_funding_system', default=None, type=str)
        self.parser.add_argument('--funding_system', default='Quanto_both', type=str)
        self.parser.add_argument('--funding_window', default=90, type=int)
        self.parser.add_argument('--funding_periods_lookback', default=0, type=int)
        self.parser.add_argument('--slow_funding_window', default=0, type=int)
        self.parser.add_argument('--move_bogdan_band', default='No', type=str)
        self.parser.add_argument('--ratio_entry_band_mov', default=1.0, type=float)
        self.parser.add_argument('--minimum_distance', default=0.4, type=float)

        self.parser.add_argument('--ratio_entry_band_mov_ind', default=0.0, type=float)
        self.parser.add_argument('--rolling_time_window_size', default=0.0, type=int)

        # Common parameters for funding_continuous_weight_concept funding system
        self.parser.add_argument("--use_same_values_generic_funding", default='true', type=str)
        self.parser.add_argument("--use_same_slowSwap_slowSpot_generic_funding", default='true', type=str)

        self.parser.add_argument('--constant_depth', default=0, type=float)
        self.parser.add_argument('--swap_market_tick_size', default=0.05, type=float)
        self.parser.add_argument("--pair_name", default="ETH-PERPETUAL~ETHUSD")

        self.parser.add_argument('--use_last_two_periods', default='False', type=str)
        self.parser.add_argument('--filter_only_training', default="False", type=str)
        self.parser.add_argument('--adjustment_entry_band', default=None, type=float)
        self.parser.add_argument('--adjustment_exit_band', default=None, type=float)
        self.parser.add_argument('--adjust_pnl_automatically', default="False", type=str)
        self.parser.add_argument('--maximum_quality', default=1000000, type=float)

    def add_specific_arguments(self):
        """
        @brief Adds specific arguments for a subclass.

        This method is intended to be overridden by subclasses to add arguments specific to their functionality.
        Subclasses should implement this method to include any additional command-line arguments required for their
        particular use case.

        @throws NotImplementedError if not overridden in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method to add specific arguments")

    def post_process_arguments(self):
        """
        @brief Processes parsed arguments to convert and validate data.

        This method is responsible for processing the parsed arguments after they have been read. It performs tasks such
        as converting string representations of booleans to actual boolean values and ensuring that certain parameters
        are synchronized based on specific conditions.

        @note This method is called automatically after parsing arguments and should not be called manually.
        """
        # Convert string booleans to actual booleans
        bool_keys = ['train_multiple_periods', 'use_last_two_periods', 'filter_only_training',
                     'adjust_pnl_automatically', 'use_same_values_generic_funding',
                     'use_same_slowSwap_slowSpot_generic_funding']

        for key in bool_keys:
            self.params[key] = self.params[key] in ["True", "true", "TRUE"]

        if self.params['use_same_slowSwap_slowSpot_generic_funding']:
            for x in range(10):
                self.params[f'hoursBeforeSpot{x}'] = self.params[f'hoursBeforeSwap{x}']
                self.params[f'slowWeightSpot{x}'] = self.params[f'slowWeightSwap{x}']
        else:
            for x in range(10):
                self.params[f'hoursBeforeSpot{x}'] = self.params[f'hoursBeforeSwap{x}']

        if self.params['use_same_values_generic_funding']:
            for x in range(10):
                self.params[f'hoursBeforeSpot{x}'] = self.params[f'hoursBeforeSwap{x}']
                self.params[f'slowWeightSpot{x}'] = self.params[f'slowWeightSwap{x}']
                self.params[f'fastWeightSpot{x}'] = self.params[f'fastWeightSwap{x}']

    def set_additional_params(self):
        """
        @brief Sets additional parameters based on parsed arguments.

        This method is responsible for setting additional parameters that depend on the values of parsed arguments. It
        includes logic for determining fees, latency settings, and other strategy-specific parameters.

        @note This method is called automatically after processing arguments and should not be called manually.
        """
        # Fees
        swap_fee, spot_fee = exchange_fees(self.params['exchange_swap'], self.params['swap_instrument'],
                                           self.params['exchange_spot'], self.params['spot_instrument'])

        if self.params['spot_fee'] is None:
            self.params['spot_fee'] = spot_fee
        if self.params['swap_fee'] is None:
            self.params['swap_fee'] = swap_fee

        self.params['band'] = 'bogdan_bands'
        self.params['area_spread_threshold'] = 0.0
        self.params['volatility'] = None
        self.params['minimum_distance'] = 0.4
        self.params['minimum_value'] = None
        self.params['trailing_value'] = None
        self.params['disable_when_below'] = None
        self.params['ratio_exit_band_mov'] = 0.0
        self.params['force_band_creation'] = True

        # Latencies
        ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(self.params['exchange_swap'],
                                                                  self.params['exchange_spot'])
        self.params.update({
            'latency_spot': ws_spot,
            'latency_swap': ws_swap,
            'latency_try_post': api_swap,
            'latency_cancel': api_swap,
            'latency_spot_balance': api_spot
        })

        # Default values
        self.params.update({
            'lookback': None,
            'recomputation_time': None,
            'target_percentage_exit': None,
            'target_percentage_entry': None,
            'entry_opportunity_source': None,
            'exit_opportunity_source': None,
            'generate_percentage_bands': False,
            'stop_trading': True,
            'high_to_current': True
        })


class InverseContractParser(GenericParser):
    """
    @brief A parser for inverse contract trading strategies.

    This class extends the GenericParser to provide specific argument parsing and configuration for inverse contract
    trading strategies. It includes additional arguments and logic specific to inverse contract trading.

    @note This class should be used when configuring inverse contract trading strategies.
    """

    def add_specific_arguments(self):
        """
        @brief Adds specific arguments for inverse contract strategies.

        This method defines a set of specific command-line arguments that are relevant to inverse contract trading
        strategies. It includes parameters for time-based weights and other strategy-specific configurations.
        """
        for x in range(5):
            self.parser.add_argument(f"--hoursBeforeSwap{x}", default=np.nan, type=int)
            self.parser.add_argument(f"--slowWeightSwap{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--fastWeightSwap{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--hoursBeforeSpot{x}", default=np.nan, type=int)
            self.parser.add_argument(f"--slowWeightSpot{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--fastWeightSpot{x}", default=np.nan, type=float)

        self.parser.add_argument('--use_local_min_max', default="False", type=str)
        self.parser.add_argument('--use_same_num_rolling_points', default="False", type=str)
        self.parser.add_argument('--use_extended_filter', default="False", type=str)
        self.parser.add_argument('--num_of_points_to_lookback_entry', default=None, type=int)
        self.parser.add_argument('--num_of_points_to_lookback_exit', default=None, type=int)

    def post_process_arguments(self):
        """
        @brief Processes parsed arguments specific to inverse contract strategies.

        This method extends the post-processing logic of the GenericParser to handle additional parameters specific to
        inverse contract trading strategies. It includes logic for converting string booleans and synchronizing
        parameters when certain conditions are met.
        """
        super().post_process_arguments()

        bool_keys = ['use_local_min_max', 'use_same_num_rolling_points', 'use_extended_filter']
        for key in bool_keys:
            self.params[key] = self.params[key] in ["True", "true", "TRUE"]

        if self.params['use_same_num_rolling_points']:
            self.params['num_of_points_to_lookback_exit'] = self.params['num_of_points_to_lookback_entry']

    def set_additional_params(self):
        """
        @brief Sets additional parameters specific to inverse contract strategies.

        This method extends the additional parameter setting logic of the GenericParser to handle inverse contract
        trading strategies. It ensures that certain parameters are configured correctly based on the parsed arguments.
        """
        super().set_additional_params()
        self.params.update({
            'adjust_pnl_automatically': self.params['adjust_pnl_automatically'] in ['True', 'true', 'TRUE']
        })


class QuantoContractParser(GenericParser):
    """
    @brief A parser for quanto contract trading strategies.

    This class extends the GenericParser to provide specific argument parsing and configuration for quanto contract
    trading strategies. It includes additional arguments and logic specific to quanto contract trading.

    @note This class should be used when configuring quanto contract trading strategies.
    """

    def add_specific_arguments(self):
        """
        @brief Adds specific arguments for quanto contract strategies.

        This method defines a set of specific command-line arguments that are relevant to quanto contract trading
        strategies. It includes parameters for ratio-based movements, thresholds, and other strategy-specific
        configurations.
        """
        self.parser.add_argument('--current_r', default=None, type=float)
        self.parser.add_argument('--high_r', default=None, type=float)
        self.parser.add_argument('--quanto_threshold', default=None, type=float)
        self.parser.add_argument('--hours_to_stop', default=None, type=int)

        self.parser.add_argument('--ratio_entry_band_mov_long', default=None, type=float)
        self.parser.add_argument('--ratio_exit_band_mov_ind_long', default=None, type=float)
        self.parser.add_argument('--rolling_time_window_size_long', default=None, type=int)

        for x in range(10):
            self.parser.add_argument(f"--hoursBeforeSwap{x}", default=np.nan, type=int)
            self.parser.add_argument(f"--slowWeightSwap{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--fastWeightSwap{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--hoursBeforeSpot{x}", default=np.nan, type=int)
            self.parser.add_argument(f"--slowWeightSpot{x}", default=np.nan, type=float)
            self.parser.add_argument(f"--fastWeightSpot{x}", default=np.nan, type=float)


def main(params):
    # Here you can use the params dictionary as needed in your code
    print("Parsed Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select parser type')
    parser.add_argument('--parser_type', type=str, required=True, choices=['inverse', 'quanto'],
                        help='Type of parser to use: inverse or quanto')

    args, remaining_args = parser.parse_known_args()

    if args.parser_type == 'inverse':
        inverse_contract_parser = InverseContractParser()
        params = inverse_contract_parser.params
    elif args.parser_type == 'quanto':
        quanto_contract_parser = QuantoContractParser()
        params = quanto_contract_parser.params
    else:
        raise ValueError("Invalid parser type specified. Choose either 'inverse' or 'quanto'.")

    main(params)
