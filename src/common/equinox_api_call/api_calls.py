import pandas as pd
import os
import requests
import time
from dotenv import load_dotenv, find_dotenv
import numpy as np
# ------------------------------------------------------------------#
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.utils.utils import handle_different_size_lists

load_dotenv(find_dotenv())


class DatalinkCreateBands:
    '''
    class to connect with equinox api and create bands and spreads
    '''

    def __init__(self, t_start: int, t_end: int,
                 swap_exchange: str,
                 swap_symbol: str,
                 spot_exchange: str,
                 spot_symbol: str,
                 window_size: int = None,
                 entry_delta_spread: float = None,
                 exit_delta_spread: float = None,
                 swap_fee: float = None,
                 spot_fee: float = None,
                 generate_percentage_bands: bool = None,
                 funding_system: str = None,
                 funding_window: int = 90,
                 funding_periods_lookback: int = 0,
                 slow_funding_window: int = 0,
                 environment: str = None,
                 recomputation_time: str = None,
                 entry_opportunity_source: str = None,
                 exit_opportunity_source: str = None,
                 target_percentage_entry: float = None,
                 target_percentage_exit: float = None,
                 lookback: str = None,
                 minimum_target: str = None,
                 use_aggregated_opportunity_points: bool = None,
                 hoursBeforeSwapList: list = None,
                 slowWeightSwapList: list = None,
                 fastWeightSwapList: list = None,
                 hoursBeforeSpotList: list = None,
                 slowWeightSpotList: list = None,
                 fastWeightSpotList: list = None,
                 ending: str = None,
                 days_in_millis: int = 1000 * 60 * 60 * 24 * 7,
                 use_bps=False):
        """
        @brief Initializes the FundingPlan object. This is a class method to be used by subclasses and should not be called directly
        @param t_start The start time of the time range to be used.
        @param t_end The end time of the time range to be used.
        @param swap_exchange The exchange that will be swappable.
        @param swap_symbol The symbol that will be swapped.
        @param spot_exchange The exchange that will be spotted.
        @param spot_symbol The symbol that will be used to exchange the spot exchange.
        @param window_size The number of time windows in which the exchange will be sped.
        @param entry_delta_spread
        @param exit_delta_spread The spread of the spot spread.
        @param swap_fee
        @param spot_fee
        @param generate_percentage_bands
        @param funding_system
        @param funding_window slow_ The number of periods that will be used for the recomputation of the fundering funding period.
        @param funding_periods_lookback
        @param slow_funding_window
        @param environment
        @param recomputation_time
        @param entry_opportunity_source
        @param exit_opportunity_source
        @param target_percentage_entry
        @param target_percentage_exit
        @param lookback
        @param minimum_target
        @param use_aggregated_opportunity_points
        @param hoursBeforeSwapList
        @param slowWeightSwapList
        @param fastWeightSwapList
        @param hoursBeforeSpotList
        @param slowWeightSpotList
        @param fastWeightSpotList
        @param ending
        @param days_in_millis
        @param use_bps
        """
        self.environment = environment

        self.t0 = t_start
        self.t1 = t_end

        self.swap_name = swap_exchange
        self.swap_symbol = swap_symbol
        self.spot_name = spot_exchange
        self.spot_symbol = spot_symbol
        self.window_size = window_size
        self.entry_delta_spread = entry_delta_spread
        self.exit_delta_spread = exit_delta_spread
        self.funding_system = funding_system
        self.funding_window = funding_window
        self.funding_periods_lookback = funding_periods_lookback
        self.slow_funding_window = slow_funding_window
        self.strategy_name = f"generic_{self.swap_name}_{self.swap_symbol}_{self.spot_name}_{self.spot_symbol}_ws_{self.window_size}_ens_{self.entry_delta_spread}_exs_{self.exit_delta_spread}"

        self.swap_fee = swap_fee
        self.spot_fee = spot_fee
        self.use_bps = use_bps

        self.generate_percentage_bands = generate_percentage_bands
        self.recomputation_time = recomputation_time
        self.entry_opportunity_source = entry_opportunity_source
        self.exit_opportunity_source = exit_opportunity_source
        self.target_percentage_entry = target_percentage_entry
        self.target_percentage_exit = target_percentage_exit
        self.lookback = lookback
        self.minimum_target = minimum_target
        self.use_aggregated_opportunity_points = use_aggregated_opportunity_points
        self.days_in_millis = days_in_millis

        # drop the nan values if they exist from the list
        hoursBeforeSwapList = [int(x) for x in hoursBeforeSwapList if str(x) != 'nan']
        slowWeightSwapList = [float(x) for x in slowWeightSwapList if str(x) != 'nan']
        fastWeightSwapList = [float(x) for x in fastWeightSwapList if str(x) != 'nan']
        hoursBeforeSpotList = [int(x) for x in hoursBeforeSpotList if str(x) != 'nan']
        slowWeightSpotList = [float(x) for x in slowWeightSpotList if str(x) != 'nan']
        fastWeightSpotList = [float(x) for x in fastWeightSpotList if str(x) != 'nan']

        # check if the lists are of same length and if not make them of same length
        hoursBeforeSwapList, slowWeightSwapList, fastWeightSwapList = handle_different_size_lists(hoursBeforeSwapList,
                                                                                                  slowWeightSwapList,
                                                                                                  fastWeightSwapList)
        hoursBeforeSpotList, slowWeightSpotList, fastWeightSpotList = handle_different_size_lists(hoursBeforeSpotList,
                                                                                                  slowWeightSpotList,
                                                                                                  fastWeightSpotList)

        self.funding_system_swap_df = pd.DataFrame({'hoursBefore': hoursBeforeSwapList,
                                                    'slowWeight': slowWeightSwapList,
                                                    'fastWeight': fastWeightSwapList})
        self.funding_system_spot_df = pd.DataFrame({'hoursBefore': hoursBeforeSpotList,
                                                    'slowWeight': slowWeightSpotList,
                                                    'fastWeight': fastWeightSpotList})

        self.funding_system_swap_df.sort_values(by='hoursBefore', ascending=False, inplace=True)
        self.funding_system_spot_df.sort_values(by='hoursBefore', ascending=False, inplace=True)

        self.funding_system_swap_df.drop_duplicates(subset='hoursBefore', keep='last', inplace=True)
        self.funding_system_spot_df.drop_duplicates(subset='hoursBefore', keep='last', inplace=True)

        # make sure that the last value of the hoursBefore is 0
        # if self.unding_system_swap_df. hoursBefore. iloc 1
        if not self.funding_system_swap_df.hoursBefore.empty:
            # Set the hoursBefore field to 0.
            if self.funding_system_swap_df.hoursBefore.iloc[-1] != 0:
                self.funding_system_swap_df.hoursBefore.iloc[-1] = 0
        # if the funding system spot is empty
        if not self.funding_system_spot_df.hoursBefore.empty:
            # Set the hoursBefore field to 0.
            if self.funding_system_spot_df.hoursBefore.iloc[-1] != 0:
                self.funding_system_spot_df.hoursBefore.iloc[-1] = 0

        # drop the nan values if they exist from the list

        self.ending = ending

        # The strategy name for this strategy.
        if self.ending != None:
            self.strategy_name = f"generic_{self.swap_name}_{self.swap_symbol}_{self.spot_name}_{self.spot_symbol}_{self.ending}"

    # Generate Bollinger bands from query. This is a wrapper around generate_bollinger_bands_from_query
    def generate_bollinger_bands_from_query(self, entry_delta_spread=None, exit_delta_spread=None):
        """
         @brief Generate Bollinger bands from query. This is a generator function that can be used to generate a list of band names to be bought by the band_bollinger function
         @param entry_delta_spread entry spread ( m / s )
         @param exit_delta_spread exit spread ( m / s )
         @return list of band names to be bought by the band_bollinger
        """
        # Set the entry and exit delta spread.
        if entry_delta_spread is None:
            entry_delta_spread = self.entry_delta_spread
            exit_delta_spread = self.exit_delta_spread
        query_string = f'''
        select central + {entry_delta_spread} as "Entry Band", central - {exit_delta_spread} as "Exit Band" from (
        select moving_average(mean_spread, {self.window_size}) as central from (
        select (mean(spread_entry) + mean(spread_exit)) / 2 as mean_spread from (
        select  10000 * (swap_entry - spot_entry) / ((swap_entry + spot_entry) / 2) as spread_entry, 10000 * (swap_exit - spot_exit) / ((swap_entry + spot_entry) / 2) as spread_exit from
        (select swap_entry, spot_entry, swap_exit, spot_exit from 
        (SELECT "price" * (1 - {'{:f}'.format(self.swap_fee)}) as swap_exit FROM "price" WHERE ("exchange" = '{self.swap_name}' and 
        "symbol" = '{self.swap_symbol}' AND "side" = 'Ask') AND time >= {self.t0 - self.window_size * 1000 * 60}ms AND time <= {self.t1}ms),
        (SELECT "price" * (1 + {'{:f}'.format(self.spot_fee)}) as spot_exit FROM "price" WHERE ("exchange" = '{self.spot_name}' and 
        "symbol" = '{self.spot_symbol}' AND "side" = 'Ask') AND time >= {self.t0 - self.window_size * 1000 * 60}ms AND time <= {self.t1}ms),
        (SELECT "price" * (1 - {'{:f}'.format(self.swap_fee)}) as swap_entry FROM "price" WHERE ("exchange" = '{self.swap_name}' and 
        "symbol" = '{self.swap_symbol}' AND "side" = 'Bid') AND time >= {self.t0 - self.window_size * 1000 * 60}ms AND time <= {self.t1}ms),
        (SELECT "price" * (1 + {'{:f}'.format(self.spot_fee)}) as spot_entry FROM "price" WHERE ("exchange" = '{self.spot_name}' and 
        "symbol" = '{self.spot_symbol}' AND "side" = 'Bid') AND time >= {self.t0 - self.window_size * 1000 * 60}ms AND time <= {self.t1}ms)

        fill(previous)
        )
        ) group by time(1m)
        )
        )'''
        connection = InfluxConnection.getInstance()
        result = connection.staging_client_spotswap_dataframe.query(query_string)
        return result['price']

    # Load bands from disk and store them in bands_dict. This is called after a call to load_bands
    def load_bands_from_disk(self):
        """
         @brief Load Bogdan bands from disk. If not found generate them and return them
         @return pd. DataFrame with band
        """
        base_dir = os.path.join(os.getenv("STORED_BANDS_DIR"), f"{self.spot_symbol}_{self.swap_symbol}")
        # Create a directory if it doesn t exist.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        filename = os.path.join(base_dir,
                                f"{self.t0}_{self.t1}_{self.window_size}_{self.spot_fee}_{self.swap_fee}.parquet.br")
        # Returns a pandas. DataFrame containing the bands of the Bogdan band data.
        if os.path.exists(filename):
            print("")
            bands = pd.read_parquet(filename, engine="pyarrow")
            bands["Entry Band"] = bands["Entry Band"] + self.entry_delta_spread
            bands["Exit Band"] = bands["Exit Band"] - self.exit_delta_spread
            return bands
        else:
            print("Bands not found on disk. Querying them...")
            bands = self.generate_bogdan_bands_from_query(0, 0)
            bands['timems'] = (bands.index.view(np.int64) + 1) / 1000000
            bands['Time'] = bands.index
            bands.reset_index(inplace=True, drop=True)
            bands = bands[bands['timems'] > self.t0 - 1000 * 60 * 60]
            bands.to_parquet(filename, engine="pyarrow", compression='brotli')
            bands["Entry Band"] = bands["Entry Band"] + self.entry_delta_spread
            bands["Exit Band"] = bands["Exit Band"] - self.exit_delta_spread
            return bands

    # Generate Bollinger bands. This is a wrapper around the band generation function
    def generate_bollinger_bands(self, from_time=None, to_time=None):
        """
         @brief Generates bands based on funding system. Bollinger bands are defined as a set of time points that correspond to the times in which the strategy is applied.
         @param from_time start time of the band generation in milliseconds
         @param to_time end time of the band generation in milliseconds
         @return list of dictionaries with keys : time_points : time points for each band in the
        """
        # Sets the from and to time.
        if from_time is None and to_time is None:
            from_time = self.t0
            to_time = self.t1

        swap_funding_weights_entry = [{"hoursBefore": int(self.funding_system_swap_df.hoursBefore.iloc[ix]),
                                       "slowWeight": self.funding_system_swap_df.slowWeight.iloc[ix],
                                       "fastWeight": self.funding_system_swap_df.fastWeight.iloc[ix]} for ix in
                                      range(len(self.funding_system_swap_df))]

        spot_funding_weights_entry = [{"hoursBefore": int(self.funding_system_spot_df.hoursBefore.iloc[ix]),
                                       "slowWeight": self.funding_system_spot_df.slowWeight.iloc[ix],
                                       "fastWeight": self.funding_system_spot_df.fastWeight.iloc[ix]} for ix in
                                      range(len(self.funding_system_spot_df))]

        body = {"window_size": self.window_size,
                "enable_logs_from_ts": 1715630280000,
                "enable_logs_to_ts": 1715630520000,
                "entry_delta_spread": self.entry_delta_spread,
                "exit_delta_spread": self.exit_delta_spread,
                "spot_name": self.spot_name,
                "spot_symbol": self.spot_symbol,
                "swap_name": self.swap_name,
                "swap_symbol": self.swap_symbol,
                "funding_system": self.funding_system,
                "swap_funding_weights_entry": swap_funding_weights_entry,
                "swap_funding_weights_exit": swap_funding_weights_entry,
                "spot_funding_weights_entry": spot_funding_weights_entry,
                "spot_funding_weights_exit": spot_funding_weights_entry,
                "funding_periods_lookback": self.funding_periods_lookback,
                "funding_window": self.funding_window,
                "slow_funding_window": self.slow_funding_window,
                "generate_percentage_bands": self.generate_percentage_bands,
                "recomputation_time": self.recomputation_time,
                "entry_opportunity_source": self.entry_opportunity_source,
                "exit_opportunity_source": self.exit_opportunity_source,
                "target_percentage_entry": self.target_percentage_entry,
                "target_percentage_exit": self.target_percentage_exit,
                "lookback": self.lookback,
                "minimum_target": self.minimum_target,
                "use_aggregated_opportunity_points": self.use_aggregated_opportunity_points,
                "from": int(from_time),
                "to": int(to_time),
                "strategy_name": self.strategy_name,
                "use_bps": self.use_bps,
                "swap_fee": self.swap_fee,
                "spot_fee": self.spot_fee}

        # If the use_bps option is set to true the body will be used to use the BPS.
        if not self.use_bps:
            del body['use_bps']
        # print(f"body : {body}")
        # This method is used to send a band and queue band creation request to the equinoxai. com
        if self.environment == 'production':

            try:
                send_post = requests.post(url="https://api.equinoxai.com/strategies/get_band",
                                          data=body,
                                          headers={
                                              "token": os.getenv("EQUINOX_API_TOKEN"),
                                              'content-type': 'application/x-www-form-urlencoded',
                                              'Cookie': os.getenv("AUTHELIA_COOKIE")
                                          })
                send_post.raise_for_status()
                print('status code: ', send_post.status_code)
                print('bands are generated but not stored in influxdb')
            # Code here will only run if the request is successful
            except requests.exceptions.HTTPError as errh:
                print(errh)
            except requests.exceptions.ConnectionError as errc:
                print(errc)
            except requests.exceptions.Timeout as errt:
                print(errt)
            except requests.exceptions.RequestException as err:
                print(err)

            return send_post.json()
        elif self.environment == 'staging':
            attempts = 0
            # Attempts to send a band creation request to the queue_band_creation API.
            while attempts < 10:
                try:
                    send_post = requests.post(url="https://api.staging.equinoxai.com/strategies/queue_band_creation",
                                              json=body,
                                              headers={
                                                  "token": os.getenv("EQUINOX_API_TOKEN"),
                                                  'content-type': 'application/json',
                                                  'Cookie': os.getenv("AUTHELIA_COOKIE_STAGING")
                                              })
                    send_post.raise_for_status()
                    print('status code: ', send_post.status_code)
                    print('bands are generated but not stored in influxdb')
                    # If the post status code is 200 then break the loop until the post is 200.
                    if send_post.status_code == 200:
                        break
                    # Code here will only run if the request is successful
                except requests.exceptions.HTTPError as errh:
                    print(errh)
                    print(f'HTTPError in attempt:{attempts}')
                    time.sleep(3 + int(attempts))
                    attempts += 1
                except requests.exceptions.ConnectionError as errc:
                    print(errc)
                    print(f'ConnectionError in attempt:{attempts}')
                    time.sleep(3 + int(attempts))
                    attempts += 1
                except requests.exceptions.Timeout as errt:
                    print(errt)
                    print(f'TimeOut in attempt:{attempts}')
                    time.sleep(3 + int(attempts))
                    attempts += 1
                except requests.exceptions.RequestException as err:
                    print(err)
                    print(f'RequestException in attempt:{attempts}')
                    time.sleep(3 + int(attempts))
                    attempts += 1

            temp_id = send_post.json()['id']
            status = send_post.json()['status']
            # send_post. json if status is queued
            if status != 'queued':
                print(f'band generation failed {send_post.json()}')
                exit()

            api_status = 'not_found'
            time_start = int(time.time() * 1000)
            time_now = time_start
            # This function is used to get band result data from the Staging API.
            while api_status != 'found' and time_now - time_start < 1000 * 60 * 60:
                try:
                    get_data_from_api = requests.post(
                        url="https://api.staging.equinoxai.com/strategies/get_band_result",
                        data={"id": temp_id},
                        headers={
                            "token": os.getenv("EQUINOX_API_TOKEN"),
                            'content-type': 'application/x-www-form-urlencoded',
                            'Cookie': os.getenv("AUTHELIA_COOKIE_STAGING")
                        })
                    api_status = get_data_from_api.json()['status']
                except requests.exceptions.HTTPError as errh:
                    print(errh)
                except requests.exceptions.ConnectionError as errc:
                    print(errc)
                except requests.exceptions.Timeout as errt:
                    print(errt)
                except requests.exceptions.RequestException as err:
                    print(err)
                time.sleep(2)
                time_now = int(time.time() * 1000)

            # Returns a list of data from the API
            if api_status == 'found':
                # This function is used to get the data from the API
                if len(get_data_from_api.json()['data']) == 0:
                    print("empty array returned")
                    exit()
                output_data = get_data_from_api.json()['data']
                ### HotFix -- Remove any buffer data ###
                filtered_output_data = []
                # Add entry to filtered_output_data if the time between from_time and to_time are within the interval of the time range.
                for entry in output_data:
                    time_in_int64 = entry[0]
                    # Add entry to filtered_output_data if from_time time_in_int64 to_time.
                    if from_time <= time_in_int64 <= to_time:
                        filtered_output_data.append(entry)
                return filtered_output_data
            else:
                print('time limit of 60min exceeded ')
                exit()
        else:
            print("wrong environment input")
            exit()

    # Generate spreads. This is a wrapper around generate_spreads that allows to pass a function to the constructor
    def generate_spreads(self):
        """
         @brief Generates spreads one-off based on environment. 
         This function requires the environment variable to be set to staging.
         Also it has a sleep time of 5min in order for the spreads to be generated.
        """
        # This method is used to generate a spread using the staging environment
        if self.environment == 'staging':
            body = {"exchange_1": self.swap_name,
                    "symbol_1": self.swap_symbol,
                    "fee_1": self.swap_fee,
                    "exchange_2": self.spot_name,
                    "symbol_2": self.spot_symbol,
                    "fee_2": self.spot_fee,
                    "from": self.t0,
                    "to": self.t1}
            try:
                send_post = requests.post(url="https://api.staging.equinoxai.com/strategies/generate_spread",
                                          data=body,
                                          headers={
                                              'content-type': 'application/x-www-form-urlencoded',
                                              'Cookie': os.getenv("AUTHELIA_COOKIE_STAGING")
                                          })
                send_post.raise_for_status()
                print(send_post.json())
                print('status code: ', send_post.status_code)
                # Code here will only run if the request is successful
            except requests.exceptions.HTTPError as errh:
                print(errh)
                print('HTTPError')
            except requests.exceptions.ConnectionError as errc:
                print(errc)
                print('ConnectionError')
            except requests.exceptions.Timeout as errt:
                print(errt)
                print('TimeOut')
            except requests.exceptions.RequestException as err:
                print(err)
                print('RequestException')
            time.sleep(300)
        else:
            print('Not the correct environment parameter, environment should be set to "staging"')


# This is a wrapper for FundingOptionsBandCreation
# The reason we need this is because of the need to create a band in order to get the funding
class FundingOptionsBandCreation():

    def __init__(self, t_start: int = 1667260800000, t_end: int = 1669766400000,
                 band_funding_option: str = None,
                 window_size_net: int = 1000, entry_delta_spread_net_entry: float = 2.0,
                 exit_delta_spread_net_exit: float = 2.0,
                 band_funding_system_net: str = 'no_funding',
                 window_size_zero: int = None, entry_delta_spread_entry_zero: float = None,
                 exit_delta_spread_exit_zero: float = None,
                 band_funding_system_zero: str = None,
                 funding_window: int = 90,
                 funding_periods_lookback: int = 0,
                 slow_funding_window: int = 0,
                 swap_exchange: str = 'BitMEX', swap_symbol: str = 'XBTUSD',
                 spot_exchange: str = 'Deribit', spot_symbol: str = 'BTC-PERPETUAL',
                 swap_fee: float = -0.0001, spot_fee: float = 0.0003,
                 environment: str = 'staging',
                 hoursBeforeSwapList: list = None,
                 slowWeightSwapList: list = None,
                 fastWeightSwapList: list = None,
                 hoursBeforeSpotList: list = None,
                 slowWeightSpotList: list = None,
                 fastWeightSpotList: list = None):
        """
        @brief Initialize the FundingModel. Must be called before any other method. It is recommended to call super (). __init__ () as first argument to avoid having to re - initialize the model in the same way as the constructor.
        @param t_start Start time of the time range in seconds.
        @param t_end End time of the time range in seconds.
        @param band_funding_option Band funding option to use.
        @param window_size_net Window size of the net.
        @param entry_delta_spread_net_entry Spread in time of the entry spread.
        @param exit_delta_spread_net_exit Spread in time of the exit spread.
        @param band_funding_system_net Baseline of the funding system.
        @param window_size_zero Number of entries to use when the window size is zero.
        @param entry_delta_spread_entry_zero Spread in time of the entry spread.
        @param exit_delta_spread_exit_zero Spread in time of the exit spread.
        @param band_funding_system_zero
        @param funding_window
        @param funding_periods_lookback
        @param slow_funding_window
        @param swap_exchange
        @param swap_symbol
        @param spot_exchange
        @param spot_symbol
        @param swap_fee
        @param spot_fee
        @param environment
        @param hoursBeforeSwapList
        @param slowWeightSwapList
        @param fastWeightSwapList
        @param hoursBeforeSpotList
        @param slowWeightSpotList
        @param fastWeightSpotList
        """

        self.t0 = t_start
        self.t1 = t_end
        self.band_funding_option = band_funding_option
        self.window_size1 = window_size_net
        self.entry_delta_spread1 = entry_delta_spread_net_entry
        self.exit_delta_spread1 = exit_delta_spread_net_exit
        self.band_funding1 = band_funding_system_net
        self.window_size2 = window_size_zero
        self.entry_delta_spread2 = entry_delta_spread_entry_zero
        self.exit_delta_spread2 = exit_delta_spread_exit_zero
        self.band_funding2 = band_funding_system_zero
        self.environment = environment
        self.funding_window = funding_window
        self.funding_periods_lookback = funding_periods_lookback
        self.slow_funding_window = slow_funding_window

        self.swap_exchange = swap_exchange
        self.swap_symbol = swap_symbol
        self.spot_exchange = spot_exchange
        self.spot_symbol = spot_symbol
        self.swap_fee = swap_fee
        self.spot_fee = spot_fee

        self.hoursBeforeSwapList = hoursBeforeSwapList
        self.slowWeightSwapList = slowWeightSwapList
        self.fastWeightSwapList = fastWeightSwapList
        self.hoursBeforeSpotList = hoursBeforeSpotList
        self.slowWeightSpotList = slowWeightSpotList
        self.fastWeightSpotList = fastWeightSpotList

        # if band_funding_option option1 option2 option3 option4 option5 option5 option6 option7 option5 option6 option7 option5 option6
        if self.band_funding_option == 'option1':
            self.band_funding1 = 'funding_both_sides_no_netting_worst_case'
            self.band_funding2 = 'funding_both_sides_no_netting_worst_case'
        elif self.band_funding_option == 'option2':
            self.band_funding1 = 'funding_both_sides_no_netting_worst_case'
            self.band_funding2 = 'no_funding'
        elif self.band_funding_option == 'option3':
            self.band_funding1 = 'funding_both_sides_capture_funding'
            self.band_funding2 = 'funding_both_sides_no_netting_worst_case'
        elif self.band_funding_option == 'option4':
            self.band_funding1 = 'funding_both_sides_capture_funding'
            self.band_funding2 = 'funding_both_sides_no_netting_worst_case'
        elif self.band_funding_option == 'option5':
            self.band_funding1 = 'funding_both_sides_no_netting'
            self.band_funding2 = 'funding_both_sides_no_netting'
        elif self.band_funding_option == 'option6':
            self.band_funding1 = 'funding_both_sides_no_netting'
            self.band_funding2 = 'no_funding'
        elif self.band_funding_option == 'all':
            # do not override the band_funding1 and band_funding2
            pass

    # Create a band from API. This is a wrapper around the create_band_from_api
    def create_band_from_api(self):
        """
         @brief Create and return a DatalinkCreateBands object from the API data.
         @return A : class : ` ~gwpy. baselib. DatalinkCreateBands `
        """
        days_in_milliseconds = 1000 * 60 * 60 * 24 * 7
        # window size of the window.
        if self.window_size2 is not None:
            ws = max(self.window_size1, self.window_size2)
        else:
            ws = self.window_size1
        datalink1 = DatalinkCreateBands(t_start=self.t0, t_end=self.t1, swap_exchange=self.swap_exchange,
                                        swap_symbol=self.swap_symbol, spot_exchange=self.spot_exchange,
                                        spot_symbol=self.spot_symbol, window_size=self.window_size1,
                                        entry_delta_spread=self.entry_delta_spread1,
                                        exit_delta_spread=self.exit_delta_spread1, swap_fee=self.swap_fee,
                                        spot_fee=self.spot_fee, funding_system=self.band_funding1,
                                        funding_window=self.funding_window,
                                        funding_periods_lookback=self.funding_periods_lookback,
                                        slow_funding_window=self.slow_funding_window, environment=self.environment,
                                        hoursBeforeSwapList=self.hoursBeforeSwapList,
                                        slowWeightSwapList=self.slowWeightSwapList,
                                        fastWeightSwapList=self.fastWeightSwapList,
                                        hoursBeforeSpotList=self.hoursBeforeSpotList,
                                        slowWeightSpotList=self.slowWeightSpotList,
                                        fastWeightSpotList=self.fastWeightSpotList, days_in_millis=days_in_milliseconds)

        bands_normal = datalink1.generate_bogdan_bands()
        # Returns band_values for the band_funding option.
        if self.band_funding_option is None:
            return format_band_values(bands_normal)
        else:
            ws = max(self.window_size1, self.window_size2)
            datalink2 = DatalinkCreateBands(t_start=self.t0, t_end=self.t1, swap_exchange=self.swap_exchange,
                                            swap_symbol=self.swap_symbol, spot_exchange=self.spot_exchange,
                                            spot_symbol=self.spot_symbol, window_size=self.window_size2,
                                            entry_delta_spread=self.entry_delta_spread2,
                                            exit_delta_spread=self.exit_delta_spread2, swap_fee=self.swap_fee,
                                            spot_fee=self.spot_fee, funding_system=self.band_funding2,
                                            funding_window=self.funding_window,
                                            funding_periods_lookback=self.funding_periods_lookback,
                                            slow_funding_window=self.slow_funding_window, environment=self.environment,
                                            hoursBeforeSwapList=self.hoursBeforeSwapList,
                                            slowWeightSwapList=self.slowWeightSwapList,
                                            fastWeightSwapList=self.fastWeightSwapList,
                                            hoursBeforeSpotList=self.hoursBeforeSpotList,
                                            slowWeightSpotList=self.slowWeightSpotList,
                                            fastWeightSpotList=self.fastWeightSpotList,
                                            days_in_millis=days_in_milliseconds)

            band_list2 = datalink2.generate_bogdan_bands()
            band1 = format_band_values(bands_normal, col_entry='Entry Band', col_exit='Exit Band', get_dates=False)
            band2 = format_band_values(band_list2, col_entry='Entry Band Enter to Zero',
                                       col_exit='Exit Band Exit to Zero', get_dates=False)
            band_values = pd.merge_ordered(band1, band2, on='timems')
            band_values['Time'] = pd.to_datetime(band_values['timems'], unit='ms', utc=True)
            band_values.dropna(inplace=True)
            band_values.reset_index(drop=True, inplace=True)
            return band_values


# Format band values for output. This is a function that can be used to format a band value for each band in the data set
def format_band_values(band_list, col_entry: str = 'Entry Band', col_exit: str = 'Exit Band', get_dates: bool = True):
    """
     @brief Formats band values to be used in plotting. This is a helper function to create a DataFrame that can be plotted to visualize the list of bands and to visualize the entries and exits in the band.
     @param band_list A list of band values as returned by : func : ` get_band_list `.
     @param col_entry The column to use for the entry band. Defaults to'Entry Band '. If this is set to False it will not be displayed.
     @param col_exit The column to use for the exit band. Defaults to'Exit Band '. If this is set to False it will not be displayed.
     @param get_dates If True the dates will be converted to milliseconds since January 1 1970.
     @return A DataFrame with the band values formatted to be used in plotting
    """
    band_values = pd.DataFrame(band_list, columns=['timems', 'side', 'value'])
    # If get_dates is True then the band_values Time is set to the time of the band.
    if get_dates:
        band_values['Time'] = pd.to_datetime(band_values['timems'], unit='ms', utc=True)
    band_values[col_entry] = band_values.loc[band_values['side'] == 'entry', 'value']
    band_values[col_exit] = band_values.loc[band_values['side'] == 'exit', 'value']
    band_values[col_exit] = band_values[col_exit].shift(-1)
    band_values.drop(columns=['side', 'value'], inplace=True)
    band_values.dropna(inplace=True)
    # band_values.index = band_values['Time']
    return band_values
