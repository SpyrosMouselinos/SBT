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
        if not self.funding_system_swap_df.hoursBefore.empty:
            if self.funding_system_swap_df.hoursBefore.iloc[-1] != 0:
                self.funding_system_swap_df.hoursBefore.iloc[-1] = 0
        if not self.funding_system_spot_df.hoursBefore.empty:
            if self.funding_system_spot_df.hoursBefore.iloc[-1] != 0:
                self.funding_system_spot_df.hoursBefore.iloc[-1] = 0

        # drop the nan values if they exist from the list

        self.ending = ending

        if self.ending != None:
            self.strategy_name = f"generic_{self.swap_name}_{self.swap_symbol}_{self.spot_name}_{self.spot_symbol}_{self.ending}"

    def generate_bogdan_bands_from_query(self, entry_delta_spread=None, exit_delta_spread=None):
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

    def load_bands_from_disk(self):
        base_dir = os.path.join(os.getenv("STORED_BANDS_DIR"), f"{self.spot_symbol}_{self.swap_symbol}")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        filename = os.path.join(base_dir,
                                f"{self.t0}_{self.t1}_{self.window_size}_{self.spot_fee}_{self.swap_fee}.parquet.br")
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

    def generate_bogdan_bands(self, from_time=None, to_time=None):
        '''
        function to create bands
        * time should be given in milliseconds
        * if environment = production then an additional argument corresponding to the ending of the strategy name
        should be given
        The function has a buffer so post-processing is needed for a clear-cut in terms of time.
        '''
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

        if not self.use_bps:
            del body['use_bps']
        # print(f"body : {body}")
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
            if status != 'queued':
                print(f'band generation failed {send_post.json()}')
                exit()

            api_status = 'not_found'
            time_start = int(time.time() * 1000)
            time_now = time_start
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

            if api_status == 'found':
                if len(get_data_from_api.json()['data']) == 0:
                    print("empty array returned")
                    exit()
                output_data = get_data_from_api.json()['data']
                ### HotFix -- Remove any buffer data ###
                filtered_output_data = []
                for entry in output_data:
                    time_in_int64 = entry[0]
                    if from_time <= time_in_int64 <= to_time:
                        filtered_output_data.append(entry)
                return filtered_output_data
            else:
                print('time limit of 60min exceeded ')
                exit()
        else:
            print("wrong environment input")
            exit()

    def generate_spreads(self):
        '''
        function to generate spreads one-off
        this function requires the environment variable to be set to "staging"
        also it has a sleep time of 5min in order for the spreads to be generated.
        '''
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

    def create_band_from_api(self):
        days_in_milliseconds = 1000 * 60 * 60 * 24 * 7
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


def format_band_values(band_list, col_entry: str = 'Entry Band', col_exit: str = 'Exit Band', get_dates: bool = True):
    band_values = pd.DataFrame(band_list, columns=['timems', 'side', 'value'])
    if get_dates:
        band_values['Time'] = pd.to_datetime(band_values['timems'], unit='ms', utc=True)
    band_values[col_entry] = band_values.loc[band_values['side'] == 'entry', 'value']
    band_values[col_exit] = band_values.loc[band_values['side'] == 'exit', 'value']
    band_values[col_exit] = band_values[col_exit].shift(-1)
    band_values.drop(columns=['side', 'value'], inplace=True)
    band_values.dropna(inplace=True)
    # band_values.index = band_values['Time']
    return band_values
