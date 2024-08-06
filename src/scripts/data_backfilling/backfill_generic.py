import requests
import math
import warnings
import datetime
import random
import numba
from dotenv import load_dotenv, find_dotenv

from old_code.maker_maker.MakerMakerMasterFile import DisplacementEvaluationNoSpreadEntry
from src.common.queries.queries import *
from src.simulations.simulation_codebase.latencies_fees.latencies_fees import set_latencies_auto, \
    exchange_fees
from src.common.utils.utils import to_decimal, get_data_for_trader

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


def correlation_values_backfill(t_start, t_end):
    """
    Computes and backfills correlation values between Deribit ETH-PERPETUAL and BitMEX XBTUSD prices 
    for different rolling windows.

    This function retrieves ask prices from the specified exchanges and computes the correlation 
    between them over rolling windows of different sizes (2h, 4h, and 8h). The results are written 
    to an InfluxDB instance for further analysis.

    Parameters:
    ----------
    t_start : int
        The start timestamp in milliseconds for fetching price data.
    t_end : int
        The end timestamp in milliseconds for fetching price data.

    Example:
    -------
    >>> correlation_values_backfill(1622548800000, 1622635200000)
    """

    # Get the InfluxDB connection instance
    connection = InfluxConnection.getInstance()

    # Fetch ask prices for ETH-PERPETUAL from Deribit and XBTUSD from BitMEX
    price1 = get_price(t_start=t_start, t_end=t_end, exchange='Deribit', symbol='ETH-PERPETUAL', side='Ask',
                       environment='production')
    price2 = get_price(t_start=t_start, t_end=t_end, exchange='BitMEX', symbol='XBTUSD', side='Ask',
                       environment='production')

    # Add 'Time' column from index
    price1['Time'] = price1.index
    price2['Time'] = price2.index

    # Merge the price dataframes on 'Time'
    price_ask = pd.merge_ordered(price1, price2, on='Time', suffixes=['_ETH', '_BTC'])

    # Re-index and resample data to 10-second intervals
    price_ask.index = price_ask.Time
    price_ask.drop(columns=['Time'], inplace=True)
    price_ask = price_ask.resample('10s').mean()

    # Calculate rolling correlations for specified window sizes
    price_corr = {}
    for ws in ['2h', '4h', '8h']:
        price_corr[f'{ws}'] = price_ask.rolling(ws).corr().iloc[:, 0]
        price_corr[f'{ws}'].reindex(level=0, copy=True)

    # Merge correlation data into a single DataFrame
    price_corr_df = pd.merge(price_corr['2h'], price_corr['4h'], left_index=True, right_index=True)
    price_corr_df = pd.merge(price_corr_df, price_corr['8h'], left_index=True, right_index=True)
    price_corr_df = price_corr_df.unstack()
    price_corr_df.drop(columns=price_corr_df.columns[[1, 3, 5]], inplace=True)
    price_corr_df = price_corr_df.resample('5T').mean()
    price_corr_df.rename(columns={price_corr_df.columns[0]: '2h', price_corr_df.columns[1]: '4h',
                                  price_corr_df.columns[2]: '8h'}, inplace=True)

    # Convert timestamps to integers in milliseconds
    price_corr_df['timestamp'] = price_corr_df.index.view(int) // 10 ** 6
    price_corr_df.dropna(inplace=True)
    price_corr_df.reset_index(drop=True, inplace=True)

    points = []

    # Prepare data points for InfluxDB
    for idx in price_corr_df.index:
        points.append({
            'time': int(price_corr_df.loc[idx, 'timestamp']),
            'measurement': 'correlations',
            'tags': {'window': 120, 'between': "BitMEX_XBTUSD/Deribit_ETH-PERPETUAL",
                     'number_of_points': 12},
            'fields': {'value': price_corr_df.iloc[idx, 0]}
        })

        points.append({
            'time': int(price_corr_df.loc[idx, 'timestamp']),
            'measurement': 'correlations',
            'tags': {'window': 240, 'between': "BitMEX_XBTUSD/Deribit_ETH-PERPETUAL",
                     'number_of_points': 24},
            'fields': {'value': price_corr_df.iloc[idx, 1]}
        })

        points.append({
            'time': int(price_corr_df.loc[idx, 'timestamp']),
            'measurement': 'correlations',
            'tags': {'window': 480, 'between': "BitMEX_XBTUSD/Deribit_ETH-PERPETUAL",
                     'number_of_points': 48},
            'fields': {'value': price_corr_df.iloc[idx, 2]}
        })

        # Write points in batches of 1000 to avoid overload
        if len(points) >= 1000:
            connection.staging_client_spotswap.write_points(points, time_precision="ms")
            points = []

    # Write any remaining points
    connection.staging_client_spotswap.write_points(points, time_precision="ms")


class BackfillCVI:
    """
    A class to backfill the Crypto Volatility Index (CVI) from an external API into InfluxDB.

    The CVI provides a measure of volatility in the cryptocurrency market. This class retrieves
    historical CVI data from an external API and writes it into an InfluxDB instance for analysis.

    Attributes:
    ----------
    influx_connection : InfluxConnection
        An instance of the InfluxConnection class for connecting to the InfluxDB database.
    """

    def __init__(self):
        """
        Initializes the BackfillCVI class.

        Establishes a connection to the InfluxDB database using the InfluxConnection class.
        """
        self.influx_connection = InfluxConnection.getInstance()

    def cvi_volatility_index(self, t_end):
        """
        Fetches and writes the CVI data to the database for the specified date.

        Parameters:
        ----------
        t_end : datetime
            The end date for fetching the CVI data. The data is fetched from the API for this date.

        Example:
        -------
        >>> cvi = BackfillCVI()
        >>> cvi.cvi_volatility_index(datetime(2023, 8, 1))
        """
        # Fetch historical CVI data from the API
        df = requests.get('https://api.dev-cvi-finance-route53.com/history?chain=Ethereum&index=CVI',
                          params={'Date': f'{t_end}'})

        # Convert JSON response to NumPy array
        cvi = np.array(df.json())
        points = []

        # Prepare data points for InfluxDB
        for ix in range(len(cvi)):
            point = {
                'time': int(cvi[ix, 0] * 1000),  # Convert to milliseconds
                'measurement': 'dvol',
                'tags': {'coin': 'CVI'},
                'fields': {'close': cvi[ix, 1]}
            }
            points.append(point)

        # Write points to the database
        self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')


class QualityOfExecutions:
    """
    A class to evaluate the quality of trade executions for a specified strategy and time range.

    This class computes the quality of executions by comparing actual executed spreads with
    theoretical bands, and writes the results to an InfluxDB instance for further analysis.

    Attributes:
    ----------
    influx_connection : InfluxConnection
        An instance of the InfluxConnection class for connecting to the InfluxDB database.
    """

    def __init__(self):
        """
        Initializes the QualityOfExecutions class.

        Establishes a connection to the InfluxDB database using the InfluxConnection class.
        """
        self.influx_connection = InfluxConnection.getInstance()

    def backfill(self, t0, t1, strategy, entry_delta_spread, exit_delta_spread, window_size, environment):
        """
        Computes and backfills execution quality data for the specified strategy and time range.

        This method retrieves executed spreads and band data from the InfluxDB database,
        calculates the quality of executions, and writes the results back to the database.

        Parameters:
        ----------
        t0 : int
            The start timestamp in milliseconds for fetching data.
        t1 : int
            The end timestamp in milliseconds for fetching data.
        strategy : str
            The trading strategy being evaluated.
        entry_delta_spread : float
            The entry delta spread used in the evaluation.
        exit_delta_spread : float
            The exit delta spread used in the evaluation.
        window_size : int
            The window size for calculating execution quality.
        environment : str
            The environment for database connection, either 'production' or 'staging'.

        Example:
        -------
        >>> qoe = QualityOfExecutions()
        >>> qoe.backfill(1622548800000, 1622635200000, 'MyStrategy', 0.5, 0.3, 20, 'production')
        """
        # Choose the appropriate client and write method based on the environment
        client = self.influx_connection.prod_client_spotswap_dataframe if environment == 'production' else self.influx_connection.staging_client_spotswap_dataframe
        write = self.influx_connection.prod_client_spotswap if environment == 'production' else self.influx_connection.staging_client_spotswap

        # Query executed spread data
        result = client.query(f'''SELECT "spread", type, 
        volume_executed_spot  FROM "executed_spread" WHERE ("strategy" = '{strategy}') AND time >= {t0}ms AND time <= 
        {t1}ms ''', epoch='ns')
        if len(result) == 0:
            return

        # Process execution data
        executions = result["executed_spread"]
        executions['Time'] = executions.index
        executions['entry_executions'] = executions[executions.type == 'entry']['spread']
        executions['exit_executions'] = executions[executions.type == 'exit']['spread']
        executions['entry_volume'] = executions[executions.type == 'entry']['volume_executed_spot']
        executions['exit_volume'] = executions[executions.type == 'exit']['volume_executed_spot']

        # Query band data and determine entry and exit bands
        result2 = self.influx_connection.prod_client_spotswap_dataframe.query(f'''SELECT "value","side" FROM "band" 
            WHERE ("strategy" ='{strategy}' AND "type" = 'live') 
            AND time >= {t0 - 60000}ms and time <= {t1}ms''', epoch='ns')
        if len(result2) == 0:
            result1 = client.query(f'''SELECT ("exit_window_avg" + 
            "entry_window_avg")/2 AS "Band" FROM bogdan_bins_{strategy} WHERE time >= {t0 - 60000}ms and time <= {t1}ms''',
                                   epoch='ns')
            bands = result1[f'bogdan_bins_{strategy}']
            bands['Time'] = bands.index
            bands['Entry Band'] = bands['Band'] + entry_delta_spread
            bands['Exit Band'] = bands['Band'] - exit_delta_spread
        else:
            bands = result2["band"]
            bands['Time'] = bands.index
            bands['Entry Band'] = bands.loc[bands['side'] == 'entry', 'value']
            bands['Exit Band'] = bands.loc[bands['side'] == 'exit', 'value']
            bands.drop(columns=['side', 'value'], inplace=True)

        # Merge execution and band data
        entry_exit_exec = pd.merge_ordered(executions, bands, on='Time')
        entry_exit_exec['Entry Band'].ffill(inplace=True)
        entry_exit_exec['Exit Band'].ffill(inplace=True)
        entry_exit_exec['timestamp'] = entry_exit_exec['Time'].astype(int) / 10 ** 6
        entry_exit_exec.reset_index(drop=True, inplace=True)
        points = []

        # Calculate and prepare execution quality data points
        for ix in entry_exit_exec.index:

            if not math.isnan(entry_exit_exec['entry_executions'].loc[ix]):
                if (1000 > entry_exit_exec.loc[ix, 'Entry Band'] > -1000) or ix == 0:
                    quality_exec = entry_exit_exec.loc[ix, 'entry_executions'] - entry_exit_exec.loc[ix, 'Entry Band']
                else:
                    quality_exec = entry_exit_exec.loc[ix, 'entry_executions'] - entry_exit_exec.loc[
                        ix - 1, 'Entry Band']

                volume = entry_exit_exec.loc[ix, 'entry_volume']
                delta_spread = entry_delta_spread

                point = {
                    'time': int(entry_exit_exec.loc[ix, 'timestamp']),
                    'measurement': 'execution_quality',
                    'tags': {
                        'strategy': strategy,
                        'type': 'entry'
                    },
                    'fields': {'diff_band': quality_exec, "volume": volume, 'delta_spread': delta_spread,
                               'window_size': window_size}
                }
                points.append(point)

            if not math.isnan(entry_exit_exec.loc[ix, 'exit_executions']):

                if (1000 > entry_exit_exec.loc[ix, 'Exit Band'] > -1000) or ix == 0:
                    quality_exec = entry_exit_exec.loc[ix, 'Exit Band'] - entry_exit_exec.loc[ix, 'exit_executions']
                else:
                    quality_exec = entry_exit_exec.loc[ix - 1, 'Exit Band'] - entry_exit_exec.loc[ix, 'exit_executions']

                volume = entry_exit_exec.loc[ix, 'exit_volume']
                delta_spread = exit_delta_spread

                point = {
                    'time': int(entry_exit_exec.loc[ix, 'timestamp']),
                    'measurement': 'execution_quality',
                    'tags': {
                        'strategy': strategy,
                        'type': 'exit'
                    },
                    'fields': {'diff_band': quality_exec, "volume": volume, 'delta_spread': delta_spread,
                               'window_size': window_size}
                }
                points.append(point)

            # Write points in batches of 10000 to avoid overload
            if len(points) > 10000:
                write.write_points(points, time_precision='ms')
                points = []

        # Write any remaining points
        write.write_points(points, time_precision='ms')


class BackfillDeribitVolatilityIndex:
    """
    A class to backfill the Deribit Volatility Index (DVol) data for BTC and ETH.

    This class retrieves historical volatility data from the Deribit API and writes it to an InfluxDB instance for
    analysis.

    Attributes:
    ----------
    influx_connection : InfluxConnection
        An instance of the InfluxConnection class for connecting to the InfluxDB database.
    """

    def __init__(self):
        """
        Initializes the BackfillDeribitVolatilityIndex class.

        Establishes a connection to the InfluxDB database using the InfluxConnection class.
        """
        self.influx_connection = InfluxConnection.getInstance()

    def deribit_volatility(self, t_start, t_end):
        """
        Retrieves volatility index data from the Deribit API for BTC and ETH.

        This method fetches volatility data for BTC and ETH from the Deribit API between the specified start and end
        timestamps.

        Parameters:
        ----------
        t_start : int
            The start timestamp in milliseconds for fetching data.
        t_end : int
            The end timestamp in milliseconds for fetching data.

        Returns:
        -------
        points : list
            A list of data points formatted for InfluxDB, each containing volatility data for a specific coin and
            timestamp.

        Example:
        -------
        >>> dvi = BackfillDeribitVolatilityIndex()
        >>> points = dvi.deribit_volatility(1622548800000, 1622635200000)
        """

        # Fetch volatility data for BTC from the Deribit API
        df = requests.get('https://www.deribit.com/api/v2/public/get_volatility_index_data',
                          params={'currency': "BTC",
                                  'start_timestamp': f"{t_start}",
                                  'resolution': 60,
                                  'end_timestamp': f"{t_end}"})

        # Convert the JSON response to a NumPy array
        dvol_btc = np.array(df.json()['result']['data'])

        # Fetch volatility data for ETH from the Deribit API
        dff = requests.get('https://www.deribit.com/api/v2/public/get_volatility_index_data',
                           params={'currency': "ETH",
                                   'start_timestamp': f"{t_start}",
                                   'resolution': 60,
                                   'end_timestamp': f"{t_end}"})

        # Convert the JSON response to a NumPy array
        dvol_eth = np.array(dff.json()['result']['data'])
        points = []

        # Determine the maximum number of data points
        max_p = max(len(dvol_btc), len(dvol_eth))

        # Prepare data points for InfluxDB
        for ix in range(max_p):
            if ix <= len(dvol_eth) - 1 and len(dvol_eth) > 0:
                point = {
                    'time': int(dvol_eth[ix, 0]),
                    'measurement': 'dvol',
                    'tags': {'coin': 'ETH'},
                    'fields': {'open': dvol_eth[ix, 1],
                               'high': dvol_eth[ix, 2],
                               'low': dvol_eth[ix, 3],
                               'close': dvol_eth[ix, 4]}
                }
                points.append(point)
            if ix <= len(dvol_btc) - 1 and len(dvol_btc) > 0:
                point = {
                    'time': int(dvol_btc[ix, 0]),
                    'measurement': 'dvol',
                    'tags': {'coin': 'BTC'},
                    'fields': {'open': dvol_btc[ix, 1],
                               'high': dvol_btc[ix, 2],
                               'low': dvol_btc[ix, 3],
                               'close': dvol_btc[ix, 4]}
                }
                points.append(point)

        return points

    def write_points(self, t0, t1, env):
        """
        Writes volatility index data points to the InfluxDB database.

        This method divides the time range into smaller intervals, retrieves volatility data for each interval, and
        writes the data points to the database.

        Parameters:
        ----------
        t0 : int
            The start timestamp in milliseconds for writing data.
        t1 : int
            The end timestamp in milliseconds for writing data.
        env : str
            The environment for database connection, either 'staging' or 'production'.

        Example:
        -------
        >>> dvi = BackfillDeribitVolatilityIndex()
        >>> dvi.write_points(1622548800000, 1622635200000, 'staging')
        """
        # Select the appropriate client based on the environment
        if env == "staging":
            client = self.influx_connection.staging_client_spotswap
        else:
            client = self.influx_connection.prod_client_spotswap

        # Check if the time range is small enough for a single request
        if t1 - t0 <= 1000 * 60 * 1000:
            points = self.deribit_volatility(t_start=t0, t_end=t1)
            client.write_points(points, time_precision='ms')
        else:
            # Divide the time range into smaller intervals and write data points
            t_start = t0
            t_end = t_start + 1000 * 60 * 1000

            while t_end <= t1:
                time.sleep(0.2)  # Sleep to avoid rate limiting
                points = self.deribit_volatility(t_start=t_start, t_end=t_end)
                client.write_points(points, time_precision='ms')
                t_start = t_start + 1000 * 60 * 1000
                t_end = t_end + 1000 * 60 * 1000


class StrategyPNL:
    """
    A class to calculate the Profit and Loss (PnL) for a specified trading strategy.

    This class computes the PnL by comparing the Mark-to-Market (MtM) values at the start and end of a specified time
    range, and writes the results to an InfluxDB instance.

    Attributes:
    ----------
    influx_connection : InfluxConnection
        An instance of the InfluxConnection class for connecting to the InfluxDB database.
    """

    def __init__(self):
        """
        Initializes the StrategyPNL class.

        Establishes a connection to the InfluxDB database using the InfluxConnection class.
        """
        self.influx_connection = InfluxConnection.getInstance()

    def strategy_pnl(self, t0, t1, strategy, transfers=0, environment='production'):
        """
        Calculates the PnL for a given strategy and time range.

        This method retrieves MtM values from the InfluxDB database, calculates the PnL by finding the difference
        between the start and end values, and optionally subtracts any transfers.

        Parameters:
        ----------
        t0 : int
            The start timestamp in milliseconds for calculating PnL.
        t1 : int
            The end timestamp in milliseconds for calculating PnL.
        strategy : str
            The trading strategy being evaluated.
        transfers : float, optional
            The amount of any transfers to subtract from the PnL calculation (default is 0).
        environment : str, optional
            The environment for database connection, either 'production' or 'staging' (default is 'production').

        Returns:
        -------
        dict
            A dictionary containing the calculated PnL, start value, start time, end value, and end time.

        Example:
        -------
        >>> pnl = StrategyPNL()
        >>> result = pnl.strategy_pnl(1622548800000, 1622635200000, 'MyStrategy', 100, 'production')
        >>> print(result)
        {'pnl': 500.0, 'start_value': 1000.0, 'start_time': 1622548800000, 'end_value': 1500.0, 'end_time': 1622635200000}
        """
        # Select the appropriate query based on the environment
        if environment == 'production':
            result = self.influx_connection.prod_client_spotswap_dataframe.query(
                f'''SELECT "current_value" AS "MtM_value" 
                FROM "mark_to_market_changes" 
                WHERE ("strategy" = '{strategy}') 
                AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        elif environment == 'staging':
            result = self.influx_connection.staging_client_spotswap_dataframe.query(
                f'''SELECT "current_value" AS "MtM_value" 
                FROM "mark_to_market_changes" 
                WHERE ("strategy" = '{strategy}') 
                AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        else:
            return

        # Return if no data is available
        if len(result) == 0:
            return

        # Process the MtM values
        df = result["mark_to_market_changes"]
        df['timems'] = df.index.view(int)
        df['MtM_value'].ffill(inplace=True)

        # Determine the time delta and rolling window size
        if t1 - t0 <= 1000 * 60 * 60:
            return
        elif t1 - t0 <= 1000 * 60 * 60 * 12:
            time_delta = int((t1 - t0) * 0.2)
            cmean = df['MtM_value'].rolling('20m', min_periods=1).mean()
            cmean['timems'] = cmean.index.view(int)
            cstd = df['MtM_value'].rolling('20m', min_periods=1).std()
            cstd['timems'] = cstd.index.view(int)
        else:
            time_delta = 1000 * 60 * 6
            cmean = df['MtM_value'].rolling('1h', min_periods=1).mean()
            cmean['timems'] = cmean.index.view(int)
            cstd = df['MtM_value'].rolling('1h', min_periods=1).std()
            cstd['timems'] = cstd.index.view(int)

        # Find local minima of std in the first period
        start_idx = cstd[cstd['timems'] >= t0, cstd['timems'] <= t0 + time_delta, 'MtM_value'].idxmin()
        start_value = cmean.loc[cmean.index == start_idx, 'MtM_value']
        start_time = cstd.loc[start_idx, 'timems']

        end_idx = cstd[cstd['timems'] >= t1 - time_delta, cstd['timems'] <= t1, 'MtM_value'].idxmin()
        end_value = cmean.loc[cmean.index == end_idx, 'MtM_value']
        end_time = cstd.loc[end_idx, 'timems']

        # Calculate PnL
        if not math.isnan(transfers):
            pnl = (end_value - start_value) - transfers
        else:
            pnl = end_value - start_value

        return {'pnl': pnl, 'start_value': start_value, 'start_time': start_time,
                'end_value': end_value, 'end_time': end_time}


class BackfillInverseQuantoProfit:
    """
    A class to backfill the Inverse Quanto Profit (IQP) data.

    This class retrieves historical IQP data for a specified strategy and time range, computes the IQP values, and
    writes them to an InfluxDB instance.

    Attributes:
    ----------
    influx_connection : InfluxConnection
        An instance of the InfluxConnection class for connecting to the InfluxDB database.
    """

    def __init__(self):
        """
        Initializes the BackfillInverseQuantoProfit class.

        Establishes a connection to the InfluxDB database using the InfluxConnection class.
        """
        self.influx_connection = InfluxConnection.getInstance()

    def inverse_quanto_profit(self, t0, t1, strategy):
        """
        Calculates the Inverse Quanto Profit (IQP) for a given strategy and time range.

        This method retrieves IQP data from the InfluxDB database, computes the IQP values using the
        fast_inverse_quanto_profit function, and writes the results to the database.

        Parameters:
        ----------
        t0 : int
            The start timestamp in milliseconds for calculating IQP.
        t1 : int
            The end timestamp in milliseconds for calculating IQP.
        strategy : str
            The trading strategy being evaluated.

        Example:
        -------
        >>> iqp = BackfillInverseQuantoProfit()
        >>> iqp.inverse_quanto_profit(1622548800000, 1622635200000, 'MyStrategy')
        """
        # Query IQP data
        past = time.time()
        df = get_quanto_profit(t0, t1, strategy)
        print(f'It took {time.time() - past} to query the data')

        # Process IQP data
        df['Time'] = df.index
        df['timems'] = df['Time'].astype(int) / 10 ** 6
        df.reset_index(drop=True, inplace=True)
        df['IQP'] = 0
        df['IQP_sum'] = 0

        # Query previous IQP values
        result = self.influx_connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "value" FROM "inverse_quanto_profit" 
            WHERE( "strategy" ='{strategy}')AND time >= {t0 - 1000 * 60 * 2}ms and time <= {t0}ms''',
            epoch='ns')

        if len(result) > 0 and len(result["inverse_quanto_profit"]) != 0 and result["inverse_quanto_profit"]['value'][
            -1] != 0:
            df.loc[0, 'IQP_sum'] = result["inverse_quanto_profit"]['value'][-1]
        max_qp = df.loc[0, 'Quanto profit per ETH']

        past = time.time()

        # Calculate IQP using fast_inverse_quanto_profit
        quanto_profits = np.array(df['Quanto profit per ETH'])
        iqps = np.array(df[['IQP', 'IQP_sum']])
        iqps = np.array(fast_inverse_quanto_profit(max_qp.astype(np.float64), quanto_profits.astype(np.float64),
                                                   iqps.astype(np.float64)))
        points = []

        # Prepare IQP data points for InfluxDB
        for ix in df.index:
            point = {
                'time': int(df.loc[ix, 'timems']),
                'measurement': 'inverse_quanto_profit',
                'tags': {'coin': 'ETH',
                         'strategy': strategy},
                'fields': {'value': iqps[ix, 1]}
            }
            points.append(point)

            if len(points) >= 10000:
                self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                points = []

        if len(points) != 0:
            self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')

        print(f'It took {time.time() - past} to compute stuff')


@numba.jit(nopython=True)
def fast_inverse_quanto_profit(max_qp, quanto_profits, iqps):
    """
    A Numba-optimized function to calculate Inverse Quanto Profit (IQP) values efficiently.

    This function iterates through the quanto profits array, updating the IQP and IQP_sum values based on changes in
    quanto profits.

    Parameters:
    ----------
    max_qp : float
        The maximum quanto profit observed so far.
    quanto_profits : np.ndarray
        An array of quanto profits for each timestamp.
    iqps : np.ndarray
        A 2D array containing IQP and IQP_sum values for each timestamp.

    Returns:
    -------
    np.ndarray
        The updated 2D array of IQP and IQP_sum values.

    Example:
    -------
    >>> max_qp = 1.5
    >>> quanto_profits = np.array([1.2, 1.4, 1.3])
    >>> iqps = np.zeros((len(quanto_profits), 2))
    >>> result = fast_inverse_quanto_profit(max_qp, quanto_profits, iqps)
    """
    for j in range(1, len(quanto_profits)):
        if quanto_profits[j] >= 1:
            if max_qp < quanto_profits[j]:
                max_qp = quanto_profits[j]
            elif max_qp > quanto_profits[j]:
                iqps[j, 0] = quanto_profits[j - 1] - quanto_profits[j]
                iqps[j, 1] = max(iqps[j - 1, 1] + iqps[j, 0], 0.0)
    return iqps


def backfill_maker_maker_evaluations(displacement, t0, t1, taker_slippage_spot=2.5, taker_slippage_swap=2.5,
                                     use_backblaze=True, use_wandb=True, to_influx=True):
    """
    A function to backfill maker-maker evaluations for a specified trading strategy.

    This function simulates trading activity for a maker-maker strategy, evaluates the quality of executions, and writes
    the results to an InfluxDB instance.

    Parameters:
    ----------
    displacement : float
        The displacement value used in the strategy.
    t0 : datetime.datetime
        The start datetime for the simulation.
    t1 : datetime.datetime
        The end datetime for the simulation.
    taker_slippage_spot : float, optional
        The slippage for taker orders on spot exchanges (default is 2.5).
    taker_slippage_swap : float, optional
        The slippage for taker orders on swap exchanges (default is 2.5).
    use_backblaze : bool, optional
        Flag to indicate whether to use Backblaze for storage (default is True).
    use_wandb : bool, optional
        Flag to indicate whether to use Weights & Biases for logging (default is True).
    to_influx : bool, optional
        Flag to indicate whether to write results to InfluxDB (default is True).

    Example:
    -------
    >>> backfill_maker_maker_evaluations(0.5, datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2))
    """
    lookback = None
    recomputation_time = None
    target_percentage_exit = None
    target_percentage_entry = None
    entry_opportunity_source = None
    exit_opportunity_source = None
    connection = InfluxConnection.getInstance()
    t_start = int(datetime.datetime(year=t0.year, month=t0.month, day=t0.day).timestamp() * 1000)
    t_end = int(datetime.datetime(year=t1.year, month=t1.month, day=t1.day).timestamp() * 1000)

    family = "deribit_xbtusd"
    environment = 'production'
    if family == 'Other':
        strategy = get_strategy_influx(environment=environment)
    elif family == 'deribit_xbtusd':
        strategy = get_strategy_families(t0=t_start, environment='production')[family][15]
    else:
        strategy = get_strategy_families(t0=t_start, environment='production')[family][0]

    strategy = "deribit_XBTUSD_maker_perpetual_3"

    if family == 'Other':
        exchange_spot = get_exhange_names(t0=t_start, t1=t_end, environment=environment)
        exchange_swap = get_exhange_names(t0=t_start, t1=t_end, environment=environment)
        spot_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot, environment=environment)[0]
        swap_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap, environment=environment)[-1]
    else:
        exchange_spot = 'Deribit'
        exchange_swap = 'BitMEX'
        spot_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot, environment=environment)[0]
        swap_instrument = get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap, environment=environment)[3]

    maker_fee_swap, taker_fee_swap = exchange_fees(exchange_swap, swap_instrument, exchange_swap, swap_instrument)
    maker_fee_spot, taker_fee_spot = exchange_fees(exchange_spot, spot_instrument, exchange_spot, spot_instrument)

    # Latencies default values
    ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(exchange_swap, exchange_spot)
    # Latencies
    latency_spot = ws_spot
    latency_try_post_spot = api_spot
    latency_cancel_spot = api_spot
    latency_balance_spot = api_swap
    latency_swap = ws_swap
    latency_try_post_swap = api_swap
    latency_cancel_swap = api_swap
    latency_balance_swap = api_spot

    displacement = displacement
    area_spread_threshold = 0

    max_trade_volume = 3000
    max_position = 110000

    file_id = random.randint(10 ** 6, 10 ** 7)

    # Convert milliseconds to datetime
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)

    params = {'t_start': t_start, 't_end': t_end, 'band': 'bogdan_bands',
              'lookback': lookback, 'recomputation_time': recomputation_time,
              'target_percentage_entry': target_percentage_entry, 'target_percentage_exit': target_percentage_exit,
              'entry_opportunity_source': entry_opportunity_source, 'exit_opportunity_source': exit_opportunity_source,
              'family': family, 'environment': environment, 'strategy': strategy, 'exchange_spot': exchange_spot,
              'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument, 'swap_instrument': swap_instrument,
              'taker_fee_spot': taker_fee_spot, 'maker_fee_spot': maker_fee_spot, 'taker_fee_swap': taker_fee_swap,
              'maker_fee_swap': maker_fee_swap, 'area_spread_threshold': area_spread_threshold,
              'latency_spot': latency_spot, 'latency_try_post_spot': latency_try_post_spot,
              'latency_cancel_spot': latency_cancel_spot, 'latency_balance_spot': latency_balance_spot,
              'latency_swap': latency_swap, 'latency_try_post_swap': latency_try_post_swap,
              'latency_cancel_swap': latency_cancel_swap, 'latency_balance_swap': latency_balance_swap,
              'taker_slippage_spot': taker_slippage_spot, 'taker_slippage_swap': taker_slippage_swap,
              'displacement': displacement,
              'max_trade_volume': max_trade_volume, 'max_position': max_position,
              'function': 'simulation_trader_maker_maker'}
    params_df = pd.DataFrame(params, index=[0])

    # Send message for simulation initialization
    now = datetime.datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
    data = {
        "message": f"Simulation (MM) of {strategy} from {date_start} to {date_end} Started at {dt_string_start} UTC",
    }

    band_values = get_entry_exit_bands(t0=t_start, t1=t_end, strategy=strategy, entry_delta_spread=0,
                                       exit_delta_spread=0, btype='central_band', environment=environment)
    band_values.rename(columns={'Band': 'Central Band'}, inplace=True)

    df = get_data_for_trader(t_start, t_end, exchange_spot, spot_instrument, exchange_swap, swap_instrument,
                             taker_fee_spot=taker_fee_spot, maker_fee_spot=maker_fee_spot,
                             taker_fee_swap=taker_fee_swap,
                             maker_fee_swap=maker_fee_swap, strategy=strategy,
                             area_spread_threshold=area_spread_threshold,
                             environment=environment)

    model = DisplacementEvaluationNoSpreadEntry(df=df, maker_fee_swap=maker_fee_swap, taker_fee_swap=taker_fee_swap,
                                                maker_fee_spot=maker_fee_spot, spot_instrument=spot_instrument,
                                                swap_instrument=swap_instrument, taker_fee_spot=taker_fee_spot,
                                                area_spread_threshold=area_spread_threshold, latency_spot=latency_spot,
                                                latency_swap=latency_swap, latency_try_post_spot=latency_try_post_spot,
                                                latency_try_post_swap=latency_try_post_swap,
                                                latency_cancel_spot=latency_cancel_spot,
                                                latency_cancel_swap=latency_cancel_swap,
                                                latency_balance_spot=latency_balance_spot,
                                                latency_balance_swap=latency_balance_swap, displacement=displacement,
                                                taker_slippage_spot=taker_slippage_spot,
                                                taker_slippage_swap=taker_slippage_swap, max_position=max_position,
                                                max_trade_volume=max_trade_volume, environment=environment)

    print("Starting simulation")
    while model.timestamp < t_end - 1000 * 60 * 5:
        for trigger in model.machine.get_triggers(model.state):
            if not trigger.startswith('to_'):
                if model.trigger(trigger):
                    break
    print("Done!")

    if len(model.executions_as_maker) == 0:
        simulated_executions_maker = pd.DataFrame(
            columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread", "central_band",
                     "was_trying_to_cancel_spot", "was_trying_to_cancel_swap", "source_at_execution_swap",
                     "dest_at_execution_swap", "source_at_execution_spot", "dest_at_execution_spot", "is_balancing",
                     "is_balancing_spot", "side"])
    else:
        simulated_executions_maker = pd.DataFrame(model.executions_as_maker)
    if len(model.executions_as_taker) == 0:
        model.executions_as_taker = pd.DataFrame(
            columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread", "central_band",
                     "was_trying_to_cancel_spot", "was_trying_to_cancel_swap", "source_at_execution_swap",
                     "dest_at_execution_swap", "source_at_execution_spot", "dest_at_execution_spot", "is_balancing",
                     "is_balancing_spot", "side"])
    else:
        simulated_executions_taker = pd.DataFrame(model.executions_as_taker)
    simulated_executions_taker['temp'] = simulated_executions_taker['executed_spread'] - simulated_executions_taker[
        'targeted_spread'] + to_decimal(simulated_executions_taker['displacement']) * simulated_executions_taker[
                                             'price']
    simulated_executions_maker['temp'] = simulated_executions_maker['executed_spread'] - simulated_executions_maker[
        'targeted_spread'] + to_decimal(simulated_executions_maker['displacement']) * simulated_executions_taker[
                                             'price']

    maker_maker_points = []
    for _, row in simulated_executions_maker.iterrows():
        maker_maker_points.append({
            'time': row['timems'],
            'measurement': 'evaluation_without_spread',
            'tags': {
                'spot_exchange': exchange_spot,
                'swap_exchange': exchange_swap,
                'spot_instrument': spot_instrument,
                'swap_instrument': swap_instrument,
                'entry_exit': row['side'],
                'did_balance': False,
                'displacement': float(row['displacement'])
            },
            'fields': {
                'displacement': float(row['displacement']),
                'r': float(row['r']),
                'executed_spread': float(row['executed_spread']),
                'targeted_spread': float(row['targeted_spread']),
                'timestamp_swap_executed': row['timestamp_swap_executed'],
                'timestamp_spot_executed': row['timestamp_spot_executed'],
                'price': row['price']
            }
        })
        if len(maker_maker_points) > 10000:
            if to_influx:
                connection.staging_client_spotswap.write_points(maker_maker_points, time_precision='ms')
            maker_maker_points = []
    if to_influx:
        connection.staging_client_spotswap.write_points(maker_maker_points, time_precision='ms')
    taker_maker_points = []
    for _, row in simulated_executions_taker.iterrows():
        taker_maker_points.append({
            'time': row['timems'],
            'measurement': 'evaluation_without_spread',
            'tags': {
                'spot_exchange': exchange_spot,
                'swap_exchange': exchange_swap,
                'spot_instrument': spot_instrument,
                'swap_instrument': swap_instrument,
                'entry_exit': row['side'],
                'did_balance': True,
                'displacement': float(row['displacement'])
            },
            'fields': {
                'displacement': float(row['displacement']),
                'r': float(row['r']),
                'executed_spread': float(row['executed_spread']),
                'targeted_spread': float(row['targeted_spread']),
                'timestamp_swap_executed': row['timestamp_swap_executed'],
                'timestamp_spot_executed': row['timestamp_spot_executed'],
                'price': row['price']
            }
        })
        if len(taker_maker_points) > 10000:
            if to_influx:
                connection.staging_client_spotswap.write_points(taker_maker_points, time_precision='ms')
            taker_maker_points = []
    if to_influx:
        connection.staging_client_spotswap.write_points(taker_maker_points, time_precision='ms')