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
    connection = InfluxConnection.getInstance()

    price1 = get_price(t_start=t_start, t_end=t_end, exchange='Deribit', symbol='ETH-PERPETUAL', side='Ask',
                       environment='production')
    price2 = get_price(t_start=t_start, t_end=t_end, exchange='BitMEX', symbol='XBTUSD', side='Ask',
                       environment='production')

    price1['Time'] = price1.index

    price2['Time'] = price2.index

    # merge the price dataframes
    price_ask = pd.merge_ordered(price1, price2, on='Time', suffixes=['_ETH', '_BTC'])

    price_ask.index = price_ask.Time
    price_ask.drop(columns=['Time'], inplace=True)
    price_ask = price_ask.resample('10s').mean()

    price_corr = {}
    for ws in ['2h', '4h', '8h']:
        price_corr[f'{ws}'] = price_ask.rolling(ws).corr().iloc[:, 0]
        price_corr[f'{ws}'].reindex(level=0, copy=True)

    price_corr_df = pd.merge(price_corr['2h'], price_corr['4h'], left_index=True, right_index=True)
    price_corr_df = pd.merge(price_corr_df, price_corr['8h'], left_index=True, right_index=True)
    price_corr_df = price_corr_df.unstack()
    price_corr_df.drop(columns=price_corr_df.columns[[1, 3, 5]], inplace=True)
    price_corr_df = price_corr_df.resample('5T').mean()
    price_corr_df.rename(columns={price_corr_df.columns[0]: '2h', price_corr_df.columns[1]: '4h',
                                  price_corr_df.columns[2]: '8h'}, inplace=True)
    price_corr_df['timestamp'] = price_corr_df.index.view(int) // 10 ** 6
    price_corr_df.dropna(inplace=True)
    price_corr_df.reset_index(drop=True, inplace=True)

    points = []

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

        if len(points) >= 1000:
            connection.staging_client_spotswap.write_points(points, time_precision="ms")
            points = []

    connection.staging_client_spotswap.write_points(points, time_precision="ms")


class BackfillCVI:
    """
    Backfill C(?)
    Volatility Index
    """

    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def cvi_volatility_index(self, t_end):
        df = requests.get('https://api.dev-cvi-finance-route53.com/history?chain=Ethereum&index=CVI',
                          params={'Date': f'{t_end}'})

        cvi = np.array(df.json())
        points = []

        # print(max_p)

        for ix in range(len(cvi)):
            point = {
                'time': int(cvi[ix, 0] * 1000),
                'measurement': 'dvol',
                'tags': {'coin': 'CVI'},
                'fields': {'close': cvi[ix, 1]}
            }
            points.append(point)

        self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
        # print(points)
        # print(len(points))


class QualityOfExecutions:
    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def backfill(self, t0, t1, strategy, entry_delta_spread, exit_delta_spread, window_size, environment):

        client = self.influx_connection.prod_client_spotswap_dataframe if environment == 'production' else self.influx_connection.staging_client_spotswap_dataframe
        write = self.influx_connection.prod_client_spotswap if environment == 'production' else self.influx_connection.staging_client_spotswap
        # entry_delta_spread and exit_delta_spread are in postgres

        # the bands
        result = client.query(f'''SELECT "spread", type, 
        volume_executed_spot  FROM "executed_spread" WHERE ("strategy" = '{strategy}') AND time >= {t0}ms AND time <= 
        {t1}ms ''', epoch='ns')
        if len(result) == 0:
            return

        executions = result["executed_spread"]
        executions['Time'] = executions.index
        executions['entry_executions'] = executions[executions.type == 'entry']['spread']
        executions['exit_executions'] = executions[executions.type == 'exit']['spread']
        executions['entry_volume'] = executions[executions.type == 'entry']['volume_executed_spot']
        executions['exit_volume'] = executions[executions.type == 'exit']['volume_executed_spot']

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

        entry_exit_exec = pd.merge_ordered(executions, bands, on='Time')
        entry_exit_exec['Entry Band'].ffill(inplace=True)
        entry_exit_exec['Exit Band'].ffill(inplace=True)
        entry_exit_exec['timestamp'] = entry_exit_exec['Time'].astype(int) / 10 ** 6
        entry_exit_exec.reset_index(drop=True, inplace=True)
        points = []

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

            if len(points) > 10000:
                write.write_points(points, time_precision='ms')
                points = []

        write.write_points(points, time_precision='ms')


class BackfillDeribitVolatilityIndex:

    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def deribit_volatility(self, t_start, t_end):
        df = requests.get('https://www.deribit.com/api/v2/public/get_volatility_index_data',
                          params={'currency': "BTC",
                                  'start_timestamp': f"{t_start}",
                                  'resolution': 60,
                                  'end_timestamp': f"{t_end}"})

        dvol_btc = np.array(df.json()['result']['data'])

        dff = requests.get('https://www.deribit.com/api/v2/public/get_volatility_index_data',
                           params={'currency': "ETH",
                                   'start_timestamp': f"{t_start}",
                                   'resolution': 60,
                                   'end_timestamp': f"{t_end}"})

        dvol_eth = np.array(dff.json()['result']['data'])
        points = []

        max_p = max(len(dvol_btc), len(dvol_eth))
        # print(max_p)

        for ix in range(max_p):
            # print(dvol_eth[ix,0])
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
        if env == "staging":
            client = self.influx_connection.staging_client_spotswap
        else:
            client = self.influx_connection.prod_client_spotswap

        if t1 - t0 <= 1000 * 60 * 1000:
            points = self.deribit_volatility(t_start=t0, t_end=t1)
            client.write_points(points, time_precision='ms')
        else:
            t_start = t0
            t_end = t_start + 1000 * 60 * 1000

            while t_end <= t1:
                time.sleep(0.2)
                points = self.deribit_volatility(t_start=t_start, t_end=t_end)
                client.write_points(points, time_precision='ms')
                t_start = t_start + 1000 * 60 * 1000
                t_end = t_end + 1000 * 60 * 1000
                # print(f'start_time: {t_start} end time: {t_end}')
        # start = t0
        # while start <= t1:
        #     points = self.deribit_volatility(t0=start, t1=t1)
        #     self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
        #     start = points[-1]['time']


class StrategyPNL:
    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def strategy_pnl(self, t0, t1, strategy, transfers=0, environment='production'):
        # define the place of the server
        if environment == 'production':
            result = self.influx_connection.prod_client_spotswap_dataframe.query(f'''SELECT "current_value" AS "MtM_value" 
                                                                FROM "mark_to_market_changes" 
                                                                WHERE ("strategy" = '{strategy}') 
                                                                AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        elif environment == 'staging':
            result = self.influx_connection.staging_client_spotswap_dataframe.query(f'''SELECT "current_value" AS "MtM_value" 
                                                                                        FROM "mark_to_market_changes" 
                                                                                        WHERE ("strategy" = '{strategy}') 
                                                                                        AND time >= {t0}ms and time <= {t1}ms''',
                                                                                    epoch='ns')
        else:
            return

        # if there are no values in this time interval then return
        if len(result) == 0:
            return

        df = result["mark_to_market_changes"]
        df['timems'] = df.index.view(int)
        df['MtM_value'].ffill(inplace=True)

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

        # find local minima of std in first period
        start_idx = cstd[cstd['timems'] >= t0, cstd['timems'] <= t0 + time_delta, 'MtM_value'].idxmin()
        start_value = cmean.loc[cmean.index == start_idx, 'MtM_value']
        start_time = cstd.loc[start_idx, 'timems']

        end_idx = cstd[cstd['timems'] >= t1 - time_delta, cstd['timems'] <= t1, 'MtM_value'].idxmin()
        end_value = cmean.loc[cmean.index == end_idx, 'MtM_value']
        end_time = cstd.loc[end_idx, 'timems']

        if not math.isnan(transfers):
            pnl = (end_value - start_value) - transfers
        else:
            pnl = end_value - start_value

        return {'pnl': pnl, 'start_value': start_value, 'start_time': start_time,
                'end_value': end_value, 'end_time': end_time}


class BackfillInverseQuantoProfit:

    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def inverse_quanto_profit(self, t0, t1, strategy):

        past = time.time()
        df = get_quanto_profit(t0, t1, strategy)
        print(f'It took {time.time() - past} to query the data')
        df['Time'] = df.index
        df['timems'] = df['Time'].astype(int) / 10 ** 6
        df.reset_index(drop=True, inplace=True)
        df['IQP'] = 0
        df['IQP_sum'] = 0

        result = self.influx_connection.prod_client_spotswap_dataframe.query(f'''SELECT "value" FROM "inverse_quanto_profit" 
                               WHERE( "strategy" ='{strategy}')AND time >= {t0 - 1000 * 60 * 2}ms and time <= {t0}ms''',
                                                                             epoch='ns')

        if len(result) > 0 and len(result["inverse_quanto_profit"]) != 0 and result["inverse_quanto_profit"]['value'][
            -1] != 0:
            df.loc[0, 'IQP_sum'] = result["inverse_quanto_profit"]['value'][-1]
        max_qp = df.loc[0, 'Quanto profit per ETH']

        past = time.time()

        quanto_profits = np.array(df['Quanto profit per ETH'])
        iqps = np.array(df[['IQP', 'IQP_sum']])
        # iqps_sum = np.array(df['IQP_sum'])
        iqps = np.array(fast_inverse_quanto_profit(max_qp.astype(np.float64), quanto_profits.astype(np.float64),
                                                   iqps.astype(np.float64)))
        #
        # for idx_p, idx_c in zip(df.index[:-1], df.index[1:]):
        #     if max_qp < df.loc[idx_c, 'Quanto profit per ETH'] and df.loc[idx_c, 'Quanto profit per ETH'] >= 1:
        #         max_qp = df.loc[idx_c, 'Quanto profit per ETH']
        #
        #     elif max_qp > df.loc[idx_c, 'Quanto profit per ETH'] and df.loc[idx_c, 'Quanto profit per ETH'] >= 1:
        #         df.loc[idx_c, 'IQP'] = df.loc[idx_p, 'Quanto profit per ETH'] - df.loc[idx_c, 'Quanto profit per ETH']
        #         df.loc[idx_c, 'IQP_sum'] = df.loc[idx_p, 'IQP_sum'] + df.loc[idx_c, 'IQP']
        #         if df.loc[idx_c, 'IQP_sum'] < 0:
        #             df.loc[idx_c, 'IQP_sum'] = 0
        points = []

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
                # pass
                self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                points = []

        if len(points) != 0:
            # pass
            # print(points)
            self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')

        print(f'It took {time.time() - past} to compute stuff')


@numba.jit(nopython=True)
def fast_inverse_quanto_profit(max_qp, quanto_profits, iqps):
    for j in range(1, len(quanto_profits)):
        if quanto_profits[j] >= 1:
            if max_qp < quanto_profits[j]:
                max_qp = quanto_profits[j]
            elif max_qp > quanto_profits[j]:
                iqps[j, 0] = quanto_profits[j - 1] - quanto_profits[j]
                iqps[j, 1] = max(iqps[j - 1, 1] + iqps[j, 0], 0.0)
                # if iqps_sum[j] < 0:
                #     iqps_sum[j] = 0
    return iqps


def backfill_maker_maker_evaluations(displacement, t0, t1, taker_slippage_spot=2.5, taker_slippage_swap=2.5,
                                     use_backblaze=True, use_wandb=True,
                                     to_influx=True):
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

    # latencies default values
    ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(exchange_swap, exchange_spot)
    # latencies
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

    # convert milliseconds to datetime
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
    # send message for simulation initialization
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
    # fig = plt.scatter(simulated_executions_maker['targeted_spread'], simulated_executions_maker['temp'])
    # plt.title(f"Displacement {displacement}")
    # plt.show()
    # fig = plt.scatter(simulated_executions_maker['targeted_spread'], simulated_executions_maker['r'])
    # plt.title(f"Displacement {displacement}")
    # plt.show()
    # fig = plt.scatter(simulated_executions_taker['targeted_spread'], simulated_executions_taker['temp'])
    # plt.title(f"Displacement {displacement}")
    # plt.show()
    # fig = plt.scatter(simulated_executions_taker['targeted_spread'], simulated_executions_taker['r'])
    # plt.title(f"Displacement {displacement}")
    # plt.show()
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
