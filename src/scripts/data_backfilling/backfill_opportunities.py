import warnings
import time
import math
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import numpy as np
from tqdm import tqdm
from src.common.queries.queries import Funding, Prices, Takers
from src.common.queries.queries import get_entry_opportunity_points, get_exit_opportunity_points
from src.common.connections.ExchangeConnections import Deribit, BitMEX
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.utils.utils import aggregated_volume

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")



class DiffOfOpportunityPoints:
    def __init__(self):
        self.influx_connection = InfluxConnection.getInstance()

    def backfill(self, strategy, t0, t1, environment):

        client = self.influx_connection.prod_client_spotswap_dataframe \
            if environment == 'production' else self.influx_connection.staging_client_spotswap_dataframe
        write = self.influx_connection.prod_client_spotswap \
            if environment == 'production' else self.influx_connection.staging_client_spotswap
        # entry_delta_spread and exit_delta_spread are in postgres

        # the bands
        opp_points_entry = get_entry_opportunity_points(t0, t1, exchange='Deribit', spot='hybrid_ETH-PERPETUAL',
                                                        swap='hybrid_ETHUSD')
        opp_points_exit = get_exit_opportunity_points(t0, t1, exchange='Deribit', spot='hybrid_ETH-PERPETUAL',
                                                      swap='hybrid_ETHUSD')

        result2 = client.query(f'''SELECT "value","side" FROM "band" 
            WHERE ("strategy" ='{strategy}' AND "type" = 'bogdan_bands') 
            AND time >= {t0 - 60000}ms and time <= {t1}ms''', epoch='ns')

        bands = result2["band"]
        bands['Time'] = bands.index
        bands['Entry Band'] = bands.loc[bands['side'] == 'entry', 'value']
        bands['Exit Band'] = bands.loc[bands['side'] == 'exit', 'value']
        bands.drop(columns=['side', 'value'], inplace=True)

        df_opp = pd.merge_ordered(opp_points_entry, opp_points_exit, on='Time')
        df = pd.merge_ordered(bands, df_opp, on='Time')
        df['Entry Band'].ffill(inplace=True)
        df['Exit Band'].ffill(inplace=True)
        df['timestamp'] = df['Time'].astype(int) / 10 ** 6

        points = []

        for ix in df.index:

            if math.isnan(df['Exit Opportunity'].loc[ix]) == False:
                diff_ext_opp = -df['Exit Opportunity'].loc[ix] + df['Exit Band'].loc[ix]

                point = {
                    'time': int(df.loc[ix, 'timestamp']),
                    'measurement': 'opportunity_band_diff',
                    'tags': {'strategy': strategy, 'type': '1', 'band': 'bogdan_band'},
                    'fields': {'diff_band': diff_ext_opp}
                }
                points.append(point)

            if math.isnan(df['Exit Opportunity_takers'].loc[ix]) == False:
                diff_ext_opp = -df['Exit Opportunity_takers'].loc[ix] + df['Exit Band'].loc[ix]

                point = {
                    'time': int(df.loc[ix, 'timestamp']),
                    'measurement': 'opportunity_band_diff',
                    'tags': {'strategy': strategy, 'type': 'exit_with_takers', 'band': 'bogdan_band'},
                    'fields': {'diff_band': diff_ext_opp}
                }
                points.append(point)

            if math.isnan(df['Exit Opportunity_takers_lat'].loc[ix]) == False:
                diff_ext_opp = -df['Exit Opportunity_takers_lat'].loc[ix] + df['Exit Band'].loc[ix]

                point = {
                    'time': int(df.loc[ix, 'timestamp']),
                    'measurement': 'opportunity_band_diff',
                    'tags': {'strategy': strategy, 'type': 'exit_with_takers_latency_200', 'band': 'bogdan_band'},
                    'fields': {'diff_band': diff_ext_opp}
                }
                points.append(point)

            if len(points) > 10000:
                write.write_points(points, time_precision='ms')
                points = []

        write.write_points(points, time_precision='ms')


class BackfillOpportunityPoints:

    def __init__(self, server_place='production', swap_symbol="XBTUSD", swap_market="BitMEX",
                 spot_symbol="BTC-PERPETUAL", spot_market="Deribit", spot_fee=0.0003, swap_fee=-0.0001):
        self.swapSymbol = swap_symbol
        self.spotSymbol = spot_symbol
        self.spotMarket = spot_market
        self.swapMarket = swap_market
        self.spotFee = spot_fee
        self.swapFee = swap_fee
        self.server = server_place

        self.influx_connection = InfluxConnection.getInstance()
        if self.server == 'production':
            self.swap_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        elif self.server == 'staging':
            self.swap_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        else:
            return

        if self.swapMarket == 'HuobiDMSwap':
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe, ['HuobiDM'],
                                              [self.swapSymbol])
        elif self.swapMarket == 'Okex':
            self.swap_takers_querier = Takers(self.influx_connection.staging_client_spotswap_dataframe,
                                              [self.swapMarket],
                                              [self.swapSymbol])
        elif self.swapMarket == 'Deribit':
            self.swap_takers_querier = Takers(self.influx_connection.staging_client_spotswap_dataframe,
                                              [self.swapMarket],
                                              [self.swapSymbol])
        else:
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe,
                                              [self.swapMarket], [self.swapSymbol])

    def get_taker_trades(self, t0, t1):
        # return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        if self.swapMarket == 'BitMEX':
            try:
                return BitMEX.historical_taker_trades(t0, t1, self.swapSymbol)
            except:
                return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        elif self.swapMarket == 'Deribit':
            try:
                df = Deribit.historical_taker_trades(t0, t1, self.swapSymbol)
                if df.empty:
                    return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
                else:
                    return df
            except:
                return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        else:
            return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)

    def backfill(self, t0, t1, latency=0):
        taker_trades = self.get_taker_trades(t0, t1)
        trades_buy = taker_trades[taker_trades['side'] == "Bid"]
        trades_sell = taker_trades[taker_trades['side'] == "Ask"]
        prices = self.swap_price_querier.query_data(t0, t1).get_data(t0, t1)
        spot_prices = self.spot_price_querier.query_data(t0, t1).get_data(t0, t1)

        prices_ask = prices[prices['side'] == 'Ask']
        prices_bid = prices[prices['side'] == 'Bid']
        prices_ask['diff'] = prices_ask['price'].diff().fillna(0)
        prices_bid['diff'] = prices_bid['price'].diff().fillna(0)
        spot_prices_ask = spot_prices[spot_prices['side'] == 'Ask']
        spot_prices_bid = spot_prices[spot_prices['side'] == 'Bid']

        points = []

        for ix, (timestamp, row) in tqdm(enumerate(prices_ask.iterrows())):
            if row['diff'] <= 0:
                continue
            swap_price = prices_ask.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_buy['time'], row['time'], side="left") - 1)
            if trades_buy['time'].iloc[sell_ix] < prices_ask['time'].iloc[ix - 1]:
                continue
            if trades_buy.iloc[sell_ix]['price'] < swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_ask['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_ask.iloc[spot_price_index]['price']
            spread = swap_price * (1 - self.swapFee) - (spot_price + spot_price * self.spotFee)

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities',
                'tags': {
                    'exchangeName': self.spotMarket,
                    'spotInstrument': f'hybrid_{self.spotSymbol}',
                    'swapInstrument': f'hybrid_{self.swapSymbol}',
                    'type': f'entry_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)
            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        for ix, (timestamp, row) in tqdm(enumerate(prices_bid.iterrows())):
            if row['diff'] >= 0:
                continue
            swap_price = prices_bid.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_sell['time'], row['time'], side="left") - 1)
            if trades_sell['time'].iloc[sell_ix] < prices_bid['time'].iloc[ix - 1]:
                continue
            if trades_sell.iloc[sell_ix]['price'] > swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_bid['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_bid.iloc[spot_price_index]['price']
            spread = swap_price * (1 + self.swapFee) - (spot_price - spot_price * self.spotFee)

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities',
                'tags': {
                    'exchangeName': self.spotMarket,
                    'spotInstrument': f'hybrid_{self.spotSymbol}',
                    'swapInstrument': f'hybrid_{self.swapSymbol}',
                    'type': f'exit_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)

            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        if self.server == 'production':
            self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
        elif self.server == 'staging':
            self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
            time.sleep(5)


class BackfillOpportunityPointsBps:

    def __init__(self, server_place='production', swap_symbol="XBTUSD", swap_market="BitMEX",
                 spot_symbol="BTC-PERPETUAL", spot_market="Deribit", spot_fee=0.0003, swap_fee=-0.0001,
                 skip_already_filled=True):
        self.swapSymbol = swap_symbol
        self.spotSymbol = spot_symbol
        self.spotMarket = spot_market
        self.swapMarket = swap_market
        self.spotFee = spot_fee
        self.swapFee = swap_fee
        self.server = server_place
        self.skip_already_filled = skip_already_filled

        self.influx_connection = InfluxConnection.getInstance()
        if self.server == 'production':
            self.swap_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.prod_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        elif self.server == 'staging':
            self.swap_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.swapMarket,
                                             self.swapSymbol)
            self.spot_price_querier = Prices(self.influx_connection.staging_client_spotswap_dataframe, self.spotMarket,
                                             self.spotSymbol)
        else:
            return

        if self.swapMarket == 'Okex':
            self.swap_takers_querier = Takers(self.influx_connection.staging_client_spotswap_dataframe,
                                              [self.swapMarket],
                                              [self.swapSymbol])
        else:
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe,
                                              [self.swapMarket], [self.swapSymbol])

    def get_taker_trades(self, t0, t1):
        # return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        # if self.swapMarket == 'BitMEX':
        #     try:
        #         return BitMEX.historical_taker_trades(t0, t1, self.swapSymbol)
        #     except:
        #         return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        # else:
        return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)

    def backfill(self, t0, t1, latency=0):
        existing_opportunity_points = self.influx_connection.staging_client_spotswap_dataframe.query(
            "select count(opportunity) as count from trading_opportunities_bps " + \
            f"where spotMarket = '{self.spotMarket}' and swapMarket = '{self.swapMarket}' " + \
            f"and spotInstrument = '{self.spotSymbol}' and swapInstrument = '{self.swapSymbol}' " + \
            f"and time > {t0}ms and time < {t1}ms group by time(6h) "
        )
        existing_opportunity_points = existing_opportunity_points.get('trading_opportunities_bps',
                                                                      pd.DataFrame(columns=["count"]))
        existing_opportunity_points = existing_opportunity_points.reset_index()
        if self.skip_already_filled and len(
                existing_opportunity_points.loc[existing_opportunity_points['count'] == 0]) == 0 and len(
                existing_opportunity_points) > 0:
            print(f"Combination {self.swapSymbol}/{self.spotSymbol} already fully backfilled.")
            return
        taker_trades = self.get_taker_trades(t0, t1)
        if taker_trades is None or len(taker_trades) == 0:
            return
        trades_buy = taker_trades[taker_trades['side'] == "Bid"]
        trades_sell = taker_trades[taker_trades['side'] == "Ask"]
        prices = self.swap_price_querier.query_data(t0, t1).get_data(t0, t1)
        spot_prices = self.spot_price_querier.query_data(t0, t1).get_data(t0, t1)
        if prices is None or spot_prices is None or len(prices) == 0 or len(spot_prices) == 0:
            return

        prices_ask = prices[prices['side'] == 'Ask']
        prices_bid = prices[prices['side'] == 'Bid']
        prices_ask['diff'] = prices_ask['price'].diff().fillna(0)
        prices_bid['diff'] = prices_bid['price'].diff().fillna(0)
        spot_prices_ask = spot_prices[spot_prices['side'] == 'Ask']
        spot_prices_bid = spot_prices[spot_prices['side'] == 'Bid']

        points = []

        for ix, (timestamp, row) in tqdm(enumerate(prices_ask.iterrows())):
            if self.skip_already_filled and len(existing_opportunity_points) > 0:
                opportunities = existing_opportunity_points.iloc[
                    existing_opportunity_points['index'].searchsorted(timestamp) - 1]
                if opportunities['count'] > 0:
                    continue
            if row['diff'] <= 0:
                continue
            swap_price = prices_ask.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_buy['time'], row['time'], side="left") - 1)
            if trades_buy['time'].iloc[sell_ix] < prices_ask['time'].iloc[ix - 1]:
                continue
            if trades_buy.iloc[sell_ix]['price'] < swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_ask['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_ask.iloc[spot_price_index]['price']
            spread = (swap_price * (1 - self.swapFee) - (spot_price + spot_price * self.spotFee)) / swap_price * 10000

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities_bps',
                'tags': {
                    'spotMarket': self.spotMarket,
                    'swapMarket': self.swapMarket,
                    'spotInstrument': f'{self.spotSymbol}',
                    'swapInstrument': f'{self.swapSymbol}',
                    'type': f'entry_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)
            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        for ix, (timestamp, row) in tqdm(enumerate(prices_bid.iterrows())):
            if self.skip_already_filled and len(existing_opportunity_points) > 0:
                opportunities = existing_opportunity_points.iloc[
                    existing_opportunity_points['index'].searchsorted(timestamp) - 1]
                if opportunities['count'] > 0:
                    continue
            if row['diff'] >= 0:
                continue
            swap_price = prices_bid.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_sell['time'], row['time'], side="left") - 1)
            if trades_sell['time'].iloc[sell_ix] < prices_bid['time'].iloc[ix - 1]:
                continue
            if trades_sell.iloc[sell_ix]['price'] > swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_bid['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            spot_price = spot_prices_bid.iloc[spot_price_index]['price']
            spread = (swap_price * (1 + self.swapFee) - (spot_price - spot_price * self.spotFee)) / swap_price * 10000

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities_bps',
                'tags': {
                    'spotMarket': self.spotMarket,
                    'swapMarket': self.swapMarket,
                    'spotInstrument': f'{self.spotSymbol}',
                    'swapInstrument': f'{self.swapSymbol}',
                    'type': f'exit_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)

            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        if self.server == 'production':
            self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
        elif self.server == 'staging':
            self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
            time.sleep(5)


class BackfillOpportunityPointsBpsFunding:

    def __init__(self, server_place='production', swap_symbol="XBTUSD", swap_market="BitMEX",
                 spot_symbol="BTC-PERPETUAL", spot_market="Deribit", spot_fee=0.0003, swap_fee=-0.0001):
        self.swapSymbol = swap_symbol
        self.spotSymbol = spot_symbol
        self.spotMarket = spot_market
        self.swapMarket = swap_market
        self.spotFee = spot_fee
        self.swapFee = swap_fee
        self.server = server_place

        self.influx_connection = InfluxConnection.getInstance()

        self.influx_client = self.influx_connection.staging_client_spotswap_dataframe
        if self.server == 'production':
            self.influx_client = self.influx_connection.prod_client_spotswap_dataframe

        self.swap_price_querier = Prices(self.influx_client, self.swapMarket, self.swapSymbol)
        self.spot_price_querier = Prices(self.influx_client, self.spotMarket, self.spotSymbol)
        self.spot_funding_querier = Funding(self.influx_client, self.spotMarket, self.spotSymbol)
        self.swap_funding_querier = Funding(self.influx_client, self.swapMarket, self.swapSymbol)

        if self.swapMarket == 'Okex':
            self.swap_takers_querier = Takers(self.influx_connection.staging_client_spotswap_dataframe,
                                              [self.swapMarket],
                                              [self.swapSymbol])
        else:
            self.swap_takers_querier = Takers(self.influx_connection.archival_client_spotswap_dataframe,
                                              [self.swapMarket], [self.swapSymbol])

    def get_taker_trades(self, t0, t1):
        # return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        if self.swapMarket == 'BitMEX':
            try:
                return BitMEX.historical_taker_trades(t0, t1, self.swapSymbol)
            except:
                return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
        else:
            return self.swap_takers_querier.query_data(t0, t1).get_data(t0, t1)

    def backfill(self, t0, t1, latency=0):
        taker_trades = self.get_taker_trades(t0, t1)
        if taker_trades is None or len(taker_trades) == 0:
            return
        trades_buy = taker_trades[taker_trades['side'] == "Bid"]
        trades_sell = taker_trades[taker_trades['side'] == "Ask"]
        prices = self.swap_price_querier.query_data(t0, t1).get_data(t0, t1)
        spot_prices = self.spot_price_querier.query_data(t0, t1).get_data(t0, t1)
        if prices is None or spot_prices is None or len(prices) == 0 or len(spot_prices) == 0:
            return

        t1_plus_8h = t1 + 1000 * 60 * 60 * 8
        funding_swap = self.swap_funding_querier.query_data(t0, t1_plus_8h).get_data(t0, t1_plus_8h)
        funding_spot = self.spot_funding_querier.query_data(t0, t1_plus_8h).get_data(t0, t1_plus_8h)
        if funding_swap is None or funding_spot is None or len(funding_spot) == 0 or len(funding_swap) == 0:
            return
        prices_ask = prices[prices['side'] == 'Ask']
        prices_bid = prices[prices['side'] == 'Bid']
        prices_ask['diff'] = prices_ask['price'].diff().fillna(0)
        prices_bid['diff'] = prices_bid['price'].diff().fillna(0)
        spot_prices_ask = spot_prices[spot_prices['side'] == 'Ask']
        spot_prices_bid = spot_prices[spot_prices['side'] == 'Bid']

        points = []

        for ix, (timestamp, row) in tqdm(enumerate(prices_ask.iterrows())):
            if row['diff'] <= 0:
                continue
            swap_price = prices_ask.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_buy['time'], row['time'], side="left") - 1)
            if trades_buy['time'].iloc[sell_ix] < prices_ask['time'].iloc[ix - 1]:
                continue
            if trades_buy.iloc[sell_ix]['price'] < swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_ask['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            swap_funding_ix = funding_swap['time'].searchsorted(timestamp)
            spot_funding_ix = funding_spot['time'].searchsorted(timestamp)
            if swap_funding_ix >= len(funding_swap) or spot_funding_ix >= len(funding_spot):
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                break
            swap_funding = funding_swap.iloc[swap_funding_ix]['funding']
            spot_funding = funding_spot.iloc[spot_funding_ix]['funding']
            funding_diff = (swap_funding - spot_funding) * 10000
            spot_price = spot_prices_ask.iloc[spot_price_index]['price']
            spread = (swap_price * (1 - self.swapFee) - (
                        spot_price + spot_price * self.spotFee)) / swap_price * 10000 + funding_diff

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities_bps_funding',
                'tags': {
                    'spotMarket': self.spotMarket,
                    'swapMarket': self.swapMarket,
                    'spotInstrument': f'{self.spotSymbol}',
                    'swapInstrument': f'{self.swapSymbol}',
                    'type': f'entry_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)
            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        for ix, (timestamp, row) in tqdm(enumerate(prices_bid.iterrows())):
            if row['diff'] >= 0:
                continue
            swap_price = prices_bid.iloc[ix - 1]['price']
            sell_ix = max(0, np.searchsorted(trades_sell['time'], row['time'], side="left") - 1)
            if trades_sell['time'].iloc[sell_ix] < prices_bid['time'].iloc[ix - 1]:
                continue
            if trades_sell.iloc[sell_ix]['price'] > swap_price:
                continue
            spot_price_index = max(0, np.searchsorted(spot_prices_bid['time'], row['time'] +
                                                      pd.Timedelta(milliseconds=latency), side="left") - 1)

            swap_funding_ix = funding_swap['time'].searchsorted(timestamp)
            spot_funding_ix = funding_spot['time'].searchsorted(timestamp)
            if swap_funding_ix >= len(funding_swap) or spot_funding_ix >= len(funding_spot):
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                break
            swap_funding = funding_swap.iloc[swap_funding_ix]['funding']
            spot_funding = funding_spot.iloc[spot_funding_ix]['funding']
            funding_diff = (swap_funding - spot_funding) * 10000
            spot_price = spot_prices_bid.iloc[spot_price_index]['price']
            spread = (swap_price * (1 + self.swapFee) - (
                        spot_price - spot_price * self.spotFee)) / swap_price * 10000 + funding_diff

            point = {
                'time': int(row['timems']),
                'measurement': 'trading_opportunities_bps_funding',
                'tags': {
                    'spotMarket': self.spotMarket,
                    'swapMarket': self.swapMarket,
                    'spotInstrument': f'{self.spotSymbol}',
                    'swapInstrument': f'{self.swapSymbol}',
                    'type': f'exit_with_takers{"" if latency == 0 else "_latency_" + str(latency)}'
                },
                'fields': {'opportunity': spread}
            }
            points.append(point)

            if len(points) > 1000:
                if self.server == 'production':
                    self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                elif self.server == 'staging':
                    self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                    points = []
                    time.sleep(5)

        if self.server == 'production':
            self.influx_connection.prod_client_spotswap.write_points(points, time_precision='ms')
        elif self.server == 'staging':
            self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
            time.sleep(5)


class BackfillMakerMakerOpportunityPoints:

    def __init__(self, swap_symbol="XBTUSD", swap_market="BitMEX", spot_symbol="BTC-PERPETUAL", spot_market="Deribit",
                 spot_fee=-0.0001, swap_fee=-0.0001, latency_spot=0, latency_swap=0):
        self.swapSymbol = swap_symbol
        self.spotSymbol = spot_symbol
        self.spotMarket = spot_market
        self.swapMarket = swap_market
        self.spotFee = spot_fee
        self.swapFee = swap_fee
        self.latency_spot = latency_spot
        self.latency_swap = latency_swap
        self.influx_connection = InfluxConnection.getInstance()

    def backfill(self, t0, t1, volume=1000):

        ################################################################################################################
        # Import taker data
        ################################################################################################################
        result1 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.swapMarket}' AND "symbol" = '{self.swapSymbol}' 
                AND "side"='Ask') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
        if len(result1) == 0:
            return
        bitmex_ask = result1["trade"]
        bitmex_ask.rename(columns={'size': 'volume'}, inplace=True)
        bitmex_ask['Time'] = bitmex_ask.index
        bitmex_ask = bitmex_ask[['Time', 'price', 'volume']]
        bitmex_ask.reset_index(drop=True, inplace=True)
        bitmex_ask['agr_volume'] = aggregated_volume(bitmex_ask)
        bitmex_ask.rename(columns={'price': 'exit_price_swap'}, inplace=True)

        result2 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.swapMarket}' AND "symbol" = '{self.swapSymbol}' 
                        AND "side"='Bid') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
        if len(result2) == 0:
            return
        bitmex_bid = result2["trade"]
        bitmex_bid.rename(columns={'size': 'volume'}, inplace=True)
        bitmex_bid['Time'] = bitmex_bid.index
        bitmex_bid = bitmex_bid[['Time', 'price', 'volume']]
        bitmex_bid.reset_index(drop=True, inplace=True)
        bitmex_bid['agr_volume'] = aggregated_volume(bitmex_bid)
        bitmex_bid.rename(columns={'price': 'entry_price_swap'}, inplace=True)

        result3 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.spotMarket}' AND "symbol" = '{self.spotSymbol}' 
                        AND "side"='Bid') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
        if len(result3) == 0:
            return
        deribit_bid = result3["trade"]
        deribit_bid.rename(columns={'size': 'volume'}, inplace=True)
        deribit_bid['Time'] = deribit_bid.index
        deribit_bid = deribit_bid[['Time', 'price', 'volume']]
        deribit_bid.reset_index(drop=True, inplace=True)
        deribit_bid['agr_volume'] = aggregated_volume(deribit_bid)
        deribit_bid.rename(columns={'price': 'exit_price_spot'}, inplace=True)

        result4 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.spotMarket}' AND "symbol" = '{self.spotSymbol}' 
                                AND "side"='Ask') AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        if len(result4) == 0:
            return
        deribit_ask = result4["trade"]
        deribit_ask.rename(columns={'size': 'volume'}, inplace=True)
        deribit_ask['Time'] = deribit_ask.index
        deribit_ask = deribit_ask[['Time', 'price', 'volume']]
        deribit_ask.reset_index(drop=True, inplace=True)
        deribit_ask['agr_volume'] = aggregated_volume(deribit_ask)
        deribit_ask.rename(columns={'price': 'entry_price_spot'}, inplace=True)

        ################################################################################################################
        # Import maker data
        ################################################################################################################
        result5 = self.influx_connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "price" FROM "price" WHERE ("exchange" = '{self.spotMarket}' AND "symbol" = '{self.spotSymbol}') AND time >={t0}ms and time <={t1}ms GROUP BY "side"''',
            epoch='ns')
        if len(result5) == 0:
            return
        price_ask = None
        price_bid = None
        for key, value in result5.items():
            if type(key) == tuple and key[1][0][1] == "Ask":
                price_ask = value
            if type(key) == tuple and key[1][0][1] == "Bid":
                price_bid = value
        price_ask["Time"] = price_ask.index
        price_bid["Time"] = price_bid.index
        price_spot = pd.merge_ordered(price_ask, price_bid, on="Time", suffixes=["_ask", "_bid"])
        price_spot.rename(columns={"price_ask": "exit_price_spot", "price_bid": "entry_price_spot"}, inplace=True)
        price_spot.reset_index(drop=True, inplace=True)

        result6 = self.influx_connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "price" FROM "price" WHERE ("exchange" = '{self.swapMarket}' AND "symbol" = '{self.swapSymbol}') AND time >={t0}ms and time <={t1}ms GROUP BY "side"''',
            epoch='ns')
        if len(result6) == 0:
            return
        price_ask = None
        price_bid = None
        for key, value in result6.items():
            if type(key) == tuple and key[1][0][1] == "Ask":
                price_ask = value
            if type(key) == tuple and key[1][0][1] == "Bid":
                price_bid = value
        price_ask["Time"] = price_ask.index
        price_bid["Time"] = price_bid.index
        price_swap = pd.merge_ordered(price_ask, price_bid, on="Time", suffixes=["_ask", "_bid"])
        price_swap.rename(columns={"price_ask": "entry_price_swap", "price_bid": "exit_price_swap"}, inplace=True)
        price_swap.reset_index(drop=True, inplace=True)

        ################################################################################################################
        # Entry Opportunity Points
        ################################################################################################################

        ################################################################################################################
        # 1.Align dataframes
        ################################################################################################################
        entry_maker_bid = pd.merge_ordered(price_swap[['Time', 'entry_price_swap']], bitmex_bid,
                                           on=['Time', 'entry_price_swap'])
        entry_maker_bid['entry_price_swap'].ffill(inplace=True)
        # for idx in entry_maker_bid.dropna(subset=['volume']).index:
        #     if int(entry_maker_bid.loc[idx-1, 'Time'].timestamp()*1000) == int(entry_maker_bid.loc[idx, 'Time'].timestamp()*1000):
        #         temp = entry_maker_bid.loc[idx - 1, ['entry_price_spot', 'volume', 'agr_volume']]
        #         entry_maker_bid.loc[idx - 1, ['entry_price_spot', 'volume', 'agr_volume']] = entry_maker_bid.loc[idx, ['entry_price_spot', 'volume', 'agr_volume']]
        #         entry_maker_bid.loc[idx, ['entry_price_spot', 'volume', 'agr_volume']] = temp

        for ix in range(1, len(entry_maker_bid) - 1):
            if (entry_maker_bid.loc[ix - 1, 'entry_price_swap'] == entry_maker_bid.loc[ix, 'entry_price_swap']) & \
                    (math.isnan(entry_maker_bid.loc[ix, 'agr_volume'])) & \
                    (math.isnan(entry_maker_bid.loc[ix - 1, 'agr_volume']) == False):
                entry_maker_bid.loc[ix, 'agr_volume'] = entry_maker_bid.loc[ix - 1, 'agr_volume']

        entry_maker_ask = pd.merge_ordered(price_spot[['Time', 'entry_price_spot']], deribit_ask,
                                           on=['Time', 'entry_price_spot'])
        for idx in entry_maker_ask.dropna(subset=['volume']).index:
            if idx == 0:
                continue
            if int(entry_maker_ask.loc[idx - 1, 'Time'].timestamp() * 1000) == int(
                    entry_maker_ask.loc[idx, 'Time'].timestamp() * 1000):
                temp = entry_maker_ask.loc[idx - 1, ['entry_price_spot', 'volume', 'agr_volume']]
                entry_maker_ask.loc[idx - 1, ['entry_price_spot', 'volume', 'agr_volume']] = entry_maker_ask.loc[
                    idx, ['entry_price_spot', 'volume', 'agr_volume']]
                entry_maker_ask.loc[idx, ['entry_price_spot', 'volume', 'agr_volume']] = temp
        entry_maker_ask['entry_price_spot'].ffill(inplace=True)

        for ix in range(1, len(entry_maker_ask) - 1):
            if (entry_maker_ask.loc[ix - 1, 'entry_price_spot'] == entry_maker_ask.loc[ix, 'entry_price_spot']) & \
                    (math.isnan(entry_maker_ask.loc[ix, 'agr_volume'])) & \
                    (math.isnan(entry_maker_ask.loc[ix - 1, 'agr_volume']) == False):
                entry_maker_ask.loc[ix, 'agr_volume'] = entry_maker_ask.loc[ix - 1, 'agr_volume']

        ################################################################################################################
        # 2.Merge the entry dataframes
        ################################################################################################################
        entry_maker_bid['timems'] = entry_maker_bid['Time'].view(np.int64) // 10 ** 6
        shifted_price_spot = entry_maker_bid[['timems', "entry_price_swap", "volume", "agr_volume"]]
        # temp_price = shift_array_possible_extension(shifted_price_spot.values, np.zeros_like(shifted_price_spot.values),self.latency_spot)

        # shifted_price_spot["timems"] = temp_price[:, 0]
        # shifted_price_spot["entry_price_swap"] = temp_price[:, 1]
        # shifted_price_spot["volume"] = temp_price[:, 2]
        # shifted_price_spot["agr_volume"] = temp_price[:, 3]
        df = pd.merge_ordered(entry_maker_bid, entry_maker_ask, on='Time', suffixes=['_bid', '_ask'])

        # shifted_price_swap = shift_array(shifted_price_spot.values,
        #                                 np.zeros_like(self.df[["timems", "entry_price_swap", "exit_price_swap"]].values),
        #                                 self.latency_swap)
        non_nan_indices_bid = np.where(~pd.isna(df.agr_volume_bid))[0]
        non_nan_indices_ask = np.where(~pd.isna(df.agr_volume_ask))[0]
        for j in range(len(non_nan_indices_bid) - 1):
            if non_nan_indices_bid[j] + 1 != non_nan_indices_bid[j + 1]:
                if df.agr_volume_bid[non_nan_indices_bid[j]] == df.agr_volume_bid[non_nan_indices_bid[j + 1]] and \
                        df.entry_price_swap[non_nan_indices_bid[j]] == df.entry_price_swap[non_nan_indices_bid[j + 1]]:
                    df.agr_volume_bid[non_nan_indices_bid[j]: non_nan_indices_bid[j + 1]] = df.agr_volume_bid[
                        non_nan_indices_bid[j]]
        for j in range(len(non_nan_indices_ask) - 1):
            if non_nan_indices_ask[j] + 1 != non_nan_indices_ask[j + 1]:
                if df.agr_volume_ask[non_nan_indices_ask[j]] == df.agr_volume_ask[non_nan_indices_ask[j + 1]] and \
                        df.entry_price_spot[non_nan_indices_ask[j]] == df.entry_price_spot[non_nan_indices_ask[j + 1]]:
                    df.agr_volume_ask[non_nan_indices_ask[j]: non_nan_indices_ask[j + 1]] = df.agr_volume_ask[
                        non_nan_indices_ask[j]]

        df['entry_price_swap'].ffill(inplace=True)
        df['entry_price_spot'].ffill(inplace=True)
        df['timems'] = df['Time'].view(np.int64) // 10 ** 6
        df['agr_volume_ask'].fillna(0, inplace=True)
        df['agr_volume_bid'].fillna(0, inplace=True)
        df = df[(~pd.isna(df.entry_price_swap)) * (~pd.isna(df.entry_price_spot))]
        df.reset_index(inplace=True, drop=True)

        ################################################################################################################
        # 3.Compute the entry opportunity points
        ################################################################################################################
        points = []

        ix = 1
        while ix < len(df) - 2:
            if ((df.loc[ix - 1, 'entry_price_swap'] < df.loc[ix, 'entry_price_swap']) & (
                    df.loc[ix - 1, 'agr_volume_bid'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'entry_price_swap']
                start_price_spot = df.loc[ix - 1, 'entry_price_spot']

                idx = ix + 1
                while ((df.loc[idx - 1, 'entry_price_spot'] <= df.loc[idx, 'entry_price_spot']) | (
                        df.loc[idx - 1, 'agr_volume_ask'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'entry_price_spot']

                price_diff_other_market = start_price_spot - end_price
                ix += 1
                pr_diff = (1 - self.swapFee) * start_price - (1 + self.spotFee) * end_price
                pr_diff = (1 - self.swapFee) * start_price - (1 + self.spotFee) * end_price
                mm_spread = (1 - self.swapFee) * start_price - (1 + self.spotFee) * start_price_spot

                pos_dur = ed_time - st_time

                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'measurement': 'trading_opportunities_maker_maker',
                        'tags': {'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                                 'swapInstrument': 'maker_XBTUSD', 'type': f'entry_with_takers_{volume}',
                                 'starting_market': 'BitMEX_Ask'
                                 },
                        'fields': {'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                                   'price_diff_other_market': price_diff_other_market,
                                   'mm_spread': mm_spread}
                    }
                    points.append(point)

            elif ((df.loc[ix - 1, 'entry_price_spot'] > df.loc[ix, 'entry_price_spot']) & (
                    df.loc[ix - 1, 'agr_volume_ask'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'entry_price_spot']
                start_price_swap = df.loc[ix - 1, 'entry_price_swap']

                idx = ix + 1
                while ((df.loc[idx - 1, 'entry_price_swap'] >= df.loc[idx, 'entry_price_swap']) |
                       (df.loc[idx - 1, 'agr_volume_bid'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'entry_price_swap']
                ix += 1
                pr_diff = (1 - self.swapFee) * end_price - (1 + self.spotFee) * start_price
                mm_spread = (1 - self.swapFee) * start_price_swap - (1 + self.spotFee) * start_price

                price_diff_other_market = start_price_swap - end_price

                pos_dur = ed_time - st_time
                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'measurement': 'trading_opportunities_maker_maker',
                        'tags': {'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                                 'swapInstrument': 'maker_XBTUSD', 'type': f'entry_with_takers_{volume}',
                                 'starting_market': 'Deribit_Bid'
                                 },
                        'fields': {'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                                   'price_diff_other_market': price_diff_other_market,
                                   'mm_spread': mm_spread}
                    }

                    points.append(point)
            else:
                ix += 1

                # write the data to influx staging if they are more than 1000
            if len(points) > 10000:
                self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                points = []

            ################################################################################################################
            # Write the data to influx staging
            ################################################################################################################

        self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')

        ################################################################################################################
        # Exit Opportunity points
        ################################################################################################################

        ################################################################################################################
        # 1.Align exit dataframes
        ################################################################################################################

        exit_maker_ask = pd.merge_ordered(price_swap[['Time', 'exit_price_swap']], bitmex_ask,
                                          on=['Time', 'exit_price_swap'])

        for idx in exit_maker_ask.dropna(subset=['volume']).index:
            if idx == 0:
                continue
            if int(exit_maker_ask.loc[idx - 1, 'Time'].timestamp() * 1000) == int(
                    exit_maker_ask.loc[idx, 'Time'].timestamp() * 1000):
                temp = exit_maker_ask.loc[idx - 1, ['exit_price_swap', 'volume', 'agr_volume']]
                exit_maker_ask.loc[idx - 1, ['exit_price_swap', 'volume', 'agr_volume']] = exit_maker_ask.loc[
                    idx, ['exit_price_swap', 'volume', 'agr_volume']]
                exit_maker_ask.loc[idx, ['exit_price_swap', 'volume', 'agr_volume']] = temp
        exit_maker_ask['exit_price_swap'].ffill(inplace=True)
        for ix in range(1, len(exit_maker_ask) - 1):
            if (exit_maker_ask.loc[ix - 1, 'exit_price_swap'] == exit_maker_ask.loc[ix, 'exit_price_swap']) & \
                    (math.isnan(exit_maker_ask.loc[ix, 'agr_volume'])) & \
                    (math.isnan(exit_maker_ask.loc[ix - 1, 'agr_volume']) == False):
                exit_maker_ask.loc[ix, 'agr_volume'] = exit_maker_ask.loc[ix - 1, 'agr_volume']
        exit_maker_bid = pd.merge_ordered(price_spot[['Time', 'exit_price_spot']], deribit_bid,
                                          on=['Time', 'exit_price_spot'])
        for idx in exit_maker_bid.dropna(subset=['volume']).index:
            if idx == 0:
                continue
            if int(exit_maker_bid.loc[idx - 1, 'Time'].timestamp() * 1000) == int(
                    exit_maker_bid.loc[idx, 'Time'].timestamp() * 1000):
                temp = exit_maker_bid.loc[idx - 1, ['exit_price_spot', 'volume', 'agr_volume']]
                exit_maker_bid.loc[idx - 1, ['exit_price_spot', 'volume', 'agr_volume']] = exit_maker_bid.loc[
                    idx, ['exit_price_spot', 'volume', 'agr_volume']]
                exit_maker_bid.loc[idx, ['exit_price_spot', 'volume', 'agr_volume']] = temp
        exit_maker_bid['exit_price_spot'].ffill(inplace=True)

        for ix in range(1, len(exit_maker_bid) - 1):
            if (exit_maker_bid.loc[ix - 1, 'exit_price_spot'] == exit_maker_bid.loc[ix, 'exit_price_spot']) & \
                    (math.isnan(exit_maker_bid.loc[ix, 'agr_volume'])) & \
                    (math.isnan(exit_maker_bid.loc[ix - 1, 'agr_volume']) == False):
                exit_maker_bid.loc[ix, 'agr_volume'] = exit_maker_bid.loc[ix - 1, 'agr_volume']

        ################################################################################################################
        # 2.Merge the exit dataframes
        ################################################################################################################

        df = pd.merge_ordered(exit_maker_bid, exit_maker_ask, on='Time', suffixes=['_bid', '_ask'])
        df['exit_price_swap'].ffill(inplace=True)
        df['exit_price_spot'].ffill(inplace=True)
        df['timems'] = df['Time'].view(np.int64) // 10 ** 6
        df['agr_volume_ask'].fillna(0, inplace=True)
        df['agr_volume_bid'].fillna(0, inplace=True)
        df = df[(~pd.isna(df.exit_price_swap)) * (~pd.isna(df.exit_price_spot))]
        df.reset_index(inplace=True, drop=True)

        ################################################################################################################
        # 3. Compute the exit opportunity points
        ################################################################################################################

        ix = 1
        while ix < len(df) - 2:
            if ((df.loc[ix - 1, 'exit_price_swap'] > df.loc[ix, 'exit_price_swap']) & (
                    df.loc[ix - 1, 'agr_volume_ask'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'exit_price_swap']
                start_price_spot = df.loc[ix - 1, 'exit_price_spot']

                idx = ix + 1
                while ((df.loc[idx - 1, 'exit_price_spot'] >= df.loc[idx, 'exit_price_spot']) | (
                        df.loc[idx - 1, 'agr_volume_bid'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                if idx == len(df):
                    break
                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'exit_price_spot']
                ix += 1
                pr_diff = (1 - self.swapFee) * start_price - (1 + self.spotFee) * end_price
                mm_spread = (1 - self.swapFee) * start_price - (1 + self.spotFee) * start_price_spot
                price_diff_other_market = start_price_spot - end_price

                pos_dur = ed_time - st_time
                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'measurement': 'trading_opportunities_maker_maker',
                        'tags': {'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                                 'swapInstrument': 'maker_XBTUSD', 'type': f'exit_with_takers_{volume}',
                                 'starting_market': 'BitMEX_Bid'
                                 },
                        'fields': {'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                                   'price_diff_other_market': price_diff_other_market,
                                   'mm_spread': mm_spread}
                    }

                    points.append(point)

            elif ((df.loc[ix - 1, 'exit_price_spot'] < df.loc[ix, 'exit_price_spot']) & (
                    df.loc[ix - 1, 'agr_volume_bid'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'exit_price_spot']
                start_price_swap = df.loc[ix - 1, 'exit_price_swap']

                idx = ix + 1
                while ((df.loc[idx - 1, 'exit_price_swap'] <= df.loc[idx, 'exit_price_swap']) | (
                        df.loc[idx - 1, 'agr_volume_ask'] < volume)) & (idx < len(df)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'exit_price_swap']
                ix += 1
                pr_diff = (1 - self.swapFee) * end_price - (1 + self.spotFee) * start_price
                mm_spread = (1 - self.swapFee) * start_price_swap - (1 + self.spotFee) * start_price
                price_diff_other_market = start_price_swap - end_price

                pos_dur = ed_time - st_time

                if math.isnan(pr_diff) == False:
                    point = {
                        'time': int(st_time),
                        'measurement': 'trading_opportunities_maker_maker',
                        'tags': {'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                                 'swapInstrument': 'maker_XBTUSD', 'type': f'exit_with_takers_{volume}',
                                 'starting_market': 'Deribit_Ask'
                                 },
                        'fields': {'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                                   'price_diff_other_market': price_diff_other_market,
                                   'mm_spread': mm_spread}
                    }

                    points.append(point)
            else:
                ix += 1

            # write the data to influx staging if they are more than 1000
            if len(points) > 10000:
                self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')
                points = []

        ################################################################################################################
        # Write the data to influx staging
        ################################################################################################################

        self.influx_connection.staging_client_spotswap.write_points(points, time_precision='ms')

    def compute_opportunities_without_writing(self, t0, t1, volume=1000):

        ################################################################################################################
        # Import taker data
        ################################################################################################################
        result1 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.swapMarket}' AND "symbol" = '{self.swapSymbol}' 
                AND "side"='Ask') AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        if len(result1) == 0:
            return
        bitmex_ask = result1["trade"]
        bitmex_ask.rename(columns={'size': 'volume'}, inplace=True)
        bitmex_ask['Time'] = bitmex_ask.index
        bitmex_ask = bitmex_ask[['Time', 'price', 'volume']]
        bitmex_ask.reset_index(drop=True, inplace=True)
        bitmex_ask['agr_volume'] = aggregated_volume(bitmex_ask)
        bitmex_ask.rename(columns={'price': 'exit_price_swap'}, inplace=True)

        result2 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.swapMarket}' AND "symbol" = '{self.swapSymbol}' 
                        AND "side"='Bid') AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        if len(result2) == 0:
            return
        bitmex_bid = result2["trade"]
        bitmex_bid.rename(columns={'size': 'volume'}, inplace=True)
        bitmex_bid['Time'] = bitmex_bid.index
        bitmex_bid = bitmex_bid[['Time', 'price', 'volume']]
        bitmex_bid.reset_index(drop=True, inplace=True)
        bitmex_bid['agr_volume'] = aggregated_volume(bitmex_bid)
        bitmex_bid.rename(columns={'price': 'entry_price_swap'}, inplace=True)

        result3 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.spotMarket}' AND "symbol" = '{self.spotSymbol}' 
                        AND "side"='Bid') AND time >= {t0}ms and time <= {t1}ms''', epoch='ns')
        if len(result3) == 0:
            return
        deribit_bid = result3["trade"]
        deribit_bid.rename(columns={'size': 'volume'}, inplace=True)
        deribit_bid['Time'] = deribit_bid.index
        deribit_bid = deribit_bid[['Time', 'price', 'volume']]
        deribit_bid.reset_index(drop=True, inplace=True)
        deribit_bid['agr_volume'] = aggregated_volume(deribit_bid)
        deribit_bid.rename(columns={'price': 'exit_price_spot'}, inplace=True)

        result4 = self.influx_connection.archival_client_spotswap_dataframe.query(
            f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.spotMarket}' AND "symbol" = '{self.spotSymbol}' 
                                AND "side"='Ask') AND time >= {t0}ms and time <= {t1}ms ''', epoch='ns')
        if len(result4) == 0:
            return
        deribit_ask = result4["trade"]
        deribit_ask.rename(columns={'size': 'volume'}, inplace=True)
        deribit_ask['Time'] = deribit_ask.index
        deribit_ask = deribit_ask[['Time', 'price', 'volume']]
        deribit_ask.reset_index(drop=True, inplace=True)
        deribit_ask['agr_volume'] = aggregated_volume(deribit_ask)
        deribit_ask.rename(columns={'price': 'entry_price_spot'}, inplace=True)

        ################################################################################################################
        # Import maker data
        ################################################################################################################
        result5 = self.influx_connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "price" FROM "price" WHERE ("exchange" == '{self.spotMarket}' and "symbol" = '{self.spotSymbol}') AND time >={t0}ms and time <={t1}ms GROUP BY "side"''',
            epoch='ns')

        if len(result5) == 0:
            return
        price_ask = None
        price_bid = None
        for key, value in result5.items():
            if type(key) == tuple and key[1][0][1] == "Ask":
                price_ask = value
            if type(key) == tuple and key[1][0][1] == "Bid":
                price_bid = value
        price_ask["Time"] = price_ask.index
        price_bid["Time"] = price_bid.index
        price_spot = pd.merge_ordered(price_ask, price_bid, on="Time", suffixes=["_ask", "_bid"])
        price_spot.rename(columns={"price_ask": "exit_price_spot", "price_bid": "entry_price_spot"}, inplace=True)
        price_spot.reset_index(drop=True, inplace=True)

        result6 = self.influx_connection.prod_client_spotswap_dataframe.query(
            f'''SELECT "price_swap" ,type FROM "orderbook_update" 
                        WHERE ("exchangeName" = '{self.spotMarket}' AND "spotInstrument" = 'maker_{self.spotSymbol}' 
                        AND "swapInstrument" = 'maker_{self.swapSymbol}') AND time >={t0}ms and time <={t1}ms AND (type='0' Or type='1')''',
            epoch='ns')

        if len(result6) == 0:
            return
        price_swap = result6['orderbook_update']
        price_swap['Time'] = price_swap.index
        colname1 = price_swap['type'].unique().tolist()

        for names in colname1:
            if names == "0":
                price_swap['entry_price_swap'] = price_swap.loc[price_swap['type'] == '0', 'price_swap']
            elif names == "1":
                price_swap['exit_price_swap'] = price_swap.loc[price_swap['type'] == '1', 'price_swap']
        price_swap.drop(columns=['price_swap', 'type'], inplace=True)
        price_swap.reset_index(drop=True, inplace=True)

        ################################################################################################################
        # Entry Opportunity Points
        ################################################################################################################

        ################################################################################################################
        # 1.Align dataframes
        ################################################################################################################
        entry_maker_bid = pd.merge_ordered(price_swap[['Time', 'entry_price_swap']], bitmex_bid,
                                           on=['Time', 'entry_price_swap'])
        entry_maker_bid['entry_price_swap'].ffill(inplace=True)
        for ix in range(1, len(entry_maker_bid) - 1):
            if (entry_maker_bid.loc[ix - 1, 'entry_price_swap'] == entry_maker_bid.loc[ix, 'entry_price_swap']) & \
                    (math.isnan(entry_maker_bid.loc[ix, 'agr_volume'])) & \
                    (math.isnan(entry_maker_bid.loc[ix - 1, 'agr_volume']) == False):
                entry_maker_bid.loc[ix, 'agr_volume'] = entry_maker_bid.loc[ix - 1, 'agr_volume']

        entry_maker_ask = pd.merge_ordered(price_spot[['Time', 'entry_price_spot']], deribit_ask,
                                           on=['Time', 'entry_price_spot'])
        entry_maker_ask['entry_price_spot'].ffill(inplace=True)

        for ix in range(1, len(entry_maker_ask) - 1):
            if (entry_maker_ask.loc[ix - 1, 'entry_price_spot'] == entry_maker_ask.loc[ix, 'entry_price_spot']) & \
                    (math.isnan(entry_maker_ask.loc[ix, 'agr_volume'])) & \
                    (math.isnan(entry_maker_ask.loc[ix - 1, 'agr_volume']) == False):
                entry_maker_ask.loc[ix, 'agr_volume'] = entry_maker_ask.loc[ix - 1, 'agr_volume']

        ################################################################################################################
        # 2.Merge the entry dataframes
        ################################################################################################################

        df = pd.merge_ordered(entry_maker_bid, entry_maker_ask, on='Time', suffixes=['_bid', '_ask'])
        df['entry_price_swap'].ffill(inplace=True)
        df['entry_price_spot'].ffill(inplace=True)
        df['timems'] = df['Time'].view(np.int64) // 10 ** 6
        df['agr_volume_ask'].fillna(0, inplace=True)
        df['agr_volume_bid'].fillna(0, inplace=True)

        ################################################################################################################
        # 3.Compute the entry opportunity points
        ################################################################################################################
        points = []

        ix = 1
        while ix < len(df) - 2:
            if ((df.loc[ix - 1, 'entry_price_swap'] < df.loc[ix, 'entry_price_swap']) & (
                    df.loc[ix, 'agr_volume_bid'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'entry_price_swap']
                start_price_spot = df.loc[ix - 1, 'entry_price_spot']

                idx = ix + 1
                while ((df.loc[idx - 1, 'entry_price_spot'] <= df.loc[idx, 'entry_price_spot']) | (
                        df.loc[idx, 'agr_volume_ask'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'entry_price_spot']

                price_diff_other_market = start_price_spot - end_price
                ix += 1
                pr_diff = (1 - self.swapFee) * start_price - (1 + self.spotFee) * end_price

                pos_dur = ed_time - st_time

                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                        'price_diff_other_market': price_diff_other_market,
                        'side': 'entry',
                        'measurement': 'trading_opportunities_maker_maker',
                        'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                        'swapInstrument': 'maker_XBTUSD', 'type': f'entry_with_takers_{volume}',
                        'starting_market': 'BitMEX_Ask'

                    }
                    points.append(point)

            elif ((df.loc[ix - 1, 'entry_price_spot'] > df.loc[ix, 'entry_price_spot']) & (
                    df.loc[ix, 'agr_volume_ask'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'entry_price_spot']
                start_price_swap = df.loc[ix - 1, 'entry_price_swap']

                idx = ix
                while ((df.loc[idx - 1, 'entry_price_swap'] >= df.loc[idx, 'entry_price_swap']) |
                       (df.loc[idx, 'agr_volume_bid'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'entry_price_swap']
                ix += 1
                pr_diff = (1 - self.swapFee) * end_price - (1 + self.spotFee) * start_price

                price_diff_other_market = start_price_swap - end_price

                pos_dur = ed_time - st_time
                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                        'price_diff_other_market': price_diff_other_market,
                        'side': 'entry',
                        'measurement': 'trading_opportunities_maker_maker',
                        'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                        'swapInstrument': 'maker_XBTUSD', 'type': f'entry_with_takers_{volume}',
                        'starting_market': 'Deribit_Bid'

                    }

                    points.append(point)
            else:
                ix += 1
        ################################################################################################################
        # Exit Opportunity points
        ################################################################################################################

        ################################################################################################################
        # 1.Align exit dataframes
        ################################################################################################################

        exit_maker_ask = pd.merge_ordered(price_swap[['Time', 'exit_price_swap']], bitmex_ask,
                                          on=['Time', 'exit_price_swap'])
        exit_maker_ask['exit_price_swap'].ffill(inplace=True)
        for idx in exit_maker_ask.dropna(subset=['volume']).index:
            if idx == 0:
                continue
            if int(exit_maker_ask.loc[idx - 1, 'Time'].timestamp() * 1000) == int(
                    exit_maker_ask.loc[idx, 'Time'].timestamp() * 1000):
                temp = exit_maker_ask.loc[idx - 1, ['exit_price_swap', 'volume', 'agr_volume']]
                exit_maker_ask.loc[idx - 1, ['exit_price_swap', 'volume', 'agr_volume']] = exit_maker_ask.loc[
                    idx, ['exit_price_swap', 'volume', 'agr_volume']]
                exit_maker_ask.loc[idx, ['exit_price_swap', 'volume', 'agr_volume']] = temp

        for ix in range(1, len(exit_maker_ask) - 1):
            if (exit_maker_ask.loc[ix - 1, 'exit_price_swap'] == exit_maker_ask.loc[ix, 'exit_price_swap']) & \
                    (math.isnan(exit_maker_ask.loc[ix, 'agr_volume'])) & \
                    (math.isnan(exit_maker_ask.loc[ix - 1, 'agr_volume']) == False):
                exit_maker_ask.loc[ix, 'agr_volume'] = exit_maker_ask.loc[ix - 1, 'agr_volume']

        exit_maker_bid = pd.merge_ordered(price_spot[['Time', 'exit_price_spot']], deribit_bid,
                                          on=['Time', 'exit_price_spot'])
        exit_maker_bid['exit_price_spot'].ffill(inplace=True)

        for ix in range(1, len(exit_maker_bid) - 1):
            if (exit_maker_bid.loc[ix - 1, 'exit_price_spot'] == exit_maker_bid.loc[ix, 'exit_price_spot']) & \
                    (math.isnan(exit_maker_bid.loc[ix, 'agr_volume'])) & \
                    (math.isnan(exit_maker_bid.loc[ix - 1, 'agr_volume']) == False):
                exit_maker_bid.loc[ix, 'agr_volume'] = exit_maker_bid.loc[ix - 1, 'agr_volume']

        ################################################################################################################
        # 2.Merge the exit dataframes
        ################################################################################################################

        df = pd.merge_ordered(exit_maker_bid, exit_maker_ask, on='Time', suffixes=['_bid', '_ask'])
        df['exit_price_swap'].ffill(inplace=True)
        df['exit_price_spot'].ffill(inplace=True)
        df['timems'] = df['Time'].view(np.int64) // 10 ** 6
        df['agr_volume_ask'].fillna(0, inplace=True)
        df['agr_volume_bid'].fillna(0, inplace=True)

        ################################################################################################################
        # 3. Compute the exit opportunity points
        ################################################################################################################

        ix = 1
        while ix < len(df) - 2:
            if ((df.loc[ix - 1, 'exit_price_swap'] > df.loc[ix, 'exit_price_swap']) & (
                    df.loc[ix, 'agr_volume_ask'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'exit_price_swap']
                start_price_spot = df.loc[ix - 1, 'exit_price_spot']

                idx = ix + 1
                while ((df.loc[idx - 1, 'exit_price_spot'] >= df.loc[idx, 'exit_price_spot']) | (
                        df.loc[idx, 'agr_volume_bid'] < volume)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                if idx == len(df):
                    break
                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'exit_price_spot']
                ix += 1
                pr_diff = (1 - self.swapFee) * start_price - (1 + self.spotFee) * end_price

                price_diff_other_market = start_price_spot - end_price

                pos_dur = ed_time - st_time
                if not math.isnan(pr_diff):
                    point = {
                        'time': int(st_time),
                        'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                        'price_diff_other_market': price_diff_other_market,
                        'side': 'exit',
                        'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                        'swapInstrument': 'maker_XBTUSD', 'type': f'exit_with_takers_{volume}',
                        'starting_market': 'BitMEX_Bid'
                    }

                    points.append(point)

            elif ((df.loc[ix - 1, 'exit_price_spot'] < df.loc[ix, 'exit_price_spot']) & (
                    df.loc[ix, 'agr_volume_bid'] >= volume)):
                st_time = df.loc[ix - 1, 'timems']
                start_price = df.loc[ix - 1, 'exit_price_spot']
                start_price_swap = df.loc[ix - 1, 'exit_price_swap']

                idx = ix + 1
                while ((df.loc[idx - 1, 'exit_price_swap'] <= df.loc[idx, 'exit_price_swap']) | (
                        df.loc[idx, 'agr_volume_ask'] < volume)) & (idx < len(df)):
                    if idx < len(df) - 1:
                        idx += 1
                    else:
                        break

                ed_time = df.loc[idx - 1, 'timems']
                end_price = df.loc[idx - 1, 'exit_price_swap']
                ix += 1
                pr_diff = (1 - self.swapFee) * end_price - (1 + self.spotFee) * start_price
                price_diff_other_market = start_price_swap - end_price

                pos_dur = ed_time - st_time

                if math.isnan(pr_diff) == False:
                    point = {
                        'time': int(st_time),
                        'opportunity': pr_diff, 'pos_duration': int(pos_dur),
                        'price_diff_other_market': price_diff_other_market,
                        'side': 'exit',
                        'measurement': 'trading_opportunities_maker_maker',
                        'exchangeName': 'Deribit', 'spotInstrument': 'maker_BTC-PERPETUAL',
                        'swapInstrument': 'maker_XBTUSD', 'type': f'exit_with_takers_{volume}',
                        'starting_market': 'Deribit_Ask'

                    }

                    points.append(point)
            else:
                ix += 1
        return points


def start_backfill(metric):
    # TODO: add backfilling of opportunities
    metrics = {
        "DiffOppPoints": DiffOfOpportunityPoints,
        "OppPoints": BackfillOpportunityPoints,
        "OppPiontsBps": BackfillOpportunityPointsBps,
        "OppPiontsBpsFunding": BackfillOpportunityPointsBpsFunding,
        "MakerMakerOppPoints": BackfillMakerMakerOpportunityPoints,
    }
    # Continue like backfill_funding_rates.py

    backfill = metrics[metric]()
    backfill.backfill()
