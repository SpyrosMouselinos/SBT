import csv
import gzip
import zipfile
from time import sleep
import numpy as np
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.queries.queries import Prices
from dateutil.parser import parse
import time
import requests
import pandas as pd
import json
import os
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv, find_dotenv
from influxdb import InfluxDBClient
import urllib3
import urllib.request

load_dotenv(find_dotenv())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
pd.set_option("display.precision", 6)


class BackfillFunding:
    """A class for backfilling funding data from various exchanges into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFunding class.

        Sets up the InfluxDB client for connection to the database.

        Attributes:
        - exchange (str): The name of the exchange.
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        self.exchange = None
        self.client = InfluxDBClient(host='influxdb',
                                     port=8086,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=5)

    @staticmethod
    def daterange(start_date, end_date):
        """Generate a range of dates from start_date to end_date.

        @param start_date The start date of the range.
        @param end_date The end date of the range.

        @return An iterator over the range of dates.

        Example:
        @code{.py}
        for single_date in BackfillFunding.daterange(datetime(2023, 1, 1), datetime(2023, 1, 10)):
            print(single_date.strftime("%Y-%m-%d"))
        @endcode
        """
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def points_to_json(self, time, symbol, funding):
        """Convert funding data to a JSON point for InfluxDB.

        @param time The timestamp of the funding data.
        @param symbol The trading symbol for the data.
        @param funding The funding rate value.

        @return A dictionary representing the JSON point for InfluxDB.

        Example:
        @code{.py}
        point = backfill_funding.points_to_json(1625241600000, 'BTC-USD', 0.0001)
        print(point)
        @endcode
        """
        return {
            "measurement": "funding",
            "tags": {
                "exchange": self.exchange,
                "symbol": symbol,
            },
            "time": int(time),
            "fields": {
                "funding": funding,
            }
        }

    @staticmethod
    def milliseconds_to_nanoseconds(timestamp_in_milliseconds):
        """Convert a timestamp from milliseconds to nanoseconds.

        @param timestamp_in_milliseconds The timestamp in milliseconds.

        @return The timestamp in nanoseconds.

        Example:
        @code{.py}
        nanoseconds = BackfillFunding.milliseconds_to_nanoseconds(1625241600000)
        print(nanoseconds)
        @endcode
        """
        return timestamp_in_milliseconds * 1000000

    @staticmethod
    def check_for_identical_timestamps(timestamps):
        """Adjust identical timestamps to ensure uniqueness.

        This method increments the timestamps to ensure each is unique,
        which is important for database operations that require unique timestamps.

        @param timestamps A list of timestamps to check.

        @return A list of unique timestamps.

        Example:
        @code{.py}
        unique_timestamps = BackfillFunding.check_for_identical_timestamps([1625241600000, 1625241600000, 1625241600001])
        print(unique_timestamps)
        @endcode
        """
        for index in range(len(timestamps) - 2):
            to_add = 1
            next_element = index + 1
            while timestamps[index] == timestamps[next_element]:
                timestamps[next_element] += to_add
                if next_element == len(timestamps) - 1:
                    break
                next_element += 1
                to_add += 1

        return timestamps

    @staticmethod
    def milliseconds_to_microseconds(timestamp_in_milliseconds):
        """Convert a timestamp from milliseconds to microseconds.

        @param timestamp_in_milliseconds The timestamp in milliseconds.

        @return The timestamp in microseconds.

        Example:
        @code{.py}
        microseconds = BackfillFunding.milliseconds_to_microseconds(1625241600000)
        print(microseconds)
        @endcode
        """
        return timestamp_in_milliseconds * 1000

    def write_trades(self, symbol, start_date, end_date):
        """Write trades data to the database. This is a placeholder method intended to be overridden by subclasses.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.
        """
        return


class BackfillEstimatedFundingRateOkx(BackfillFunding):
    """A class for backfilling estimated funding rates from OKX into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillEstimatedFundingRateOkx class.

        Sets up the InfluxDB client and additional attributes specific to OKX.

        Attributes:
        - exchange (str): The name of the exchange ("Okex").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        - prices (Prices): An instance of the Prices class for querying price data.
        """
        BackfillFunding.__init__(self)
        self.exchange = "Okex"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        self.prices = None

    def points_to_json(self, time, symbol, funding):
        """Convert estimated funding data to a JSON point for InfluxDB.

        @param time The timestamp of the funding data.
        @param symbol The trading symbol for the data.
        @param funding The estimated funding rate value.

        @return A dictionary representing the JSON point for InfluxDB.

        Example:
        @code{.py}
        point = backfill_funding_okx.points_to_json(1625241600000, 'BTC-USD', 0.0001)
        print(point)
        @endcode
        """
        return {
            "measurement": "estimated_next_funding",
            "tags": {
                "exchange": self.exchange,
                "symbol": symbol,
            },
            "time": int(time),
            "fields": {
                "funding": funding,
            }
        }

    def get_avg_price(self, timestamp):
        """Calculate the average price from bid and ask prices.

        @param timestamp The timestamp for which to calculate the average price.

        @return The average price or None if no data is available.

        Example:
        @code{.py}
        avg_price = backfill_funding_okx.get_avg_price(1625241600000)
        print(avg_price)
        @endcode
        """
        price = self.prices.get_data(timestamp - 1000 * 60, timestamp)
        if price is None or len(price[price['side'] == 'Bid']) == 0 or len(price[price['side'] == 'Ask']) == 0:
            return None
        last_bid = price[price['side'] == 'Bid'].iloc[-1]['price']
        last_ask = price[price['side'] == 'Ask'].iloc[-1]['price']
        return (last_bid + last_ask) / 2

    def write_trades(self, symbol, start_date, end_date, market=None):
        """Write estimated funding trades data to the database.

        This method retrieves and writes estimated funding rate data from OKX for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.
        @param market The market type (optional).

        Example:
        @code{.py}
        backfill_funding_okx.write_trades('BTC-USD-SWAP', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        current_date = start_date
        start = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)

        prices_instance = Prices(InfluxConnection().staging_client_spotswap_dataframe, self.exchange,
                                 f"okex_{symbol.lower()}")
        self.prices = prices_instance.query_data(start, end)
        points_to_write = []
        point_symbol = ""
        current_funding_start = start
        current_funding_end = start + 1000 * 60 * 60 * 8
        current_funding_items = []
        current_price_items = []
        start = current_funding_end
        while current_funding_start < end:
            print(f"Processing funding from {start}")
            loop = True
            while loop:
                try:
                    response = requests.get(
                        f"https://www.okx.com/api/v5/market/history-index-candles?instId={symbol.split('-SWAP')[0]}&after={start}",
                        stream=True)
                    result = response.json()['data']

                    response_swap = requests.get(
                        f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&after={start}",
                        stream=True
                    )
                    result_swap = response_swap.json()['data']
                except Exception as e:
                    print(e)
                    continue
                for item in result_swap:
                    if current_funding_start < int(item[0]) < current_funding_end:
                        current_price_items.append(item)
                for item in result:
                    if current_funding_start < int(item[0]) < current_funding_end:
                        current_funding_items.append(item)
                    elif int(item[0]) < current_funding_start:
                        loop = False
                start = int(result[-1][0])
                sleep(0.2)
            current_funding_start = current_funding_end
            current_funding_end = current_funding_start + 1000 * 60 * 60 * 8
            start = current_funding_end

            print(len(current_funding_items))
            df = pd.DataFrame(current_funding_items, columns=["timestamp", "open", "high", "low", "close", "confirm"])
            current_prices = pd.DataFrame(current_price_items,
                                          columns=["timestamp", "open", "high", "low", "close", "vol", "volccy",
                                                   "volccyquote", "confirm"])
            current_funding_items = []
            current_price_items = []
            df = pd.merge(df, current_prices, on="timestamp")

            df['timestamp'] = df['timestamp'].astype(np.int64)
            df['open_x'] = df['open_x'].astype(np.float64)
            df['open_y'] = df['open_y'].astype(np.float64)

            df = df.sort_values(by=["timestamp"], ignore_index=True)
            df['diff'] = df.apply(lambda row: (row['open_y'] - row['open_x']) / row['open_x'], axis=1)

            for ix, row in df.iterrows():
                slice = df.iloc[0:ix]
                if len(slice) == 0:
                    continue
                funding_rate = slice['diff'].mean()
                points_to_write.append(self.points_to_json(row['timestamp'], f"okex_{symbol.lower()}", funding_rate))
            if len(points_to_write) > 1000:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []
        if len(points_to_write) > 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []

        print(f"Backfill estimated funding for exchange {self.exchange} and symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        """Write estimated funding data for all symbols to the database.

        This method retrieves and writes estimated funding rate data from OKX for all symbols that meet certain conditions.

        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_okx.write_all(datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        tickers = json.loads(requests.get("https://www.okex.com/api/v5/market/tickers?instType=SWAP").text)
        print(f"Total Symbols: {len(tickers)}")
        for ticker in tickers['data']:
            if float(ticker['volCcy24h']) * float(ticker['open24h']) > 30000:
                self.write_trades(symbol=ticker['instId'], start_date=start_date, end_date=end_date)


class BackfillFundingBinance(BackfillFunding):
    """A class for backfilling funding rates from Binance into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingBinance class.

        Sets up the InfluxDB client and additional attributes specific to Binance.

        Attributes:
        - exchange (str): The name of the exchange ("Binance").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = "Binance"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def write_trades(self, symbol, start_date, end_date, market=None):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from Binance for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.
        @param market The market type (e.g., "fapi" for futures API or "dapi" for delivery API).

        Example:
        @code{.py}
        backfill_funding_binance.write_trades('BTCUSD', datetime(2023, 1, 1), datetime(2023, 1, 10), market="fapi")
        @endcode
        """
        start = int(start_date.timestamp()) * 1000  # API accepts timestamps in milliseconds
        end = int(end_date.timestamp()) * 1000
        points_to_write = []
        point_symbol = ""
        data = None
        if market == "fapi":
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000" \
                  f"&startTime={start}&endTime={end}"
            point_symbol = f"binance_futures_{symbol.lower()}"
        else:
            url = f"https://dapi.binance.com/dapi/v1/fundingRate?symbol={symbol}&limit=1000" \
                  f"&startTime={start}&endTime={end}"
            point_symbol = f"binance_swap_{symbol.lower()}"
        try:
            response = requests.get(url)
            data = json.loads(response.text)
        except Exception as e:
            print(f"Api error {e}")
        if not data:
            print("No data found")
            return
        funding_df = pd.DataFrame(data)
        funding_df["fundingRate"] = funding_df["fundingRate"].astype("float")
        for index, row in funding_df.iterrows():
            row_points = self.points_to_json(time=row["fundingTime"],
                                             symbol=point_symbol,
                                             funding=row["fundingRate"])
            points_to_write.append(row_points)
            if len(points_to_write) >= 100:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []
        if len(points_to_write) > 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []
        print(f"Backfill for exchange {self.exchange} and symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        """Write funding data for all symbols to the database.

        This method retrieves and writes funding rate data from Binance for all symbols that meet certain conditions.

        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_binance.write_all(datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        dapi_tickers = json.loads(requests.get("https://dapi.binance.com/dapi/v1/ticker/24hr").text)
        print(f"Total Symbols: {len(dapi_tickers)}")
        for ticker in dapi_tickers:
            if float(ticker['baseVolume']) * float(ticker['lastPrice']) > 50000:
                print("Backfilling ", ticker['symbol'])
                self.write_trades(symbol=ticker['symbol'], start_date=start_date, end_date=end_date, market="dapi")

        fapi_tickers = json.loads(requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr").text)
        print(f"Total Symbols: {len(fapi_tickers)}")
        for ticker in fapi_tickers:
            if float(ticker['quoteVolume']) * float(ticker['lastPrice']) > 50000:
                print("Backfilling ", ticker['symbol'])
                self.write_trades(symbol=ticker['symbol'], start_date=start_date, end_date=end_date, market="fapi")


class BackfillFundingBinanceCoin(BackfillFunding):
    """A class for backfilling funding rates from Binance Coin-M futures into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingBinanceCoin class.

        Sets up the InfluxDB client and additional attributes specific to Binance Coin-M futures.

        Attributes:
        - exchange (str): The name of the exchange ("Binance").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = "Binance"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def write_trades(self, symbol, start_date, end_date):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from Binance Coin-M futures for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_binance_coin.write_trades('BTCUSD_PERP', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        start = int(start_date.timestamp()) * 1000  # API accepts timestamps in milliseconds
        end = int(end_date.timestamp()) * 1000
        points_to_write = []

        data = None
        url = f"https://dapi.binance.com/dapi/v1/fundingRate?symbol={symbol}&limit=1000" \
              f"&startTime={start}&endTime={end}"
        try:
            response = requests.get(url)
            data = json.loads(response.text)
        except Exception as e:
            print(f"Api error {e}")
        if not data:
            print("No data found")
            return
        funding_df = pd.DataFrame(data)
        funding_df["fundingRate"] = funding_df["fundingRate"].astype("float")
        for index, row in funding_df.iterrows():
            row_points = self.points_to_json(time=row["fundingTime"],
                                             symbol=f"binance_swap_{symbol.lower()}",
                                             funding=row["fundingRate"])
            points_to_write.append(row_points)
            if len(points_to_write) >= 100:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []
        if len(points_to_write) > 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []
        print(f"Backfill for exchange {self.exchange} and symbol {symbol} successfully ended")


class BackfillFundingFTX(BackfillFunding):
    """A class for backfilling funding rates from FTX into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingFTX class.

        Sets up the InfluxDB client and additional attributes specific to FTX.

        Attributes:
        - exchange (str): The name of the exchange ("FTX").
        """
        BackfillFunding.__init__(self)
        self.exchange = "FTX"

    def write_trades(self, symbol, start_date, end_date):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from FTX for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_ftx.write_trades('BTC-PERP', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        start = int(start_date.timestamp())  # API accepts timestamps in seconds only
        end = int(end_date.timestamp())
        points_to_write = []
        data = None
        url = f"https://ftx.com/api/funding_rates?future={symbol}" \
              f"&start_time={start}&end_time={end}"
        try:
            response = requests.get(url)
            data = json.loads(response.text)["result"]
        except Exception as e:
            print(f"Api error {e}")
        if not data:
            print("No data found")
            return
        funding_df = pd.DataFrame(data)
        funding_df["time"] = pd.to_datetime(funding_df["time"], format='%Y-%m-%dT%H:%M:%S.%f').map(
            pd.Timestamp.timestamp).astype(int)
        funding_df["time"] = funding_df["time"] * 1000
        funding_df["rate"] = funding_df["rate"].apply(lambda x: '%.9f' % x)
        for index, row in funding_df.iterrows():
            row_points = self.points_to_json(time=row["time"],
                                             symbol=symbol,
                                             funding=float(row["rate"]))
            points_to_write.append(row_points)
            if len(points_to_write) >= 100:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []
        if len(points_to_write) > 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []

        earliest = int(parse(data[-1]['time']).timestamp())
        if earliest > start and parse(data[-1]['time']) != end_date:
            time.sleep(0.2)
            self.write_trades(symbol, start_date, parse(data[-1]['time']))

        print(f"Backfill for exchange {self.exchange} and symbol {symbol} successfully ended")


class BackfillFundingHuobi(BackfillFunding):
    """A class for backfilling funding rates from Huobi DM Swap into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingHuobi class.

        Sets up the InfluxDB client and additional attributes specific to Huobi DM Swap.

        Attributes:
        - exchange (str): The name of the exchange ("HuobiDMSwap").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = "HuobiDMSwap"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def write_trades(self, symbol, start_date, end_date, page_index=1):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from Huobi DM Swap for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.
        @param page_index The page index for paginated API responses, defaults to 1.

        Example:
        @code{.py}
        backfill_funding_huobi.write_trades('BTC-USD', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        start = int(start_date.timestamp()) * 1000  # API accepts timestamps in milliseconds
        end = int(end_date.timestamp()) * 1000
        points_to_write = []
        data = None
        if "USDT" in symbol:
            url = f"https://api.hbdm.com/linear-swap-api/v1/swap_historical_funding_rate?contract_code={symbol}&page_size=50&page_index={page_index}"
        else:
            url = f"https://api.hbdm.com/swap-api/v1/swap_historical_funding_rate?contract_code={symbol}&page_size=50&page_index={page_index}"

        try:
            response = requests.get(url)
            data = json.loads(response.text)["data"]
        except Exception as e:
            print(f"Api error {e}")
        if not data:
            print("No data found")
            return ()
        for row in data['data']:
            row_points = self.points_to_json(time=row["funding_time"],
                                             symbol=symbol,
                                             funding=float(row["funding_rate"]))
            points_to_write.append(row_points)
            if len(points_to_write) >= 100:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []

        if len(points_to_write) >= 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []

        if int(data['data'][-1]['funding_time']) > start and page_index < 50:
            time.sleep(0.2)
            self.write_trades(symbol, start_date, end_date, page_index + 1)

        if page_index == 1:
            print(f"Backfill for exchange {self.exchange} and symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        """Write funding data for all symbols to the database.

        This method retrieves and writes funding rate data from Huobi DM Swap for all symbols that meet certain conditions.

        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_huobi.write_all(datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        swap_tickers = json.loads(requests.get("https://api.hbdm.com/v2/swap-ex/market/detail/batch_merged").text)
        print(f"Total Symbols: {len(swap_tickers['ticks'])}")
        for ticker in swap_tickers['ticks']:
            if float(ticker['vol']) > 20000:
                print("Backfilling ", ticker['contract_code'])
                self.write_trades(symbol=ticker['contract_code'], start_date=start_date, end_date=end_date)

        usdt_tickers = json.loads(requests.get("https://api.hbdm.com/linear-swap-ex/market/detail/batch_merged").text)
        print(f"Total Symbols: {len(usdt_tickers['ticks'])}")
        for ticker in usdt_tickers['ticks']:
            if float(ticker['trade_turnover']) > 20000:
                print("Backfilling ", ticker['contract_code'])
                self.write_trades(symbol=ticker['contract_code'], start_date=start_date, end_date=end_date)


class BackfillFundingOkex(BackfillFunding):
    """A class for backfilling funding rates from Okex into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingOkex class.

        Sets up the InfluxDB client and additional attributes specific to Okex.

        Attributes:
        - exchange (str): The name of the exchange ("Okex").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = "Okex"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def write_trades(self, symbol, start_date, end_date):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from Okex for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_okex.write_trades('BTC-USD-SWAP', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        start = int(start_date.timestamp()) * 1000  # API accepts timestamps in milliseconds
        end = int(end_date.timestamp()) * 1000
        points_to_write = []
        data = None
        url = f"https://okx.com/api/v5/public/funding-rate-history?instId={symbol}&before={start}&after={end}"
        try:
            response = requests.get(url)
            data = json.loads(response.text)["data"]
        except Exception as e:
            print(f"Api error {e}")
        if not data:
            print("No data found")
            return ()
        funding_df = pd.DataFrame(data)
        funding_df["fundingRate"] = funding_df["fundingRate"].astype("float")
        for index, row in funding_df.iterrows():
            row_points = self.points_to_json(time=row["fundingTime"],
                                             symbol=f"okex_{symbol.lower()}",
                                             funding=row["fundingRate"])
            points_to_write.append(row_points)
            if len(points_to_write) >= 100:
                self.client.write_points(points_to_write, time_precision='ms')
                points_to_write = []

        if len(points_to_write) >= 0:
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []

        print(f"Backfill for exchange {self.exchange} and symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        """Write funding data for all symbols to the database.

        This method retrieves and writes funding rate data from Okex for all symbols that meet certain conditions.

        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_okex.write_all(datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        tickers = json.loads(requests.get("https://www.okex.com/api/v5/market/tickers?instType=SWAP").text)
        print(f"Total Symbols: {len(tickers['data'])}")
        for ticker in tickers['data']:
            if float(ticker['volCcy24h']) * float(ticker['open24h']) > 50000:
                print("Backfilling ", ticker['instId'])
                self.write_trades(symbol=ticker['instId'], start_date=start_date, end_date=end_date)


class BackfillPremiumIndexBinance(BackfillFunding):
    """A class for backfilling premium index funding rates from Binance into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillPremiumIndexBinance class.

        Sets up the InfluxDB client and additional attributes specific to Binance.

        Attributes:
        - exchange (str): The name of the exchange ("Binance").
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = "Binance"
        self.client = InfluxDBClient(host='influxdb.staging.equinoxai.com',
                                     port=443,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=5,
                                     timeout=1,
                                     ssl=True, verify_ssl=True, gzip=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def points_to_json(self, time, symbol, funding):
        """Convert funding data to JSON format for InfluxDB.

        This method creates a JSON representation of funding data, which can be written to an InfluxDB database.

        @param time The timestamp of the data point.
        @param symbol The trading symbol.
        @param funding The funding rate.

        @return A dictionary in JSON format representing the data point.

        Example:
        @code{.py}
        json_data = backfill_premium_index_binance.points_to_json(1622548800000, 'BTCUSD', 0.0002)
        @endcode
        """
        return {
            "measurement": "predicted_funding",
            "tags": {
                "exchange": self.exchange,
                "symbol": symbol,
            },
            "time": int(time),
            "fields": {
                f"funding": funding,
            }
        }

    def write_trades(self, symbol, start_date, end_date, market=None):
        """Write premium index data to the database.

        This method retrieves and writes premium index data from Binance for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the premium index data.
        @param end_date The end date for the premium index data.
        @param market The market type, either "fapi" or "dapi".

        Example:
        @code{.py}
        backfill_premium_index_binance.write_trades('BTCUSD_PERP', datetime(2023, 1, 1), datetime(2023, 1, 10), market="fapi")
        @endcode
        """
        current_date = start_date
        start = current_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        points_to_write = []
        point_symbol = ""
        while start < end:
            filename = f"{symbol}-1m-{start}.zip"
            print(f"{filename}")
            if market == "fapi":
                url = f"https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/{symbol}/1m/{filename}"
                point_symbol = f"binance_futures_{symbol.lower()}"
            else:
                url = f"https://data.binance.vision/data/futures/cm/daily/premiumIndexKlines/{symbol}/1m/{filename}"
                point_symbol = f"binance_swap_{symbol.lower()}"
            try:
                response = requests.get(url, stream=True)
                with open(filename, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)
            except Exception as e:
                print(e)
                continue
            if not os.path.exists(filename):
                break
            try:
                zf = zipfile.ZipFile(filename)
            except zipfile.BadZipfile:
                print(f"filename: {filename} is not uploaded yet")
                break
            zf.namelist()
            premium_indexes_df = pd.read_csv(zf.open(filename.replace('.zip', '.csv')))

            start_time = premium_indexes_df.iloc[0]['open_time']

            while premium_indexes_df['open_time'].searchsorted(start_time) < len(premium_indexes_df):
                end_time = round(start_time + 1000 * 60 * 60 * 8)
                rows = premium_indexes_df[
                    premium_indexes_df['open_time'].between(start_time, end_time, inclusive="left")]
                rows['ix'] = range(1, len(rows) + 1)
                rows['funding_rate'] = rows.apply(lambda x: x['ix'] * x['open'], axis=1).cumsum() / np.array(
                    list(range(1, len(rows) + 1))).cumsum()
                rows['funding_rate'] = rows['funding_rate'] + rows['funding_rate'].apply(lambda x: 0.0001 - x).clip(
                    lower=-0.0005, upper=0.0005)
                start_time = end_time
                points_to_write = []
                for index, row in rows.iterrows():
                    row_points = self.points_to_json(time=row["open_time"], symbol=point_symbol,
                                                     funding=row["funding_rate"])
                    points_to_write.append(row_points)
                self.client.write_points(points_to_write, time_precision='ms')

            # do stuff
            current_date = current_date + timedelta(days=1)
            start = current_date.strftime('%Y-%m-%d')
            zf.close()
            os.remove(filename)
        print(f"Backfill Premium Index for exchange {self.exchange} and symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        """Write premium index data for all symbols to the database.

        This method retrieves and writes premium index data from Binance for all symbols that meet certain conditions.

        @param start_date The start date for the premium index data.
        @param end_date The end date for the premium index data.

        Example:
        @code{.py}
        backfill_premium_index_binance.write_all(datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        fapi_tickers = json.loads(requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr").text)
        print(f"Total Symbols: {len(fapi_tickers)}")
        for ticker in fapi_tickers:
            if float(ticker['quoteVolume']) * float(ticker['lastPrice']) > 50000:
                print("Backfilling ", ticker['symbol'])
                self.write_trades(symbol=ticker['symbol'], start_date=start_date, end_date=end_date, market="fapi")

        dapi_tickers = json.loads(requests.get("https://dapi.binance.com/dapi/v1/ticker/24hr").text)
        print(f"Total Symbols: {len(dapi_tickers)}")
        for ticker in dapi_tickers:
            if float(ticker['baseVolume']) * float(ticker['lastPrice']) > 50000:
                print("Backfilling ", ticker['symbol'])
                self.write_trades(symbol=ticker['symbol'], start_date=start_date, end_date=end_date, market="dapi")


class BackfillDeribitRealTimeFunding(BackfillFunding):
    """A class for backfilling real-time funding rates from Deribit into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillDeribitRealTimeFunding class.

        Sets up the InfluxDB client and additional attributes specific to Deribit.

        Attributes:
        - exchange (str): The name of the exchange (None for Deribit).
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.exchange = None
        self.client = InfluxDBClient('influxdb.staging.equinoxai.com',
                                     443,
                                     os.getenv("DB_USERNAME"),
                                     os.getenv("DB_PASSWORD"),
                                     'spotswap', ssl=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def points_to_json(self, time, symbol, funding):
        """Convert funding data to JSON format for InfluxDB.

        This method creates a JSON representation of funding data, which can be written to an InfluxDB database.

        @param time The timestamp of the data point.
        @param symbol The trading symbol.
        @param funding The funding rate.

        @return A dictionary in JSON format representing the data point.

        Example:
        @code{.py}
        json_data = backfill_deribit_real_time_funding.points_to_json(1622548800000, 'BTCUSD', 0.0002)
        @endcode
        """
        return {
            "measurement": "real_time_funding",
            "tags": {
                "exchange": "Deribit",
                "symbol": symbol,
            },
            "time": int(time),
            "fields": {
                f"funding": funding,
                f"funding_8h": funding,

            }
        }

    def query_funding(self, time_from, time_to, symbol):
        """Query funding data from InfluxDB.

        This method retrieves funding rate data from InfluxDB for a given symbol and time range.

        @param time_from The start time for the funding data.
        @param time_to The end time for the funding data.
        @param symbol The trading symbol.

        @return A list of funding rate data points.

        Example:
        @code{.py}
        funding_data = backfill_deribit_real_time_funding.query_funding(datetime(2023, 1, 1), datetime(2023, 1, 10), 'BTCUSD')
        @endcode
        """
        query = f'''SELECT funding * 8 FROM "funding" WHERE (exchange = 'Deribit') AND symbol = '{symbol}' AND time > {int(time_from.replace(tzinfo=timezone.utc).timestamp() * 1000)}ms AND time <= {int(time_to.replace(tzinfo=timezone.utc).timestamp() * 1000)}ms '''
        funding = self.client.query(query)
        return list(funding)

    def write_trades(self, time_from, time_to, symbol):
        """Write real-time funding data to the database.

        This method retrieves and writes real-time funding rate data from Deribit for a given symbol and time range.

        @param time_from The start time for the funding data.
        @param time_to The end time for the funding data.
        @param symbol The trading symbol.

        Example:
        @code{.py}
        backfill_deribit_real_time_funding.write_trades(datetime(2023, 1, 1), datetime(2023, 1, 10), 'BTCUSD')
        @endcode
        """
        points = []
        one_hour_funding = self.query_funding(time_from, time_to, symbol)

        for every_hour in one_hour_funding[0]:
            every_second = datetime.strptime(every_hour["time"], '%Y-%m-%dT%H:%M:%SZ') - timedelta(seconds=1)
            previous_funding_time = datetime.strptime(every_hour["time"], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=1)
            while every_second > previous_funding_time:
                points.append(
                    self.points_to_json(time=int(every_second.replace(tzinfo=timezone.utc).timestamp() * 1000),
                                        symbol=symbol, funding=every_hour["funding"]))
                every_second = every_second - timedelta(seconds=1)
            self.client.write_points(points, time_precision='ms')
            print(
                f'Backfilled real_time_funding value:{every_hour["funding"]} from: {datetime.strptime(every_hour["time"], "%Y-%m-%dT%H:%M:%SZ")} to: {previous_funding_time}')
            points = []


class BackfillFundingBitMEX(BackfillFunding):
    """A class for backfilling funding rates from BitMEX into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillFundingBitMEX class.

        Sets up the InfluxDB client and additional attributes specific to BitMEX.

        Attributes:
        - influx_connection (InfluxConnection): The InfluxDB connection instance.
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        BackfillFunding.__init__(self)
        self.influx_connection = InfluxConnection()
        self.client = InfluxDBClient('influxdb.staging.equinoxai.com',
                                     443,
                                     os.getenv("DB_USERNAME"),
                                     os.getenv("DB_PASSWORD"),
                                     'spotswap', ssl=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    @staticmethod
    def daterange(start_date, end_date):
        """Generate a range of dates from start_date to end_date.

        @param start_date The start date.
        @param end_date The end date.

        @return An iterator over dates from start_date to end_date.

        Example:
        @code{.py}
        for date in BackfillFundingBitMEX.daterange(datetime(2023, 1, 1), datetime(2023, 1, 10)):
            print(date)
        @endcode
        """
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def points_to_json(self, time, symbol, funding):
        """Convert funding data to JSON format for InfluxDB.

        This method creates a JSON representation of funding data, which can be written to an InfluxDB database.

        @param time The timestamp of the data point.
        @param symbol The trading symbol.
        @param funding The funding rate.

        @return A dictionary in JSON format representing the data point.

        Example:
        @code{.py}
        json_data = backfill_funding_bitmex.points_to_json(1622548800000, 'BTCUSD', 0.0002)
        @endcode
        """
        return {
            "measurement": "funding",
            "tags": {
                "exchange": "BitMEX",
                "symbol": symbol,
            },
            "time": int(time),
            "fields": {
                f"funding": float(funding),
            }
        }

    def write_points(self, array, side, price_microsecond_shift=0):
        """Write aggregated points data to JSON format.

        This method creates a JSON representation of aggregated points data for InfluxDB.

        @param array The array containing points data.
        @param side The trading side (e.g., "Bid" or "Ask").
        @param price_microsecond_shift The microsecond shift for price.

        @return A dictionary in JSON format representing the aggregated points data.

        Example:
        @code{.py}
        json_res = backfill_funding_bitmex.write_points(array, "Bid", price_microsecond_shift=0)
        @endcode
        """
        volume = np.sum(array[:, 2])
        res = np.array([array[0, 0] + price_microsecond_shift, array[-1, 1], volume])

        json_res = self.points_to_json(res[0], side, res[1])
        return json_res

    def write_trades(self, symbol, start_date, end_date):
        """Write funding trades data to the database.

        This method retrieves and writes funding rate data from BitMEX for a given symbol and date range.

        @param symbol The trading symbol.
        @param start_date The start date for the trades data.
        @param end_date The end date for the trades data.

        Example:
        @code{.py}
        backfill_funding_bitmex.write_trades('BTCUSD', datetime(2023, 1, 1), datetime(2023, 1, 10))
        @endcode
        """
        # TODO: NEEDS RENAMING AND ADJUSTMENT
        t0 = start_date
        t1 = end_date
        start_time = datetime.fromtimestamp(time.mktime(t0)).isoformat(sep='T', timespec='milliseconds') + 'Z'
        end_time = datetime.fromtimestamp(time.mktime(t1)).isoformat(sep='T', timespec='milliseconds') + 'Z'
        start = time.time()
        print(f"Started backfilling funding for bitmex from {start_time} to {end_time}")
        result = []
        query_counter = 0
        query_count = 100
        points_to_write = []
        retries = 0
        while True:
            url = f"https://www.bitmex.com/api/v1/funding?count={query_count}&start={query_counter}&reverse=false&symbol={symbol}&startTime={start_time}&endTime={end_time}"
            current_query_result = requests.get(url).json()
            if not isinstance(current_query_result, list):
                print(f"BitMEX API call thew error, waiting 10s: {current_query_result['error']['message']}")
                sleep(10)
                continue
            result.append(current_query_result)
            if len(current_query_result) < query_count:
                if len(current_query_result) == 0:
                    print(f'Query length: {len(current_query_result)} less than {query_count}.')
                    if retries == 3:
                        print("Error, exiting at ")
                        break
                    else:
                        retries += 1
                        continue

                print(
                    f'Query length: {len(current_query_result)} less than {query_count}. From {current_query_result[0]["timestamp"]} to {current_query_result[-1]["timestamp"]}')
            if retries > 0:
                retries = 0
            if query_counter == 0:
                query_counter += 1
            query_counter += query_count
            sleep(1.5)
            # print(f"BitMEX API call nr {query_counter // query_count} succeeded.")
            if len(result) == 3:
                result = [item for sublist in result for item in sublist]
                for item in result:
                    timems = datetime.strptime(item["timestamp"], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                        tzinfo=timezone.utc).timestamp() * 1000
                    points_to_write.append(
                        self.points_to_json(time=timems, symbol=item["symbol"], funding=item["fundingRate"]))

                self.client.write_points(points_to_write, time_precision='ms')
                print(
                    f'Inserted to influx: {len(points_to_write)} points. From {datetime.fromtimestamp(points_to_write[0]["time"] / 1000)} to {datetime.fromtimestamp(points_to_write[-1]["time"] / 1000)}')
                start_time = datetime.fromtimestamp(points_to_write[-1]["time"] / 1000).isoformat(sep='T',
                                                                                                  timespec='milliseconds') + 'Z'
                query_counter = 0
                print(f"New startTime: {start_time} , New queryTime: {query_counter}")

                points_to_write = []
                result = []

                # convert to points and write
        if len(result) > 0:
            result = [item for sublist in result for item in sublist]
            for item in result:
                timems = datetime.strptime(item["timestamp"], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                    tzinfo=timezone.utc).timestamp() * 1000
                points_to_write.append(
                    self.points_to_json(time=timems, symbol=item["symbol"], funding=item["fundingRate"]))
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []
            result = []
        end = time.time()
        print(f"Total time elapsed: {(end - start) / 60} minutes. Total requests: {query_counter // query_count}")


class BackfillQuotesBitMEX:
    """A class for backfilling quotes from BitMEX into an InfluxDB database."""

    def __init__(self):
        """Initialize the BackfillQuotesBitMEX class.

        Sets up the InfluxDB client and additional attributes specific to BitMEX.

        Attributes:
        - influx_connection (InfluxConnection): The InfluxDB connection instance.
        - client (InfluxDBClient): The InfluxDB client used for database operations.
        """
        self.influx_connection = InfluxConnection()
        self.client = InfluxDBClient('influxdb.staging.equinoxai.com',
                                     443,
                                     os.getenv("DB_USERNAME"),
                                     os.getenv("DB_PASSWORD"),
                                     'spotswap', ssl=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    @staticmethod
    def daterange(start_date, end_date):
        """Generate a range of dates from start_date to end_date.

        @param start_date The start date.
        @param end_date The end date.

        @return An iterator over dates from start_date to end_date.

        Example:
        @code{.py}
        for date in BackfillQuotesBitMEX.daterange(datetime(2023, 1, 1), datetime(2023, 1, 10)):
            print(date)
        @endcode
        """
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def points_to_json(self, time, symbol, side, price):
        """Convert quotes data to JSON format for InfluxDB.

        This method creates a JSON representation of quotes data, which can be written to an InfluxDB database.

        @param time The timestamp of the data point.
        @param symbol The trading symbol.
        @param side The trading side (e.g., "Bid" or "Ask").
        @param price The price value.

        @return A dictionary in JSON format representing the data point.

        Example:
        @code{.py}
        json_data = backfill_quotes_bitmex.points_to_json(1622548800000, 'BTCUSD', 'Bid', 30000.0)
        @endcode
        """
        return {
            "measurement": "price",
            "tags": {
                "exchange": "BitMEX",
                "symbol": symbol,
                "side": side,
            },
            "time": int(time),
            "fields": {
                f"price": price,
            }
        }

    def write_points(self, array, side, price_microsecond_shift=0):
        """Write aggregated points data to JSON format.

        This method creates a JSON representation of aggregated points data for InfluxDB.

        @param array The array containing points data.
        @param side The trading side (e.g., "Bid" or "Ask").
        @param price_microsecond_shift The microsecond shift for price.

        @return A dictionary in JSON format representing the aggregated points data.

        Example:
        @code{.py}
        json_res = backfill_quotes_bitmex.write_points(array, "Bid", price_microsecond_shift=0)
        @endcode
        """
        volume = np.sum(array[:, 2])
        res = np.array([array[0, 0] + price_microsecond_shift, array[-1, 1], volume])

        json_res = self.points_to_json(res[0], side, res[1], res[2])
        return json_res

    def write_prices(self, t0, t1, symbol):
        """Write quotes data to the database.

        This method retrieves and writes quotes data from BitMEX for a given symbol and date range.

        @param t0 The start date for the quotes data.
        @param t1 The end date for the quotes data.
        @param symbol The trading symbol.

        Example:
        @code{.py}
        backfill_quotes_bitmex.write_prices(datetime(2023, 1, 1), datetime(2023, 1, 10), 'BTCUSD')
        @endcode
        """
        start_time = datetime.fromtimestamp(time.mktime(t0)).isoformat(sep='T', timespec='milliseconds') + 'Z'
        end_time = datetime.fromtimestamp(time.mktime(t1)).isoformat(sep='T', timespec='milliseconds') + 'Z'
        start = time.time()
        print(f"Started backfilling prices for bitmex from {start_time} to {end_time}")
        result = []
        query_counter = 0
        query_count = 1000
        points_to_write = []
        retries = 0
        while True:
            url = f"https://www.bitmex.com/api/v1/quote?count={query_count}&start={query_counter}&reverse=false&symbol={symbol}&startTime={start_time}&endTime={end_time}"
            current_query_result = requests.get(url).json()
            if not isinstance(current_query_result, list):
                print(f"BitMEX API call thew error, waiting 10s: {current_query_result['error']['message']}")
                sleep(10)
                continue
            result.append(current_query_result)
            if len(current_query_result) < query_count:
                if len(current_query_result) == 0:
                    print(f'Query length: {len(current_query_result)} less than {query_count}.')
                    if retries == 3:
                        print("Error, exiting at ")
                        break
                    else:
                        retries += 1
                        continue

                print(
                    f'Query length: {len(current_query_result)} less than {query_count}. From {current_query_result[0]["timestamp"]} to {current_query_result[-1]["timestamp"]}')
            if retries > 0:
                retries = 0
            if query_counter == 0:
                query_counter += 1
            query_counter += query_count
            sleep(1.5)
            # print(f"BitMEX API call nr {query_counter // query_count} succeeded.")
            if len(result) == 10:
                result = [item for sublist in result for item in sublist]
                previous_ask_price = 0.0
                previous_bid_price = 0.0
                for item in result:
                    timems = datetime.strptime(item["timestamp"], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                        tzinfo=timezone.utc).timestamp() * 1000

                    if item["askPrice"] != previous_ask_price or item["bidPrice"] != previous_bid_price:
                        points_to_write.append(self.points_to_json(time=timems, symbol=item["symbol"], side="Ask",
                                                                   price=float(item["askPrice"])))
                        previous_ask_price = float(item["askPrice"])
                        points_to_write.append(self.points_to_json(time=timems, symbol=item["symbol"], side="Bid",
                                                                   price=float(item["bidPrice"])))
                        previous_bid_price = float(item["bidPrice"])
                self.client.write_points(points_to_write, time_precision='ms')
                print(
                    f'Inserted to influx: {len(points_to_write)} points. From {datetime.fromtimestamp(points_to_write[0]["time"] / 1000)} to {datetime.fromtimestamp(points_to_write[-1]["time"] / 1000)}')
                if datetime.fromtimestamp(points_to_write[-1]["time"] / 1000) > datetime.strptime(start_time,
                                                                                                  '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(
                    days=1, hours=3):
                    print(datetime.fromtimestamp(points_to_write[-1]["time"] / 1000),
                          datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(days=1, hours=3))
                    new_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(days=1)
                    start_time = new_time.isoformat(sep='T', timespec='milliseconds') + 'Z'
                    query_counter = 0
                    print(f"New startTime: {start_time} , New queryTime: {query_counter}")

                points_to_write = []
                result = []

                # convert to points and write
        if len(result) > 0:
            result = [item for sublist in result for item in sublist]
            previous_ask_price = 0.0
            previous_bid_price = 0.0
            for item in result:
                timems = datetime.strptime(item["timestamp"], '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                    tzinfo=timezone.utc).timestamp() * 1000
                if item["askPrice"] != previous_ask_price or item["bidPrice"] != previous_bid_price:
                    points_to_write.append(
                        self.points_to_json(time=timems, symbol=item["symbol"], side="Ask",
                                            price=float(item["askPrice"])))
                    previous_ask_price = float(item["askPrice"])
                    points_to_write.append(
                        self.points_to_json(time=timems, symbol=item["symbol"], side="Bid",
                                            price=float(item["bidPrice"])))
                    previous_bid_price = float(item["bidPrice"])
            self.client.write_points(points_to_write, time_precision='ms')
            points_to_write = []
            result = []
        end = time.time()
        print(f"Total time elapsed: {(end - start) / 60} minutes. Total requests: {query_counter // query_count}")

    def write_trades(self, start_date, end_date, symbol):
        """Write trade data to the database from downloaded files.

        This method retrieves and writes trade data from BitMEX for a given symbol and date range by downloading and processing data files.

        @param start_date The start date for the trade data.
        @param end_date The end date for the trade data.
        @param symbol The trading symbol.

        Example:
        @code{.py}
        backfill_quotes_bitmex.write_trades(datetime(2023, 1, 1), datetime(2023, 1, 10), 'BTCUSD')
        @endcode
        """
        for single_date in self.daterange(start_date, end_date):
            date = single_date.strftime("%Y%m%d")
            urllib.request.urlretrieve(f"https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/{date}.csv.gz",
                                       f"quotes_download/{date}.csv.gz")
            df = pd.read_csv(f"quotes_download/{date}.csv.gz", compression='gzip')
            df = df[df["symbol"] == symbol]
            last_price_ask = None
            last_price_bid = None
            df['diff_ask'] = np.abs(df['askPrice'].diff())
            df['diff_ask'].iloc[0] = 1
            df['diff_bid'] = np.abs(df['bidPrice'].diff())
            df['diff_bid'].iloc[0] = 1
            df['diff'] = df['diff_ask'] + df['diff_bid']
            df = df[df['diff'] != 0]
            points_to_write = []
            for _, line in df.iterrows():
                dt_obj = datetime.strptime(line['timestamp'][:-3], '%Y-%m-%dD%H:%M:%S.%f')
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                microsecond = int(dt_obj.timestamp() * 1000000)
                if last_price_ask is None and last_price_bid is None:
                    last_price_ask = line['askPrice']
                    points_to_write.append(self.points_to_json(microsecond, symbol, 'Ask', last_price_ask))
                    last_price_bid = line['bidPrice']
                    points_to_write.append(self.points_to_json(microsecond, symbol, 'Bid', last_price_bid))
                    continue

                if line['diff_ask'] != 0:
                    last_price_ask = line['askPrice']
                    points_to_write.append(self.points_to_json(microsecond, symbol, 'Ask', last_price_ask))
                if line['diff_bid'] != 0:
                    last_price_bid = line['bidPrice']
                    points_to_write.append(self.points_to_json(microsecond, symbol, 'Bid', last_price_bid))

                if len(points_to_write) > 10000:
                    self.influx_connection.staging_client_spotswap.write_points(points_to_write, time_precision='u')
                    points_to_write = []

            if len(points_to_write) > 0:
                self.influx_connection.staging_client_spotswap.write_points(points_to_write, time_precision='u')
                points_to_write = []

            os.remove(f"quotes_download/{date}.csv.gz")


class BackfillTradesBitMEX:
    """
    A class to backfill trade data from BitMEX into an InfluxDB database.

    This class fetches trade data from BitMEX for a given date range and trading symbol,
    processes the data, and writes it to an InfluxDB database.

    Attributes:
    ----------
    client : InfluxDBClient
        An instance of the InfluxDBClient used for database operations.

    Methods:
    -------
    daterange(start_date, end_date):
        Generates a range of dates from start_date to end_date.

    points_to_json(time, side, price, size, symbol):
        Converts trade data to JSON format for InfluxDB.

    write_points(array, side, symbol, price_microsecond_shift=0):
        Writes aggregated trade data points in JSON format.

    write_trades(start_date, end_date, symbol):
        Fetches and writes trade data to the database for a given date range and symbol.
    """

    def __init__(self):
        """
        Initializes the BackfillTradesBitMEX class.

        Sets up the InfluxDB client with necessary credentials and settings.

        Attributes:
        ----------
        client : InfluxDBClient
            The InfluxDB client for database operations.
        """
        self.client = InfluxDBClient(
            'simulations-influxdb.staging.equinoxai.com',
            443,
            os.getenv("DB_USERNAME"),
            os.getenv("DB_PASSWORD"),
            'spotswap', ssl=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    @staticmethod
    def daterange(start_date, end_date):
        """
        Generates a range of dates from start_date to end_date.

        Parameters:
        ----------
        start_date : datetime
            The start date of the range.
        end_date : datetime
            The end date of the range.

        Yields:
        ------
        datetime
            Each date in the range from start_date to end_date.

        Example:
        -------
        >>> for date in BackfillTradesBitMEX.daterange(datetime(2023, 1, 1), datetime(2023, 1, 5)):
        >>>     print(date)
        datetime(2023, 1, 1)
        datetime(2023, 1, 2)
        datetime(2023, 1, 3)
        datetime(2023, 1, 4)
        datetime(2023, 1, 5)
        """
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    @staticmethod
    def points_to_json(time, side, price, size, symbol):
        """
        Converts trade data to JSON format for InfluxDB.

        Parameters:
        ----------
        time : int
            The timestamp of the trade data.
        side : str
            The side of the trade, either 'Buy' or 'Sell'.
        price : float
            The price at which the trade occurred.
        size : int
            The size of the trade.
        symbol : str
            The trading symbol.

        Returns:
        -------
        dict
            A dictionary in JSON format representing the trade data.

        Example:
        -------
        >>> trade_data = BackfillTradesBitMEX.points_to_json(1622548800000, 'Buy', 30000.0, 10, 'BTCUSD')
        """
        return {
            "measurement": "trade",
            "tags": {
                "exchange": "BitMEX",
                "symbol": symbol,
                "side": side,
            },
            "time": int(time),
            "fields": {
                f"price": price,
                "size": size
            }
        }

    def write_points(self, array, side, symbol, price_microsecond_shift=0):
        """
        Writes aggregated trade data points in JSON format.

        Parameters:
        ----------
        array : np.ndarray
            The array containing trade data points.
        side : str
            The side of the trade, either 'Bid' or 'Ask'.
        symbol : str
            The trading symbol.
        price_microsecond_shift : int, optional
            Microsecond shift applied to the price for adjustment (default is 0).

        Returns:
        -------
        dict
            A dictionary in JSON format representing the aggregated trade data points.

        Example:
        -------
        >>> points = np.array([[1622548800000, 30000.0, 10], [1622548800001, 30001.0, 5]])
        >>> json_res = backfill.write_points(points, 'Bid', 'BTCUSD')
        """
        volume = np.sum(array[:, 2])
        res = np.array([array[0, 0] + price_microsecond_shift, array[-1, 1], volume])

        json_res = self.points_to_json(res[0], side, res[1], res[2], symbol=symbol)
        return json_res

    def write_trades(self, start_date, end_date, symbol):
        """
        Fetches and writes trade data to the database for a given date range and symbol.

        This method retrieves trade data from BitMEX, processes it, and writes it to the InfluxDB database.

        Parameters:
        ----------
        start_date : datetime
            The start date of the data retrieval.
        end_date : datetime
            The end date of the data retrieval.
        symbol : str
            The trading symbol.

        Example:
        -------
        >>> backfill.write_trades(datetime(2023, 1, 1), datetime(2023, 1, 10), 'BTCUSD')
        """
        for single_date in self.daterange(start_date, end_date):
            print(single_date, symbol)
            date = single_date.strftime("%Y%m%d")
            urllib.request.urlretrieve(f"https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/{date}.csv.gz",
                                       f"{date}.csv.gz")
            f = gzip.open(f"{date}.csv.gz", mode="rt")
            csvobj = csv.reader(f, delimiter=',', quotechar="'")
            sell_trades = []
            buy_trades = []
            points_to_write = []
            price_microsecond_shift_bid = 0
            price_microsecond_shift_ask = 0
            for line in csvobj:
                if symbol in line[1]:
                    dt_obj = datetime.strptime(line[0][:-6],
                                               '%Y-%m-%dD%H:%M:%S.%f')
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    millisec = int(dt_obj.timestamp() * 1000000)
                    processed_line = [int(millisec), float(line[4]), int(line[3])]
                    price = processed_line[1]

                    if "Sell" in line[2]:
                        if len(sell_trades) == 0:
                            sell_trades.append(processed_line)
                        elif millisec != sell_trades[0][0]:
                            points = self.write_points(np.array(sell_trades), "Ask", symbol=symbol,
                                                       price_microsecond_shift=price_microsecond_shift_bid)  # TODO I am aware of the fact that I'm writing 'bid' points as 'Ask' - it's working as intended
                            points_to_write.append(points)
                            sell_trades = []
                            sell_trades.append(processed_line)
                            price_microsecond_shift_bid = 0
                        elif millisec == sell_trades[0][0] and price != sell_trades[0][1]:
                            points = self.write_points(np.array(sell_trades), "Ask", symbol=symbol,
                                                       price_microsecond_shift=price_microsecond_shift_bid)
                            points_to_write.append(points)
                            sell_trades = []
                            sell_trades.append(processed_line)
                            price_microsecond_shift_bid += 1
                        elif millisec == sell_trades[0][0] and price == sell_trades[0][1]:
                            sell_trades.append(processed_line)
                        else:
                            print("shouldn't run")

                    if "Buy" in line[2]:
                        if len(buy_trades) == 0:
                            buy_trades.append(processed_line)
                        elif millisec != buy_trades[0][0]:
                            points = self.write_points(np.array(buy_trades), "Bid", symbol=symbol,
                                                       price_microsecond_shift=price_microsecond_shift_ask)
                            points_to_write.append(points)
                            buy_trades = []
                            buy_trades.append(processed_line)
                            price_microsecond_shift_ask = 0
                        elif millisec == buy_trades[0][0] and price != buy_trades[0][1]:
                            points = self.write_points(np.array(buy_trades), "Bid", symbol=symbol,
                                                       price_microsecond_shift=price_microsecond_shift_ask)
                            points_to_write.append(points)
                            buy_trades = []
                            buy_trades.append(processed_line)
                            price_microsecond_shift_ask += 1
                        elif millisec == buy_trades[0][0] and price == buy_trades[0][1]:
                            buy_trades.append(processed_line)
                        else:
                            print("shouldn't run")

                    if len(points_to_write) >= 10000:
                        self.client.write_points(points_to_write, time_precision='u')
                        points_to_write = []

            points = self.write_points(np.array(sell_trades), "Ask", symbol=symbol)
            points_to_write.append(points)

            points = self.write_points(np.array(buy_trades), "Bid", symbol=symbol)
            points_to_write.append(points)

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='u')
                points_to_write = []

            f.close()
            os.remove(f"{date}.csv.gz")


def start_backfill(exchanges):
    """
    Starts the backfilling process for specified exchanges.

    This function iterates over the provided exchanges and performs backfilling
    of trade data using the appropriate backfill class.

    Parameters:
    ----------
    exchanges : dict
        A dictionary containing exchange information and symbols to be backfilled.

    Example:
    -------
    >>> exchanges = {
    >>>     "Exchanges": [
    >>>         {
    >>>             "exchangeName": "TradesBitMEX",
    >>>             "symbols": [
    >>>                 {
    >>>                     "symbolName": "BTCUSD",
    >>>                     "from": "2023-01-01 00:00:00",
    >>>                     "to": "2023-01-10 00:00:00"
    >>>                 }
    >>>             ]
    >>>         }
    >>>     ]
    >>> }
    >>> start_backfill(exchanges)
    """
    backfill_exchanges = {
        "Okex": BackfillFundingOkex,
        "OkexAll": BackfillFundingOkex,
        "HuobiAll": BackfillFundingHuobi,
        "BinanceAll": BackfillFundingBinance,
        "FTX": BackfillFundingFTX,
        "Binance": BackfillFundingBinance,
        "DeribitRealTime": BackfillDeribitRealTimeFunding,
        "BitMex": BackfillDeribitRealTimeFunding,
        "QuotesBitMEX": BackfillQuotesBitMEX,
        "TradesBitMEX": BackfillTradesBitMEX,
    }
    for exchange in exchanges["Exchanges"]:
        print(exchange["exchangeName"])
        backfiller = backfill_exchanges[exchange["exchangeName"]]()
        if "All" in exchange["exchangeName"]:
            backfiller.write_all(start_date=datetime.strptime(exchange["from"], "%Y-%m-%d %H:%M:%S"),
                                 end_date=datetime.strptime(exchange["to"], "%Y-%m-%d %H:%M:%S"))
        else:
            for symbol in exchange["symbols"]:
                print(symbol["symbolName"])
                if datetime.strptime(symbol["from"], "%Y-%m-%d %H:%M:%S") > datetime.now():
                    print(f'Error: invalid start_time={symbol["from"]}')
                    continue
                else:
                    backfiller.write_trades(symbol=symbol["symbolName"],
                                            start_date=datetime.strptime(symbol["from"], "%Y-%m-%d %H:%M:%S"),
                                            end_date=datetime.strptime(symbol["to"], "%Y-%m-%d %H:%M:%S"))
