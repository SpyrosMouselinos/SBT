import json
import argparse
import os.path
import zipfile
from time import sleep

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import requests
from influxdb import InfluxDBClient
from src.common.connections.DatabaseConnections import InfluxConnection


class BackfillTrades:
    def __init__(self):
        self.exchange = None
        self.client = InfluxDBClient(host='simulations-influxdb',
                                     port=8086,
                                     username=os.getenv("DB_USERNAME"),
                                     password=os.getenv("DB_PASSWORD"),
                                     database='spotswap',
                                     retries=10,
                                     timeout=5)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    @staticmethod
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def points_to_json(self, time, side, price, size, symbol):
        return {
            "measurement": "trade",
            "tags": {
                "exchange": self.exchange,
                "symbol": symbol,
                "side": side,
            },
            "time": int(time),
            "fields": {
                f"price": price,
                "size": size
            }
        }

    @staticmethod
    def milliseconds_to_nanoseconds(timestamp_in_milliseconds):
        return timestamp_in_milliseconds * 1000000

    @staticmethod
    def check_for_identical_timestamps(timestamps):
        for index in range(len(timestamps) - 2):
            to_add = 1
            next_element = index + 1
            while timestamps[index] == timestamps[next_element]:
                timestamps[next_element] += to_add
                if next_element == len(
                        timestamps) - 1:  # in case the timestamp mills remains same until the last element of dataframe, we break so we wont get out of index
                    break
                next_element += 1
                to_add += 1

        return timestamps

    @staticmethod
    def milliseconds_to_microseconds(timestamp_in_milliseconds):
        return timestamp_in_milliseconds * 1000

    def write_trades(self, symbol, start_date, end_date):
        return


class BackfillPricesLMAX(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "LMAX"
        self.client = InfluxDBClient('influxdb.staging.equinoxai.com',
                                     443,
                                     os.getenv("DB_USERNAME"),
                                     os.getenv("DB_PASSWORD"),
                                     'spotswap', ssl=True)
        self.client._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")

    def write_trades(self, symbol, start_date=None, end_date=None):

        path = "/home/kt/Downloads/BTCUSD & ETHUSD"
        for path, subdirs, files in os.walk(path):
            for name in files:
                filename = os.path.join(path, name)
                symbol = filename.split('/')[5]
                if symbol == 'BTCUSD':
                    df = pd.read_csv(filename)
                    df = df.dropna()
                    df = df.drop_duplicates(subset=['TIMESTAMP'], keep='last').reset_index(drop=True)

                    bids_df = df[["TIMESTAMP", "BID_PRICE_1", "BID_QTY_1"]]
                    asks_df = df[["TIMESTAMP", "ASK_PRICE_1", "ASK_QTY_1"]]

                    bids_df = bids_df.groupby(
                        (bids_df["BID_PRICE_1"] != bids_df["BID_PRICE_1"].shift()).cumsum().values).first()
                    asks_df = asks_df.groupby(
                        (asks_df["ASK_PRICE_1"] != asks_df["ASK_PRICE_1"].shift()).cumsum().values).first()
                    print(filename)
                    points_to_write = []
                    for index, row in bids_df.iterrows():
                        row_points = self.price_points_to_json(time=row["TIMESTAMP"],
                                                               side="Bid",
                                                               price=row["BID_PRICE_1"],
                                                               size=row["BID_QTY_1"],
                                                               symbol=symbol)
                        points_to_write.append(row_points)
                        if len(points_to_write) >= 10000:
                            self.client.write_points(points_to_write, time_precision='ms')
                            points_to_write = []

                    if len(points_to_write) > 0:
                        self.client.write_points(points_to_write, time_precision='ms')

                    points_to_write = []
                    for index, row in asks_df.iterrows():
                        row_points = self.price_points_to_json(time=row["TIMESTAMP"],
                                                               side="Ask",
                                                               price=row["ASK_PRICE_1"],
                                                               size=row["ASK_QTY_1"],
                                                               symbol=symbol)
                        points_to_write.append(row_points)
                        if len(points_to_write) >= 10000:
                            self.client.write_points(points_to_write, time_precision='ms')
                            points_to_write = []

                    if len(points_to_write) > 0:
                        self.client.write_points(points_to_write, time_precision='ms')

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")

    def price_points_to_json(self, time, side, price, size, symbol):
        return {
            "measurement": "price",
            "tags": {
                "exchange": self.exchange,
                "symbol": symbol,
                "side": side,
            },
            "time": int(time),
            "fields": {
                f"price": price
            }
        }


class BackfillTradesBinance(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Binance"
        self.client = InfluxConnection.getInstance().archival_client_spotswap

    def write_trades(self, symbol, start_date, end_date, market=None):
        current_date = start_date
        start = current_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        points_to_write = []
        point_symbol = ""
        while start < end:
            filename = f"{symbol}-trades-{start}.zip"
            print(f"{filename}")
            if market == "fapi":
                url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{filename}"
                point_symbol = f"binance_futures_{symbol.lower()}"
            else:
                url = f"https://data.binance.vision/data/futures/cm/daily/trades/{symbol}/{filename}"
                point_symbol = f"binance_swap_{symbol.lower()}"
            try:
                response = requests.get(url, stream=True)
                with open(filename, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)
            except:
                continue
            if not os.path.exists(filename):
                break
            try:
                zf = zipfile.ZipFile(filename)
            except zipfile.BadZipfile:
                print(f"filename: {filename} is not uploaded yet")
                break
            zf.namelist()

            if market == "fapi":
                cols = ["id", "price", "size", "quote_qty", "time", "side"]
            else:
                cols = ["id", "price", "contracts", "size", "time", "side"]
            trades_df = pd.read_csv(zf.open(filename.replace('.zip', '.csv')),
                                    names=cols,
                                    header=None,
                                    skiprows=1,
                                    dtype={"id": str, "price": float, "size": float, "time": int})
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["dollars"] = trades_df["size"] * trades_df["price"]
            trades_df["time"] = self.milliseconds_to_microseconds(trades_df["time"])
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] is True else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=point_symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='u')
                    points_to_write = []

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='u')
                points_to_write = []

            current_date = current_date + timedelta(days=1)
            start = current_date.strftime('%Y-%m-%d')
            zf.close()
            os.remove(filename)
        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
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


class BackfillTradesBinanceCoin(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Binance"
        self.headers = {"X-MBX-APIKEY": os.getenv("BINANCE_KEY")}
        self.client = InfluxConnection.getInstance().archival_client_spotswap

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        from_id = None
        points_to_write = []
        while start < end:
            sleep(0.02)

            if from_id is None:
                url = f"https://dapi.binance.com/dapi/v1/historicalTrades?symbol={symbol}&limit=1000"
            else:
                url = f"https://dapi.binance.com/dapi/v1/historicalTrades?symbol={symbol}&limit=1000&fromId={from_id}"
            try:
                response = requests.get(url, headers=self.headers)
                data = response.json()
            except:
                continue
            if not data:
                break
            trades_df = pd.DataFrame(data)
            trades_df["baseQty"] = trades_df["baseQty"].astype("float")
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = abs(trades_df["baseQty"] * trades_df["price"])
            trades_df["time"] = self.milliseconds_to_nanoseconds(trades_df["time"])
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["isBuyerMaker"] is True else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=f"binance_swap_{symbol.lower()}")
                points_to_write.append(row_points)
                if len(points_to_write) >= 5000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []
            row_min = trades_df.loc[trades_df["id"] == trades_df["id"].min()]
            from_id = row_min['id'].values[0] - 1000
            end = int(trades_df["time"].min() / (1000 * 1000) - 1)
            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start / 1000)}")

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesBinanceUSDT(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Binance"
        self.headers = {"X-MBX-APIKEY": os.getenv("BINANCE_KEY")}
        self.client = InfluxConnection.getInstance().archival_client_spotswap

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp() * 1000)
        initial_end = int(end_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        from_id = None
        points_to_write = []
        while start < end:
            sleep(0.02)

            # from_id = 135336338
            if from_id is None:
                url = f"https://fapi.binance.com/fapi/v1/historicalTrades?symbol={symbol}&limit=1000"
            else:
                url = f"https://fapi.binance.com/fapi/v1/historicalTrades?symbol={symbol}&limit=1000&fromId={from_id}"

            try:
                response = requests.get(url, headers=self.headers)
                data = response.json()
            except:
                continue
            if not data:
                break
            if response.status_code >= 400:
                print(f"API Error:{data}")
                if from_id is not None:
                    from_id += 1
                continue
            trades_df = pd.DataFrame(data)
            trades_df["quoteQty"] = trades_df["quoteQty"].astype("float")
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = abs(trades_df["quoteQty"])
            trades_df["time"] = self.milliseconds_to_nanoseconds(trades_df["time"])
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["isBuyerMaker"] is True else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=f"binance_futures_{symbol.lower()}")
                points_to_write.append(row_points)
                if len(points_to_write) >= 5000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []
            row_min = trades_df.loc[trades_df["id"] == trades_df["id"].min()]
            from_id = row_min['id'].values[0] - 1000
            end = int(trades_df["time"].min() / (1000 * 1000) - 1)
            if end - initial_end > 1000 * 60 * 60 * 4:
                from_id -= 100000
            elif end - initial_end > 1000 * 60 * 60 * 1:
                from_id -= 10000
            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start / 1000)}")

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesBitfinex(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Bitfinex"

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        points_to_write = []
        while start < end:
            sleep(1)

            url = f"https://api-pub.bitfinex.com/v2/trades/{symbol}/hist?limit=10000&start={start}&end={end}&sort=1"
            try:
                response = requests.get(url)
                data = response.json()
            except:
                continue
            if not data:
                break

            trades_df = pd.DataFrame(data, columns=["Channel", "Timestamp", "Size", "Price"])
            trades_df["Price"] = trades_df["Price"].astype('float')
            trades_df["Dollars"] = abs(trades_df["Size"] * trades_df["Price"])
            trades_df["Timestamp"] = self.milliseconds_to_nanoseconds(trades_df["Timestamp"])
            trades_df["Timestamp"] = pd.Series(self.check_for_identical_timestamps(trades_df["Timestamp"].tolist()),
                                               dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["Size"] < 0 else "Bid"
                row_points = self.points_to_json(time=row["Timestamp"],
                                                 side=side,
                                                 price=row["Price"],
                                                 size=row["Dollars"],
                                                 symbol=symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []

            start = int(trades_df["Timestamp"].max() / (1000 * 1000) + 1)

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesBitflyer(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Bitflyer"

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        from_id = None
        points_to_write = []
        while start < end:
            sleep(1)

            if from_id is None:
                url = f"https://api.bitflyer.com/v1/executions?product_code={symbol}&count=500"
            else:
                url = f"https://api.bitflyer.com/v1/executions?product_code={symbol}&count=500&before={from_id}"
            try:
                response = requests.get(url)
                data = response.json()
            except Exception as e:
                print(e)
                break
            if not data:
                break
            try:
                trades_df = pd.DataFrame(data)
            except Exception as e:
                print(data)
                break
            trades_df["size"] = trades_df["size"].astype("float")
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = abs(trades_df["size"] * trades_df["price"])

            trades_df["time"] = pd.to_datetime(trades_df["exec_date"], unit="ns")
            trades_df["time"] = trades_df["time"].astype(np.int64)
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "SELL" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='n')
                    print("Saved to db")
                    points_to_write = []
            row_min = trades_df.loc[trades_df["id"] == trades_df["id"].min()]
            from_id = row_min['id'].values[0]
            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                print("Saved to db")
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start / 1000)}")

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesDeribit(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Deribit"
        self.headers = {}
        self.client = InfluxConnection.getInstance().staging_client_spotswap

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        points_to_write = []
        while start < end:
            sleep(0.02)
            url = f"https://history.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time?instrument_name={symbol}&count=10000&start_timestamp={start}&sorting=asc"
            print(url)
            try:
                response = requests.get(url, headers=self.headers)
                data = response.json()
            except Exception as e:
                print(f"API error response, killing: {e}")
                break
            if not data:
                break
            if response.status_code >= 400:
                print(f"API Error:{data}")
                continue
            trades_df = pd.DataFrame(data['result']['trades'])
            # trades_df["timestamp"] = trades_df["timestamp"].astype("float")
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["amount"] = abs(trades_df["amount"])
            trades_df["timestamp"] = self.milliseconds_to_nanoseconds(trades_df["timestamp"])
            trades_df["timestamp"] = pd.Series(self.check_for_identical_timestamps(trades_df["timestamp"].tolist()),
                                               dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["direction"] == "sell" else "Bid"
                row_points = self.points_to_json(time=row["timestamp"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["amount"],
                                                 symbol=symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 5000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []
            start = int(trades_df.iloc[-1]['timestamp'] / (1000 * 1000) - 1)
            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['timestamp'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['timestamp'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start / 1000)}")

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesFTX(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "FTX"

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp())  # API accepts timestamps in seconds only
        end = int(end_date.timestamp())
        points_to_write = []
        while start < end:
            sleep(1)
            url = f"https://ftx.com/api/markets/{symbol}/trades?start_time={start}&end_time={end}"
            try:
                response = requests.get(url)
                data = response.json()["result"]
            except:
                continue
            if not data:
                break
            trades_df = pd.DataFrame(data)
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = abs(trades_df["size"] * trades_df["price"])
            trades_df["time"] = trades_df["time"].astype('datetime64[ns]').astype("int")
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "sell" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []

            end = int(trades_df["time"].min() / (1000 * 1000 * 1000) - 1)

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start)}")
        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesHuobi(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "HuobiDMSwap"
        self.client = InfluxConnection.getInstance().archival_client_spotswap

    def write_trades(self, symbol, start_date, end_date):

        current_date = start_date
        start = current_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        points_to_write = []

        while start < end:

            filename = f"{symbol}-trades-{start}.zip"
            print(f"{filename}")
            if "USDT" in symbol:
                url = f"https://futures.huobi.com/data/trades/linear-swap/daily/{symbol}/{filename}"
            else:
                url = f"https://futures.huobi.com/data/trades/swap/daily/{symbol}/{filename}"
            try:
                response = requests.get(url, stream=True)
                with open(filename, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)
            except:
                continue
            if not os.path.exists(filename):
                break
            try:
                zf = zipfile.ZipFile(filename)
            except zipfile.BadZipfile:
                print(f"filename: {filename} is not uploaded yet")
                break
            zf.namelist()
            if "USDT" in symbol:
                cols = ["orderid", "time", "price", "contracts", "size", "notional", "side"]
            else:
                cols = ["orderid", "time", "price", "contracts", "size", "side"]
            trades_df = pd.read_csv(zf.open(filename.replace('.zip', '.csv')), names=cols, header=None)
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = trades_df["size"] * trades_df["price"]
            trades_df["time"] = self.milliseconds_to_microseconds(trades_df["time"])
            trades_df["time"] = trades_df["time"].astype(np.int64)
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "sell" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=f"{symbol}")
                points_to_write.append(row_points)
                if len(points_to_write) >= 5000:
                    self.client.write_points(points_to_write, time_precision='u')
                    points_to_write = []

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='u')
                points_to_write = []

            current_date = current_date + timedelta(days=1)
            start = current_date.strftime('%Y-%m-%d')
            zf.close()
            os.remove(filename)

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        swap_tickers = json.loads(requests.get("https://api.hbdm.com/v2/swap-ex/market/detail/batch_merged").text)
        print(f"Total Symbols: {len(swap_tickers)}")
        for ticker in swap_tickers['ticks']:
            if float(ticker['vol']) > 20000:
                print("Backfilling ", ticker['contract_code'])
                self.write_trades(symbol=ticker['contract_code'], start_date=start_date, end_date=end_date)

        usdt_tickers = json.loads(requests.get("https://api.hbdm.com/linear-swap-ex/market/detail/batch_merged").text)
        print(f"Total Symbols: {len(usdt_tickers)}")
        for ticker in usdt_tickers['ticks']:
            if float(ticker['vol']) > 20000:
                print("Backfilling ", ticker['contract_code'])
                self.write_trades(symbol=ticker['contract_code'], start_date=start_date, end_date=end_date)


class BackfillTradesKraken(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "Kraken"
        self.client = InfluxConnection.getInstance().archival_client_spotswap
        self.symbol_influx = {
            "BTC/EUR": "XBT/EUR",
            "BTC/USD": "XBT/USD"
        }

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp())  # API accepts timestamps in seconds only
        end = int(end_date.timestamp())
        points_to_write = []
        while start < end:
            sleep(1)
            url = f"https://api.kraken.com/0/public/Trades?pair={symbol}&since={start}"
            try:
                response = requests.get(url)
                data = response.json()["result"]
            except:
                continue
            if not data or not data[symbol] or len(data[symbol]) <= 1:
                break
            trades_df = pd.DataFrame(data[symbol], columns=["price", "size", "time", "side", "type", "misc"])
            trades_df['time'] = (trades_df['time'] * 1000 * 1000).astype(int)
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["size"] = trades_df["size"].astype("float")
            trades_df["dollars"] = abs(trades_df["size"] * trades_df["price"])

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "s" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=self.symbol_influx.get(symbol, symbol))
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='u')
                    points_to_write = []

            start = int(trades_df["time"].max() / (1000 * 1000) - 1)

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='u')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start)}")
        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


class BackfillTradesOkex(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
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

    def write_trades(self, symbol, start_date, end_date, contract_size=None):

        current_date = start_date
        start = current_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        points_to_write = []

        while start <= end:

            filename = f"{symbol}-trades-{start}.zip"
            print(f"{filename}")
            url = f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{start.replace('-', '')}/{filename}"
            try:
                response = requests.get(url, stream=True)
                with open(filename, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)
            except:
                continue
            if not os.path.exists(filename):
                break
            try:
                zf = zipfile.ZipFile(filename)
            except zipfile.BadZipfile:
                print(f"filename: {filename} is not uploaded yet")
                break
            zf.namelist()
            cols = ["id", "side", "contracts", "price", "time"]
            trades_df = pd.read_csv(zf.open(filename.replace('.zip', '.csv')),
                                    names=cols,
                                    header=None,
                                    encoding="ISO-8859-1", skiprows=1)
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["price"] = trades_df["price"].astype("float")
            cryptocurrency = symbol.split('-')[0]
            trades_df["dollars"] = [contract_size.get(cryptocurrency, lambda x: x)(x) for x in trades_df["price"]] * \
                                   trades_df["contracts"]
            trades_df["time"] = self.milliseconds_to_microseconds(trades_df["time"])
            trades_df["time"] = trades_df["time"].astype(np.int64)

            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()),
                                          dtype=object)  # dtype=object so it wont change the int timestamps to float and affect nanoseconds

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "SELL" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=f"okex_{symbol.lower()}")
                points_to_write.append(row_points)
                if len(points_to_write) >= 5000:
                    self.client.write_points(points_to_write, time_precision='u')
                    points_to_write = []

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='u')
                points_to_write = []

            current_date = current_date + timedelta(days=1)
            start = current_date.strftime('%Y-%m-%d')
            zf.close()
            os.remove(filename)

        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")

    def write_all(self, start_date, end_date):
        tickers = json.loads(requests.get("https://www.okex.com/api/v5/market/tickers?instType=SWAP").text)
        print(f"Total Symbols: {len(tickers['data'])}")
        for ticker in tickers['data']:
            if float(ticker['volCcy24h']) * float(ticker['open24h']) > 50000:
                self.write_trades(symbol=ticker['instId'], start_date=start_date, end_date=end_date)


class BackfillTradesWOO(BackfillTrades):

    def __init__(self):
        BackfillTrades.__init__(self)
        self.exchange = "WOO"
        self.client = InfluxConnection.getInstance().archival_client_spotswap

    def write_trades(self, symbol, start_date, end_date):
        start = int(start_date.timestamp())  # API accepts timestamps in seconds only
        end = int(end_date.timestamp())
        points_to_write = []
        while start < end:
            sleep(1)
            url = f"https://api.woo.org/v1/client/trades?symbol={symbol}&start_t={start}&end_t={end}"
            try:
                response = requests.get(url)
                data = response.json()["result"]
            except:
                continue
            if not data:
                break
            trades_df = pd.DataFrame(data)
            trades_df = trades_df.sort_values(["time"], ascending=True).reset_index(drop=True)
            trades_df["price"] = trades_df["price"].astype("float")
            trades_df["dollars"] = abs(trades_df["size"] * trades_df["price"])
            trades_df["time"] = trades_df["time"].astype('datetime64[ns]').astype("int")
            trades_df["time"] = pd.Series(self.check_for_identical_timestamps(trades_df["time"].tolist()), dtype=object)

            for index, row in trades_df.iterrows():
                side = "Ask" if row["side"] == "sell" else "Bid"
                row_points = self.points_to_json(time=row["time"],
                                                 side=side,
                                                 price=row["price"],
                                                 size=row["dollars"],
                                                 symbol=symbol)
                points_to_write.append(row_points)
                if len(points_to_write) >= 10000:
                    self.client.write_points(points_to_write, time_precision='n')
                    points_to_write = []

            end = int(trades_df["time"].min() / (1000 * 1000 * 1000) - 1)

            if len(points_to_write) > 0:
                self.client.write_points(points_to_write, time_precision='n')
                points_to_write = []
            print(
                f"Data successfully Backfilled from {datetime.fromtimestamp(trades_df['time'].min() / (1000 * 1000 * 1000))}"
                f" to {datetime.fromtimestamp(trades_df['time'].max() / (1000 * 1000 * 1000))}, "
                f"ends in {datetime.fromtimestamp(start)}")
        print(f"Backfill for exchange {self.exchange} an symbol {symbol} successfully ended")


def start_backfill(exchanges):
    backfill_exchanges = {
        "Bitfinex": BackfillTradesBitfinex,
        "FTX": BackfillTradesFTX,
        "BinanceCoin": BackfillTradesBinanceCoin,
        "BinanceUSDT": BackfillTradesBinanceUSDT,
        "BinanceAll": BackfillTradesBinance,
        "Bitflyer": BackfillTradesBitflyer,
        "Okex": BackfillTradesOkex,
        "OkexAll": BackfillTradesOkex,
        "Huobi": BackfillTradesHuobi,
        "HuobiAll": BackfillTradesHuobi,
        "Kraken": BackfillTradesKraken,
        "Deribit": BackfillTradesDeribit
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
                backfiller.write_trades(symbol=symbol["symbolName"],
                                        start_date=datetime.strptime(symbol["from"], "%Y-%m-%d %H:%M:%S"),
                                        end_date=datetime.strptime(symbol["to"], "%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exchanges", required=True, help="JSON of exchanges")
    args = parser.parse_args()
    exchanges = json.loads(args.exchanges)
    start_backfill(exchanges=exchanges)
