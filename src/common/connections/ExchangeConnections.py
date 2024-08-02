import gzip
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
from websocket import WebSocketApp
from pyee import EventEmitter
import time
import threading
import json
from enum import Enum
from src.common.utils.utils import Util

class TakerSide(Enum):
    SELL = 1
    BUY = 2


class TakerTrade:
    def __init__(self, price, size, side: TakerSide, timestamp, exchange, symbol):
        self.price = price
        self.size = size
        self.side = side
        self.timestamp = timestamp
        self.exchange = exchange
        self.symbol = symbol


class GenericExchange(EventEmitter):
    url = ""
    name = ""
    ws: WebSocketApp = None
    LIMIT_NO_UPDATES_SECONDS = 90
    CHECK_WEBSOCKET_UP_SECONDS = 65

    def __init__(self):
        super(GenericExchange, self).__init__()
        self.market_listeners = []
        self.message_handlers = {}
        self.logger = None
        self.last_started = 0
        self.logger = Util.get_logger(self.name)
        self.timestamp_last_update = 0
        self.data_timer = None

    def start(self):
        current_ms = round(time.time() * 1000)
        if self.ws is not None and self.ws.sock is not None and self.ws.sock.connected:
            self.logger.info("Websocket already or still connected, skipping restart...")
            return
        elif current_ms - self.last_started < 1000:
            self.logger.info(f"Already restarted {self.last_started - current_ms}ms ago, skipping restart...")
            return
        else:
            self.logger.info(f"Connecting...")
        self.last_started = current_ms
        self.ws = WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        t = threading.Thread(target=self.ws.run_forever)
        t.daemon = True
        t.start()
        t1 = threading.Thread(target=self.check_websocket_up)
        t1.daemon = True
        t1.start()

    def handle_no_data(self):
        self.logger.info(f"No data for {self.LIMIT_NO_UPDATES_SECONDS} seconds. Closing websocket connection.")
        self.ws.close()

    def update_data_timer(self):
        if self.data_timer is None:
            self.data_timer = threading.Timer(self.LIMIT_NO_UPDATES_SECONDS, self.handle_no_data)
            self.data_timer.start()
        else:
            self.data_timer.cancel()
            self.data_timer = threading.Timer(self.LIMIT_NO_UPDATES_SECONDS, self.handle_no_data)
            self.data_timer.start()

    def check_websocket_up(self):
        while True:
            time.sleep(self.CHECK_WEBSOCKET_UP_SECONDS)
            if self.ws is None or self.ws.sock is None or not self.ws.sock.connected:
                self.logger.info(f"Websocket connection down.")
                self.start()
                return

    def on_open(self, ws):
        pass

    def on_message(self, ws, message):
        pass

    def on_error(self, ws, error):
        self.logger.error(f"Error: {error}")
        self.start()

    def on_close(self, ws, close_status_code, close_reason):
        self.logger.info(f"Closing ws. Status code {close_status_code}, reason {close_reason}")
        self.start()


class BinanceUSDMFutures(GenericExchange):
    instance = None
    symbol = 'BTCUSDT'
    url = f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"
    name = "Binance"

    ws: WebSocketApp = None

    def __init__(self):
        super(BinanceUSDMFutures, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        if BinanceUSDMFutures.instance is None:
            BinanceUSDMFutures.instance = BinanceUSDMFutures()
        return BinanceUSDMFutures.instance

    def on_open(self, ws):
        self.logger.info(f"{self.symbol} connection opened...")
        for listener in self.market_listeners:
            listener()

    def on_ping_(self, ping_msg):
        self.ws.send('{"pong": %d}' % (ping_msg))
        # print(f"Received ping, not replying") #

    def subscribe_to_taker_trades(self):
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"{self.symbol}@aggTrade"] = self.on_taker_trade

    def on_message(self, ws, message):
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['s'] + "@" + jmessage["e"]](jmessage)
        except:
            pass

    def on_taker_trade(self, message):
        trade = TakerTrade(
            float(message['p']),
            round(float(message['q']) * float(message['p'])),
            TakerSide.SELL if message['m'] else TakerSide.BUY,
            message['E'],
            "Binance",
            message['s']
        )
        self.emit("trades", [trade])

    def current_price(self, symbol="BTC-USD"):
        return self.quotes[symbol]['ask'] if len(self.quotes) > 0 else None

    def on_pong(self, ws):
        pass


class BitMEX(GenericExchange):
    instance = None
    url = "wss://www.bitmex.com/realtime"
    name = "BitMEX"
    LEGACY_TICKS = {"XBTUSD": 0.01, "XBTZ17": 0.1, "XBJZ17": 1}

    def __init__(self):
        super(BitMEX, self).__init__()
        self.market_listeners = []
        self.message_handlers = {}
        self.instrument_api = lambda count, start: f"https://www.bitmex.com/api/v1/instrument?count={count}&start={start}&reverse=false"
        self.orderbooks = {}
        # currently assuming one instrument
        self.instrument, self.instrument_idx = self.get_instrument_and_idx()
        if self.instrument_idx is None:
            return

    def instrument_tick_size(self, instrument):
        return self.LEGACY_TICKS.get(instrument['symbol'], instrument['tickSize'])

    def get_instrument_and_idx(self, symbol="XBTUSD"):
        start = 0
        count = 500
        found = False
        while not found:
            instruments_list = json.loads(requests.get(self.instrument_api(count, start)).content)
            if len(instruments_list) == 0:
                return None, None
            for instrument_idx, instrument in enumerate(instruments_list):
                if instrument['symbol'] == symbol:
                    return instrument, instrument_idx
            start += 500

    def price_from_id(self, id, symbol="XBTUSD"):
        if self.instrument_idx is not None:
            return (100000000 * self.instrument_idx - id) * self.instrument_tick_size(self.instrument)
        instrument, instrument_idx = self.get_instrument_and_idx(symbol)
        return (100000000 * instrument_idx - id) * self.instrument_tick_size(instrument)

    @staticmethod
    def get_instance():
        if BitMEX.instance is None:
            BitMEX.instance = BitMEX()
        return BitMEX.instance

    def on_open(self, ws):
        self.logger.info(f"Connection opened...")
        for listener in self.market_listeners:
            listener()

    def subscribe_to_orderbook(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["orderBookL2:XBTUSD"]}')
        if self.subscribe_to_orderbook in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_orderbook)
        self.message_handlers['orderBookL2'] = self.on_orderbook

    def subscribe_to_taker_trades(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["trade:XBTUSD"]}')
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers['trade'] = self.on_taker_trade

    def on_message(self, ws, message):
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['table']](jmessage)
            self.update_timer()
        except:
            pass

    # def on_orderbook(self, message):
    #     if not message['data'] or len(message['data']) == 0:
    #         return
    #     symbol = message['data'][0]['symbol']
    #     book = self.orderbooks.get(symbol, MyDepthStatistics(80))
    #
    #     if message['action'] == 'partial':
    #         book = MyDepthStatistics(80)
    #
    #     for update in message['data']:
    #         size = 0
    #         if message['action'] != 'delete':
    #             size = update['size']
    #         book.add_order(int(self.price_from_id(update['id']) * 100), size, update['side'] == "Buy", 50)
    #
    #     if message['action'] == 'partial':
    #         book.book_built = True
    #
    #     self.orderbooks[symbol] = book

    def timestamp_to_millis(self, timestamp):
        return int(
            datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp() * 1000)

    def on_taker_trade(self, message):
        trades = []
        for trade in message['data']:
            trades.append(TakerTrade(
                trade['price'],
                trade['size'],
                TakerSide.SELL if trade['side'] == "Sell" else TakerSide.BUY,
                self.timestamp_to_millis(trade['timestamp']),
                "BitMEX",
                trade['symbol']
            ))
            # print(f"Time taker trade BitMEX {trade['timestamp']}")
        self.emit("trades", trades)

    def current_orderbook(self, symbol="XBTUSD"):
        book = self.orderbooks[symbol]
        result_askvolumes = np.cumsum(book.askVolumes())
        result_bidvolumes = np.cumsum(book.bidVolumes())
        temp = np.hstack([result_askvolumes, result_bidvolumes, book.added_volume_ask, book.added_volume_bid,
                          book.removed_volume_ask, book.removed_volume_bid])
        book.reset_statistics()
        return temp

    def current_price(self, symbol="XBTUSD"):
        return self.orderbooks[symbol]._askPrices[0] / 100 if len(self.orderbooks) > 0 else None

    def current_best_quotes(self, symbol="XBTUSD"):
        return self.orderbooks[symbol]._askPrices[0] / 100, self.orderbooks[symbol]._bidPrices[0] / 100

    def on_pong(self, ws):
        pass

    @staticmethod
    def historical_taker_trades(t0, t1, symbol="XBTUSD"):
        start_time = datetime.utcfromtimestamp(t0 // 1000).isoformat(sep='T', timespec='milliseconds') + 'Z'
        end_time = datetime.utcfromtimestamp(t1 // 1000).isoformat(sep='T', timespec='milliseconds') + 'Z'

        result = []
        query_counter = 0
        query_count = 1000
        while True:
            current_query_result = requests.get(
                f"https://www.bitmex.com/api/v1/trade?count={query_count}&start={query_counter}&reverse=false&symbol={symbol}&startTime={start_time}&endTime={end_time}").json()
            if not isinstance(current_query_result, list):
                print(f"BitMEX API call thew error, waiting 10s: {current_query_result['error']['message']}")
                time.sleep(10)
                continue

            result.append(current_query_result)
            if len(current_query_result) < query_count:
                break
            if query_counter == 0:
                query_counter += 1
            query_counter += query_count
            time.sleep(1.2)
            print(f"BitMEX API call nr {query_counter // query_count} succeeded.")

        result = [item for sublist in result for item in sublist]

        df = pd.DataFrame(result)
        if len(df) > 0:
            df = df.drop(
                columns=["tickDirection", "trdMatchID", "grossValue", "homeNotional", "foreignNotional", "symbol"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            c1 = {'Buy': 'Bid', 'Sell': 'Ask'}
            df.replace({'side': c1}, inplace=True)
            df.rename(columns={'timestamp': 'time'}, inplace=True)
        return df


class Bybit(GenericExchange):
    instance = None
    symbol = 'BTCUSD'
    name = "Bybit"
    url = "wss://stream.bybit.com/realtime"

    def __init__(self):
        super(Bybit, self).__init__()

    @staticmethod
    def get_instance():
        if Bybit.instance is None:
            Bybit.instance = Bybit()
        return Bybit.instance

    def on_open(self, ws):
        self.logger.info("Connection opened...")
        for listener in self.market_listeners:
            listener()

    def subscribe_to_taker_trades(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["trade.' + self.symbol + '"]}')
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"trade.{self.symbol}"] = self.on_taker_trade

    def on_message(self, ws, message):
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['topic']](jmessage)
            self.update_data_timer()
        except:
            pass

    def on_taker_trade(self, message):
        trades = []
        for trade in message['data']:
            trades.append(TakerTrade(
                trade['price'],
                trade['size'],
                TakerSide.SELL if trade['side'] == "Sell" else TakerSide.BUY,
                trade['trade_time_ms'],
                "Bybit",
                trade['symbol']
            ))
            # print(f"Time taker trade Bybit {trade['trade_time_ms']}")
        self.emit("trades", trades)

    def on_pong(self, ws):
        pass


class Deribit(GenericExchange):
    instance = None
    name = "Deribit"
    symbol = 'BTC-PERPETUAL'
    url = "wss://www.deribit.com/ws/api/v2"

    def __init__(self):
        super(Deribit, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        if Deribit.instance is None:
            Deribit.instance = Deribit()
        return Deribit.instance

    def on_open(self, ws):
        self.logger.info(f"{self.symbol} connection opened...")
        self.ws.send('{"jsonrpc" : "2.0",'
                     '"method" : "public/set_heartbeat","params" : {'
                     '"interval" : 30'
                     '}}')
        for listener in self.market_listeners:
            listener()

    def subscribe_to_orderbook(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"jsonrpc": "2.0", '
                         '"method": "public/subscribe", '
                         '"params": {"channels": ["book.' + self.symbol + '.100ms"]}}')
        if self.subscribe_to_orderbook in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_orderbook)
        self.message_handlers['book'] = self.on_orderbook

    def subscribe_to_quotes(self):

        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"jsonrpc": "2.0",'
                         ' "method": "public/subscribe",'
                         ' "params": {"channels": ["quote.' + self.symbol + '"]}}')
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f"quote.{self.symbol}"] = self.on_quote

    def on_message(self, ws, message):
        jmessage = json.loads(message)
        if 'test_request' in message:
            self.ws.send('{"jsonrpc": "2.0", "method": "public/test"]}')
            return
        try:
            self.message_handlers[jmessage['params']['channel']](jmessage)
            self.update_data_timer()
        except:
            pass

    def on_quote(self, message):
        if not message['params']:
            return
        symbol = message['params']['channel']
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = message['params']['data']['best_ask_price']
        current_quotes['bid'] = message['params']['data']['best_bid_price']
        self.quotes[symbol] = current_quotes

    # def on_orderbook(self, message):
    #     print(message['params']['data'])
    #     if not message['params'] or len(message['params']) == 0:
    #         return
    #     symbol = message['data'][0]['symbol']
    #     book = self.orderbooks.get(symbol, MyDepthStatistics(80))
    #
    #     if message['action'] == 'partial':
    #         book = MyDepthStatistics(80)
    #
    #     for update in message['data']:
    #         size = 0
    #         if message['action'] != 'delete':
    #             size = update['size']
    #         book.add_order(int(self.price_from_id(update['id']) * 100), size, update['side'] == "Buy", 50)
    #
    #     if message['action'] == 'partial':
    #         book.book_built = True
    #
    #     self.orderbooks[symbol] = book

    def current_price(self, symbol="BTC-PERPETUAL"):
        return self.quotes[f"quote.{symbol}"]['ask']

    def on_pong(self, ws):
        pass

    @staticmethod
    def historical_taker_trades(t0, t1, symbol="BTC-PERPETUAL"):
        result = []
        query_count = 1000
        print(f"{datetime.fromtimestamp(t0 / 1000)}")
        while t0 < t1:
            current_query_result = requests.get(
                f"https://test.deribit.com/api/v2/public/get_last_trades_by_instrument?count={1000}&instrument_name={symbol}&start_timestamp={t0}&end_timestamp={t1}&sorting=desc").json()
            if "error" in current_query_result.keys():
                print(
                    f"Deribit API call thew error, waiting 10s: {current_query_result['error']['message']} {current_query_result['error']['data']['reason']}")
                time.sleep(10)
                continue
            if len(current_query_result["result"]["trades"]) == 0:
                break
            result.append(current_query_result["result"]["trades"])

            time.sleep(1.2)
            print(
                f"Deribit API call from {datetime.fromtimestamp(current_query_result['result']['trades'][-1]['timestamp'] / 1000)} "
                f"to {datetime.fromtimestamp(t1 / 1000)} succeeded.")
            if t1 != current_query_result["result"]["trades"][-1]["timestamp"]:
                t1 = current_query_result["result"]["trades"][-1]["timestamp"]
            else:
                t1 = current_query_result["result"]["trades"][-1]["timestamp"] - 10
        result = [item for sublist in result for item in sublist]

        df = pd.DataFrame(result)
        if len(df) > 0:
            df = df.drop(
                columns=["tick_direction", "trade_id", "index_price", "trade_seq", "instrument_name"])
            df['timestamp'] = [datetime.utcfromtimestamp(row / 1000).isoformat(sep='T', timespec='milliseconds') + 'Z'
                               for row in df['timestamp']]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['amount'] = df['amount'].astype(int)
            c1 = {'buy': 'Bid', 'sell': 'Ask'}
            df.replace({'direction': c1}, inplace=True)
            df.rename(columns={'timestamp': 'time', 'direction': 'side', 'amount': 'size'}, inplace=True)
        return df


class OkexWeekly(GenericExchange):
    instance = None
    name = "OkexWeekly"
    uly = 'BTC-USD'
    alias = 'this_week'
    url = "wss://ws.okex.com:8443/ws/v5/public"
    rest_api_url = 'https://www.okex.com/'
    symbol = None
    KEEP_ALIVE = 25
    settlement_hour = 8

    def __init__(self):
        super(OkexWeekly, self).__init__()
        self.quotes = {}

    def set_symbol(self):
        instruments = requests.get(
            self.rest_api_url + f"api/v5/public/instruments?instType=FUTURES&uly={self.uly}").json()
        for instrument in instruments['data']:
            if instrument['alias'] == self.alias:
                self.symbol = instrument['instId']
                break

    @staticmethod
    def get_instance():
        if OkexWeekly.instance is None:
            OkexWeekly.instance = OkexWeekly()
        return OkexWeekly.instance

    def start(self):
        if self.symbol is not None:
            timestamp_now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            timestamp_settlement = int((datetime.strptime("20" + self.symbol.split("-")[-1], "%Y%m%d") + timedelta(
                hours=self.settlement_hour)).replace(tzinfo=timezone.utc).timestamp() * 1000)
            if timestamp_now > timestamp_settlement:
                time.sleep(300)
        self.set_symbol()
        super(OkexWeekly, self).start()
        time.sleep(3)
        t2 = threading.Thread(target=self.keep_alive)
        t2.start()

    def keep_alive(self):
        while True:
            timestamp_now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            timestamp_settlement = int((datetime.strptime("20" + self.symbol.split("-")[-1], "%Y%m%d") + timedelta(
                hours=self.settlement_hour)).replace(tzinfo=timezone.utc).timestamp() * 1000)
            if timestamp_now > timestamp_settlement:
                self.logger.info(f"Handling settlement of {self.symbol}")
                self.ws.close()

            time.sleep(self.KEEP_ALIVE)
            if self.ws.sock and self.ws.sock.connected:
                self.ws.send('{"op": "ping"}')

    def on_open(self, ws):
        self.logger.info(f"Okex {self.symbol} connection opened...")
        for listener in self.market_listeners:
            listener()

    def subscribe_to_quotes(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": [{"channel": "tickers", "instId": "%s"}]}' % (self.symbol))
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f"tickers.{self.symbol}"] = self.on_quote

    def on_message(self, ws, message):
        jmessage = json.loads(message)
        try:
            self.message_handlers[f"{jmessage['arg']['channel']}.{jmessage['arg']['instId']}"](jmessage)
            self.update_timer()
        except:
            pass

    def on_quote(self, message):
        if not message['data']:
            return
        symbol = message['arg']['instId']
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = float(message['data'][0]['askPx'])
        current_quotes['bid'] = float(message['data'][0]['bidPx'])
        self.quotes[symbol] = current_quotes

    def current_price(self, symbol=None):
        return self.quotes[self.symbol if symbol is None else symbol]['ask']

    def on_pong(self, ws):
        pass


class OkexQuarterly(OkexWeekly):
    instance = None
    name = "OkexQuarterly"
    uly = 'BTC-USD'
    alias = 'quarter'
    url = "wss://ws.okex.com:8443/ws/v5/public"
    rest_api_url = 'https://www.okex.com/'
    symbol = None

    def __init__(self):
        super(OkexQuarterly, self).__init__()

    @staticmethod
    def get_instance():
        if OkexQuarterly.instance is None:
            OkexQuarterly.instance = OkexQuarterly()
        return OkexQuarterly.instance

class HuobiDMCoinSwap(GenericExchange):
    instance = None
    symbol = 'BTC-USD'
    url = "wss://api.hbdm.com/swap-ws"
    name = "HuobiDMCoinSwap"

    ws: WebSocketApp = None

    def __init__(self):
        super(HuobiDMCoinSwap, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        if HuobiDMCoinSwap.instance is None:
            HuobiDMCoinSwap.instance = HuobiDMCoinSwap()
        return HuobiDMCoinSwap.instance


    def on_open(self, ws):
        self.logger.info(f"{self.symbol} connection opened...")
        for listener in self.market_listeners:
            listener()

    def on_ping_(self, ping_msg):
        self.ws.send('{"pong": %d}' % (ping_msg))
        #print(f"Received ping, not replying") #

    def subscribe_to_quotes(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"sub": "market.' + self.symbol + '.bbo"}')
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f'market.{self.symbol}.bbo'] = self.on_quote

    def subscribe_to_taker_trades(self):
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"sub": "market.' + self.symbol + '.trade.detail"}')
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"market.{self.symbol}.trade.detail"] = self.on_taker_trade

    def on_message(self, ws, message):
        message = gzip.decompress(message).decode("utf-8")
        jmessage = json.loads(message)
        if 'ping' in jmessage.keys():
            self.on_ping_(jmessage['ping'])
            return
        try:
            self.message_handlers[jmessage['ch']](jmessage)
            self.update_timer()
        except:
            pass

    def on_quote(self, message):
        if not message['tick']:
            return
        symbol = message['ch'].split(".")[1]
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = message['tick']['ask'][0]
        current_quotes['bid'] = message['tick']['bid'][0]
        self.quotes[symbol] = current_quotes

    def on_taker_trade(self, message):
        trades = []
        for trade in message['tick']['data']:
            trades.append(TakerTrade(
                trade['price'],
                round(trade['quantity'] * trade['price']),
                TakerSide.SELL if trade['direction'] == "sell" else TakerSide.BUY,
                message['tick']['data'][0]['ts'],
                "HuobiDM",
                message['ch'].split(".")[1]
            ))
        self.emit("trades", trades)

    def current_price(self, symbol="BTC-USD"):
        return self.quotes[symbol]['ask'] if len(self.quotes) > 0 else None

    def on_pong(self, ws):
        pass


class HuobiDMUSDTSwap(HuobiDMCoinSwap):
    instance = None
    symbol = 'BTC-USDT'
    url = "wss://api.hbdm.com/linear-swap-ws"
    name = "HuobiDMUSDTSwap"

    def __init__(self):
        super(HuobiDMUSDTSwap, self).__init__()

    @staticmethod
    def get_instance():
        if HuobiDMUSDTSwap.instance is None:
            HuobiDMUSDTSwap.instance = HuobiDMUSDTSwap()
        return HuobiDMUSDTSwap.instance

    def current_price(self, symbol="BTC-USDT"):
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        pass


class HuobiDMCoinFuturesQuarterly(HuobiDMCoinSwap):
    instance = None
    symbol = 'BTC_CQ'
    url = "wss://api.hbdm.com/ws"
    name = "HuobiDMFuturesQuarterly"

    def __init__(self):
        super(HuobiDMCoinFuturesQuarterly, self).__init__()

    @staticmethod
    def get_instance():
        if HuobiDMCoinFuturesQuarterly.instance is None:
            HuobiDMCoinFuturesQuarterly.instance = HuobiDMCoinFuturesQuarterly()
        return HuobiDMCoinFuturesQuarterly.instance

    def current_price(self, symbol="BTC_CQ"):
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        pass


class HuobiDMCoinFuturesWeekly(HuobiDMCoinFuturesQuarterly):
    instance = None
    symbol = 'BTC_CW'
    name = "HuobiDMFuturesQuarterly"

    def __init__(self):
        super(HuobiDMCoinFuturesWeekly, self).__init__()

    @staticmethod
    def get_instance():
        if HuobiDMCoinFuturesWeekly.instance is None:
            HuobiDMCoinFuturesWeekly.instance = HuobiDMCoinFuturesWeekly()
        return HuobiDMCoinFuturesWeekly.instance

    def current_price(self, symbol="BTC_CW"):
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        pass
