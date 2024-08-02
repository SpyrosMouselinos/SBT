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

# Taker side is a side that can be used to do things such as get rid of taker
class TakerSide(Enum):
    SELL = 1
    BUY = 2


# This is called when you want to taker trades. The trade is returned
class TakerTrade:
    def __init__(self, price, size, side: TakerSide, timestamp, exchange, symbol):
        """
         @brief Initializes the taker with the given parameters. This is the place where you can set the values in the order.
         @param price The price of the order. It must be greater than or equal to 0.
         @param size The size of the order. It must be greater than 0.
         @param side The side of the order. It must be one of the : class : ` TakerSide ` values.
         @param timestamp The timestamp of the order. It must be a datetime. datetime object.
         @param exchange The exchange that will be used to pay the order.
         @param symbol The symbol that will be used to pay the order
        # This is a generic exchange. You can add or remove exchanges using this
        """
        self.price = price
        self.size = size
        self.side = side
        self.timestamp = timestamp
        self.exchange = exchange
        self.symbol = symbol


# This is a generic exchange.
class GenericExchange(EventEmitter):
    url = ""
    name = ""
    ws: WebSocketApp = None
    LIMIT_NO_UPDATES_SECONDS = 90
    CHECK_WEBSOCKET_UP_SECONDS = 65

    def __init__(self):
        """
         @brief Initialize exchange. This is called by __init__ and should not be called directly
        """
        super(GenericExchange, self).__init__()
        self.market_listeners = []
        self.message_handlers = {}
        self.logger = None
        self.last_started = 0
        self.logger = Util.get_logger(self.name)
        self.timestamp_last_update = 0
        self.data_timer = None

    def start(self):
        """
         @brief Start the websocket. This is the method that should be called by the client when it wants to connect to the websocket.
         @return True if the connection was successful False otherwise. If we are already connected or if the time since the last start is less than 1000 seconds it will restart
        """
        current_ms = round(time.time() * 1000)
        # If we re connected to the websocket we re connected to.
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
        """
         @brief Handle no data. Closes the websocket connection and logs a message to the
        """
        self.logger.info(f"No data for {self.LIMIT_NO_UPDATES_SECONDS} seconds. Closing websocket connection.")
        self.ws.close()

    def update_data_timer(self):
        """
         @brief Update the timer to avoid too frequent updates. This is called every N seconds
        """
        # Start a timer for the data update.
        if self.data_timer is None:
            self.data_timer = threading.Timer(self.LIMIT_NO_UPDATES_SECONDS, self.handle_no_data)
            self.data_timer.start()
        else:
            self.data_timer.cancel()
            self.data_timer = threading.Timer(self.LIMIT_NO_UPDATES_SECONDS, self.handle_no_data)
            self.data_timer.start()

    def check_websocket_up(self):
        """
         @brief Check if we are connected to the websocket. This is a blocking function.
         @return True if we are connected False otherwise. In case of error it will return
        """
        # Sleeps for a websocket connection down.
        while True:
            time.sleep(self.CHECK_WEBSOCKET_UP_SECONDS)
            # Start the websocket connection. If the websocket is not connected we ll start the connection down.
            if self.ws is None or self.ws.sock is None or not self.ws.sock.connected:
                self.logger.info(f"Websocket connection down.")
                self.start()
                return

    def on_open(self, ws):
        """
         @brief Called when WebSocket connection is opened. Subclasses should override this method to perform actions such as setting up WebSocket connection and sending WebSocket data to client.
         @param ws WebSocket connection that was opened ( used to determine which channel is connected
        """
        pass

    def on_message(self, ws, message):
        """
         @brief Called when a message is received. This is a no - op for websockets
         @param ws WebSocket that received the message
         @param message Message that was received
        """
        pass

    def on_error(self, ws, error):
        """
         @brief Called when there is an error. This is a no - op for this client
         @param ws WebSocket that sent the error
         @param error Error that was received from the websocket or None if
        """
        self.logger.error(f"Error: {error}")
        self.start()

    def on_close(self, ws, close_status_code, close_reason):
        """
         @brief Called when WebSocket connection is closed. This is the last step in the websocket handshake process.
         @param ws WebSocket instance that was closed. Used to get status code and reason
         @param close_status_code WebSocket close status code.
         @param close_reason WebSocket close reason. Note that this may be different from the status code
        """
        self.logger.info(f"Closing ws. Status code {close_status_code}, reason {close_reason}")
        self.start()


# This class is called by USDMFutures to get binance information.
class BinanceUSDMFutures(GenericExchange):
    instance = None
    symbol = 'BTCUSDT'
    url = f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"
    name = "Binance"

    ws: WebSocketApp = None

    def __init__(self):
        """
         @brief Initialize the USDMFutures class. This is the base class for BinanceUSD
        """
        super(BinanceUSDMFutures, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        """
         @brief Get the singleton instance of BinanceUSDMFutures. This is a singleton so we don't have to worry about lazy instantiation.
         @return The singleton instance of BinanceUSDMFutures
        """
        # This is a singleton instance of BinanceUSDMFutures.
        if BinanceUSDMFutures.instance is None:
            BinanceUSDMFutures.instance = BinanceUSDMFutures()
        return BinanceUSDMFutures.instance

    def on_open(self, ws):
        """
         @brief Called when connection is opened. This is the place to do things like log the open and close the connection
         @param ws WebSocket that was used to
        """
        self.logger.info(f"{self.symbol} connection opened...")
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def on_ping_(self, ping_msg):
        """
         @brief Called when a ping message is received. Sends a pong to the server
         @param ping_msg The number of pings
        """
        self.ws.send('{"pong": %d}' % (ping_msg))
        # print(f"Received ping, not replying") #

    def subscribe_to_taker_trades(self):
        """
         @brief Subscribe to trades from the market. This is a way to get trade data to be used when you have a trade that is not part of the market.
         @return Nothing. Instead subscribes to trade data and returns
        """
        # If subscription_to_taker_trades is in market_listeners return true.
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"{self.symbol}@aggTrade"] = self.on_taker_trade

    def on_message(self, ws, message):
        """
         @brief Called when a message is received. This is the method that will be called by Twisted for every message received on the websocket
         # This is a hack to make it easier to read the MESA code
         @param ws The websocket that received the message
         @param message The message that was recieved from the
        """
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['s'] + "@" + jmessage["e"]](jmessage)
        except:
            pass

    def on_taker_trade(self, message):
        """
         @brief Event handler for trades sent by Taker. This is used to create a trade based on the trade type
         @param message Dictionary containing key / value pairs from the
        """
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
        """
         @brief Returns the current price of a symbol. This is an unfortunate way to get the current price of a symbol without having to re - ask the quote
         @param symbol Symbol to look up ( BTC - USD )
         @return Current price or None if there is no quote for
        """
        return self.quotes[symbol]['ask'] if len(self.quotes) > 0 else None

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass


# This is a BitMEX exchange.
class BitMEX(GenericExchange):
    instance = None
    url = "wss://www.bitmex.com/realtime"
    name = "BitMEX"
    LEGACY_TICKS = {"XBTUSD": 0.01, "XBTZ17": 0.1, "XBJZ17": 1}

    def __init__(self):
        """
         @brief Initialize BitMEX. This is the first method that you need to call in your subclass.
         @return True if initialization was successful False otherwise. Note that you can't do anything that depends on the constructor
        """
        super(BitMEX, self).__init__()
        self.market_listeners = []
        self.message_handlers = {}
        self.instrument_api = lambda count, start: f"https://www.bitmex.com/api/v1/instrument?count={count}&start={start}&reverse=false"
        self.orderbooks = {}
        # currently assuming one instrument
        self.instrument, self.instrument_idx = self.get_instrument_and_idx()
        # Returns the instrument index if the instrument is not None.
        if self.instrument_idx is None:
            return

    def instrument_tick_size(self, instrument):
        """
         @brief Return the tick size of an instrument. This is based on the symbol and ticksize
         @param instrument Instrument to look up tick size for
         @return Instrument tick size in Mpc / h or None if not
        """
        return self.LEGACY_TICKS.get(instrument['symbol'], instrument['tickSize'])

    def get_instrument_and_idx(self, symbol="XBTUSD"):
        """
         @brief Get instrument and index. This is a wrapper around instrument_api to make it easier to use in tests
         @param symbol symbol of the instrument to get
         @return tuple of instrument and index or None if not found
        """
        start = 0
        count = 500
        found = False
        # Returns the instrument and instrument index of the first instrument in the list.
        while not found:
            instruments_list = json.loads(requests.get(self.instrument_api(count, start)).content)
            # Returns the instrument list and the instrument list.
            if len(instruments_list) == 0:
                return None, None
            # Returns instrument instrument_idx instrument_idx.
            for instrument_idx, instrument in enumerate(instruments_list):
                # Returns instrument instrument_idx instrument_idx. instrument symbol
                if instrument['symbol'] == symbol:
                    return instrument, instrument_idx
            start += 500

    def price_from_id(self, id, symbol="XBTUSD"):
        """
         @brief Get price from id. This is used to calculate the price of a tick based on the instrument and tick size
         @param id id of the tick to calculate price for
         @param symbol symbol of the tick to calculate price for ( default " XBTUSD " )
         @return price of the tick ( int ) or None if not
        """
        # Returns the number of ticks per instrument.
        if self.instrument_idx is not None:
            return (100000000 * self.instrument_idx - id) * self.instrument_tick_size(self.instrument)
        instrument, instrument_idx = self.get_instrument_and_idx(symbol)
        return (100000000 * instrument_idx - id) * self.instrument_tick_size(instrument)

    @staticmethod
    def get_instance():
        """
         @brief Returns the BitMEX instance. This is a singleton so you can access it multiple times without redefining it.
         @return A : class : ` BitMEX ` instance
        """
        # Create a new instance of BitMEX.
        if BitMEX.instance is None:
            BitMEX.instance = BitMEX()
        return BitMEX.instance

    def on_open(self, ws):
        """
         @brief Called when WebSocket connection is opened. This is the place to do things that need to be done in response to a WebSocket connection opening
         @param ws WebSocket connection that was
        """
        self.logger.info(f"Connection opened...")
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def subscribe_to_orderbook(self):
        """
         @brief Subscribe to orderbook events. This is called by L { market_on_orderbook } when it is known that we are subscribing to orderbook events.
         @return A deferred whose callback is invoked when the subscription is complete
        """
        # Subscribe to the websocket.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["orderBookL2:XBTUSD"]}')
        # Returns true if the subscription_to_orderbook is in the market_listeners list.
        if self.subscribe_to_orderbook in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_orderbook)
        self.message_handlers['orderBookL2'] = self.on_orderbook

    def subscribe_to_taker_trades(self):
        """
         @brief Subscribe to trades from XBTUSD. This is a way to get trade data from market
         @return True if subscription was successful
        """
        # Subscribe to the websocket.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["trade:XBTUSD"]}')
        # If subscription_to_taker_trades is in market_listeners return true.
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers['trade'] = self.on_taker_trade

    def on_message(self, ws, message):
        """
         @brief Called when a message is received from the websocket. This is the callback for websocket. py
         @param ws The websocket that received the message
         @param message The message that was
        """
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['table']](jmessage)
            self.update_timer()
        except:
            pass


    def timestamp_to_millis(self, timestamp):
        """
         @brief Converts a timestamp to milliseconds. This is useful for determining how long an event took to be in the future
         @param timestamp The timestamp to convert.
         @return The number of milliseconds since midnight January 1 1970 00 : 00
        """
        return int(
            datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc).timestamp() * 1000)

    def on_taker_trade(self, message):
        """
         @brief When taker trades are received. This is used to get the trade data from BitMEX
         @param message Message that triggered the
        """
        trades = []
        # This method will append the trade data to the trades list.
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
        """
         @brief Returns the sum of ask and bid volumes for the given symbol. This is useful for determining how much to ask and bid in a given time period
         @param symbol Symbol for which to get the sum
         @return Array of sum of
        """
        book = self.orderbooks[symbol]
        result_askvolumes = np.cumsum(book.askVolumes())
        result_bidvolumes = np.cumsum(book.bidVolumes())
        temp = np.hstack([result_askvolumes, result_bidvolumes, book.added_volume_ask, book.added_volume_bid,
                          book.removed_volume_ask, book.removed_volume_bid])
        book.reset_statistics()
        return temp

    def current_price(self, symbol="XBTUSD"):
        """
         @brief Get the current price of a symbol. This is used to determine how much to ask for the next trade
         @param symbol The symbol to look up
         @return The current price or None if there are no orders
        """
        return self.orderbooks[symbol]._askPrices[0] / 100 if len(self.orderbooks) > 0 else None

    def current_best_quotes(self, symbol="XBTUSD"):
        """
         @brief Returns the current best quotes for the given symbol. This is useful for determining how much to ask and bid for a given symbol
         @param symbol The symbol to get the best quotes for
         @return A tuple of ( ask bid ) where ask is the number of asks and bid is the number of bid
        """
        return self.orderbooks[symbol]._askPrices[0] / 100, self.orderbooks[symbol]._bidPrices[0] / 100

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

    @staticmethod
    # This function returns a list of historical trades that are in the past. The list is sorted by decreasing price
    def historical_taker_trades(t0, t1, symbol="XBTUSD"):
        """
         @brief Queries BitMEX to get trades between t0 and t1.
         @param t0 start time of the trade in milliseconds since 1970 - 01 - 01T00 : 00 : 00Z
         @param t1 end time of the trade in milliseconds since 1970 - 01 - 01T00 : 00 : 00Z
         @param symbol trade symbol to query. Default XBTUSD
         @return list of dicts each dict has the following keys : price ( str
        """
        start_time = datetime.utcfromtimestamp(t0 // 1000).isoformat(sep='T', timespec='milliseconds') + 'Z'
        end_time = datetime.utcfromtimestamp(t1 // 1000).isoformat(sep='T', timespec='milliseconds') + 'Z'

        result = []
        query_counter = 0
        query_count = 1000
        # BitMEX API call to BitMEX API.
        while True:
            current_query_result = requests.get(
                f"https://www.bitmex.com/api/v1/trade?count={query_count}&start={query_counter}&reverse=false&symbol={symbol}&startTime={start_time}&endTime={end_time}").json()
            # This function will wait for the current query result and wait for 10s to complete.
            if not isinstance(current_query_result, list):
                print(f"BitMEX API call thew error, waiting 10s: {current_query_result['error']['message']}")
                time.sleep(10)
                continue

            result.append(current_query_result)
            # If there are more results in the query_count query_count then break the query.
            if len(current_query_result) < query_count:
                break
            # Increment the query counter by one.
            if query_counter == 0:
                query_counter += 1
            query_counter += query_count
            time.sleep(1.2)
            print(f"BitMEX API call nr {query_counter // query_count} succeeded.")

        result = [item for sublist in result for item in sublist]

        df = pd.DataFrame(result)
        # Drop tick direction trdMatchID grossValue homeNotional foreignNotional symbol
        if len(df) > 0:
            df = df.drop(
                columns=["tickDirection", "trdMatchID", "grossValue", "homeNotional", "foreignNotional", "symbol"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            c1 = {'Buy': 'Bid', 'Sell': 'Ask'}
            df.replace({'side': c1}, inplace=True)
            df.rename(columns={'timestamp': 'time'}, inplace=True)
        return df


# This is a Bybit exchange.
class Bybit(GenericExchange):
    instance = None
    symbol = 'BTCUSD'
    name = "Bybit"
    url = "wss://stream.bybit.com/realtime"

    def __init__(self):
        """
         @brief Initialize bybit. This is called by __init__ and can be overridden in subclasses
        """
        super(Bybit, self).__init__()

    @staticmethod
    def get_instance():
        """
         @brief Get the singleton instance of : class : ` Bybit `. This is a singleton so you can access it without instantiating it yourself.
         @return The singleton instance of : class : ` Bybit `
        """
        # Create a new instance of Bybit.
        if Bybit.instance is None:
            Bybit.instance = Bybit()
        return Bybit.instance

    def on_open(self, ws):
        """
         @brief Called when connection is opened. This is the place to do things that need to be done in response to a WebSocket connection being opened
         @param ws WebSocket connection that was
        """
        self.logger.info("Connection opened...")
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def subscribe_to_taker_trades(self):
        """
         @brief Subscribe to trades from the market. This is called when you have a trade that is marked as " Taker "
         @return True if the subscription was successful False
        """
        # Subscribe to the websocket socket.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": ["trade.' + self.symbol + '"]}')
        # If subscription_to_taker_trades is in market_listeners return true.
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"trade.{self.symbol}"] = self.on_taker_trade

    def on_message(self, ws, message):
        """
         @brief Called when a message is received from the websocket. This is the method that will be called in response to any incoming messages
         @param ws The websocket that received the message
         @param message The message that was recieved from the
        """
        jmessage = json.loads(message)
        try:
            self.message_handlers[jmessage['topic']](jmessage)
            self.update_data_timer()
        except:
            pass

    def on_taker_trade(self, message):
        """
         @brief When taker trades are received. This is used to get the time trade bybit and send it to the client
         @param message Message that triggered the
        """
        trades = []
        # Trades are the trade data.
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
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

# This is the Deribit exchange
class Deribit(GenericExchange):
    instance = None
    name = "Deribit"
    symbol = 'BTC-PERPETUAL'
    url = "wss://www.deribit.com/ws/api/v2"

    def __init__(self):
        """
         @brief Initialize Deribit. This is called by __init__ to initialize the class
        """
        super(Deribit, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        """
         @brief Returns the Deribit instance. This is a singleton so we don't have to worry about this every time we need it.
         @return The singleton instance of Deribit that has been
        """
        # Create a new instance of Deribit.
        if Deribit.instance is None:
            Deribit.instance = Deribit()
        return Deribit.instance

    def on_open(self, ws):
        """
         @brief Called when connection is opened. Sends a heartbeat to the Market Server
         @param ws WebSocket connection to the
        """
        self.logger.info(f"{self.symbol} connection opened...")
        self.ws.send('{"jsonrpc" : "2.0",'
                     '"method" : "public/set_heartbeat","params" : {'
                     '"interval" : 30'
                     '}}')
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def subscribe_to_orderbook(self):
        """
         @brief Subscribe to orderbook events. This is called by Market when it is time to start receiving orders
         @return True if subscription was successful False
        """
        # Subscribe to the book.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"jsonrpc": "2.0", '
                         '"method": "public/subscribe", '
                         '"params": {"channels": ["book.' + self.symbol + '.100ms"]}}')
        # Returns true if the subscription_to_orderbook is in the market_listeners list.
        if self.subscribe_to_orderbook in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_orderbook)
        self.message_handlers['book'] = self.on_orderbook

    def subscribe_to_quotes(self):
        """
        @brief Subscribe to quotes. Market will be able to subscribe to quotes by sending a message to the websocket.
        @return True if successful False otherwise. This is a blocking call
        """

        # Subscribe to the symbol.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"jsonrpc": "2.0",'
                         ' "method": "public/subscribe",'
                         ' "params": {"channels": ["quote.' + self.symbol + '"]}}')
        # Returns true if the market_listeners have been subscribed to quotes.
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f"quote.{self.symbol}"] = self.on_quote

    def on_message(self, ws, message):
        """
         @brief Called when a message is received. This is the method that will be called by Twisted when a WebSocket connection is made to the server.
         @param ws The websocket that received the message. It is used to get the channel to which the message belongs
         @param message The message that was received
         @return None to indicate that the message was handled or a list of messages that were
        """
        jmessage = json.loads(message)
        # This method is called when the test_request is received from the server.
        if 'test_request' in message:
            self.ws.send('{"jsonrpc": "2.0", "method": "public/test"]}')
            return
        try:
            self.message_handlers[jmessage['params']['channel']](jmessage)
            self.update_data_timer()
        except:
            pass

    def on_quote(self, message):
        """
         @brief Updates the price of the quote. Stores the price of the ask and bid in self. quotes
         @param message Message that triggered the event.
         @return None but does nothing if there is no message to
        """
        # If params is empty return nothing.
        if not message['params']:
            return
        symbol = message['params']['channel']
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = message['params']['data']['best_ask_price']
        current_quotes['bid'] = message['params']['data']['best_bid_price']
        self.quotes[symbol] = current_quotes


    def current_price(self, symbol="BTC-PERPETUAL"):
        """
         @brief Get the current price of a symbol. By default this will return BTC - PERPETUAL
         @param symbol Symbol to look up.
         @return Price of the symbol or None if not found ( default
        """
        return self.quotes[f"quote.{symbol}"]['ask']

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

    @staticmethod
    # This function returns a list of historical trades that are in the past. The list is sorted by decreasing price
    def historical_taker_trades(t0, t1, symbol="BTC-PERPETUAL"):
        """
         @brief Queries deribit for trades between t0 and t1. This is a helper function for historical_trades
         @param t0 start timestamp of the interval to query ( inclusive ). It should be a datetime. datetime object
         @param t1 end timestamp of the interval to query ( inclusive ). It should be a datetime. datetime object
         @param symbol symbol of the ticker. Defaults to BTC - PERPETUAL
         @return list of dicts each dict has two keys : price ( str
        """
        result = []
        query_count = 1000
        print(f"{datetime.fromtimestamp(t0 / 1000)}")
        # Get the last trades from deribit. com. deribit. com
        while t0 < t1:
            current_query_result = requests.get(
                f"https://test.deribit.com/api/v2/public/get_last_trades_by_instrument?count={1000}&instrument_name={symbol}&start_timestamp={t0}&end_timestamp={t1}&sorting=desc").json()
            # This function will wait for the error in the query result.
            if "error" in current_query_result.keys():
                print(
                    f"Deribit API call thew error, waiting 10s: {current_query_result['error']['message']} {current_query_result['error']['data']['reason']}")
                time.sleep(10)
                continue
            # If there are no trades in the query result return the first trades.
            if len(current_query_result["result"]["trades"]) == 0:
                break
            result.append(current_query_result["result"]["trades"])

            time.sleep(1.2)
            print(
                f"Deribit API call from {datetime.fromtimestamp(current_query_result['result']['trades'][-1]['timestamp'] / 1000)} "
                f"to {datetime.fromtimestamp(t1 / 1000)} succeeded.")
            # The timestamp of the last time the query result is returned.
            if t1 != current_query_result["result"]["trades"][-1]["timestamp"]:
                t1 = current_query_result["result"]["trades"][-1]["timestamp"]
            else:
                t1 = current_query_result["result"]["trades"][-1]["timestamp"] - 10
        result = [item for sublist in result for item in sublist]

        df = pd.DataFrame(result)
        # Drop the timestamp and amount columns and drop the timestamp columns.
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

# This is the OkexWeekly exchange
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
        """
         @brief Initialize instance variables and data. Called by __init__ to initialize instance variables
        """
        super(OkexWeekly, self).__init__()
        self.quotes = {}

    def set_symbol(self):
        """
         @brief Get symbol from FUTURES API and store it in self. symbol
        """
        instruments = requests.get(
            self.rest_api_url + f"api/v5/public/instruments?instType=FUTURES&uly={self.uly}").json()
        # Find the symbol of the instrument
        for instrument in instruments['data']:
            # If the instrument is an alias then the symbol is set to the symbol of the instrument.
            if instrument['alias'] == self.alias:
                self.symbol = instrument['instId']
                break

    @staticmethod
    def get_instance():
        """
         @brief Get the singleton instance of OkexWeekly. This is a singleton so we don't have to worry about lazy instantiation.
         @return The singleton instance of OkexWeekly or None
        """
        # Create a new instance of the OkexWeekly class.
        if OkexWeekly.instance is None:
            OkexWeekly.instance = OkexWeekly()
        return OkexWeekly.instance

    def start(self):
        """
         @brief Start OkexWeekly and check settlement_hour to see if we need to
        """
        # This method will sleep for a certain amount of time.
        if self.symbol is not None:
            timestamp_now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            timestamp_settlement = int((datetime.strptime("20" + self.symbol.split("-")[-1], "%Y%m%d") + timedelta(
                hours=self.settlement_hour)).replace(tzinfo=timezone.utc).timestamp() * 1000)
            # Sleeps for a certain amount of time.
            if timestamp_now > timestamp_settlement:
                time.sleep(300)
        self.set_symbol()
        super(OkexWeekly, self).start()
        time.sleep(3)
        t2 = threading.Thread(target=self.keep_alive)
        t2.start()

    def keep_alive(self):
        """
         @brief Keep Alive to check settlement of symbol in order to avoid deadlock
        """
        # This method is called by the websocket to handle settlement of the symbol.
        while True:
            timestamp_now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            timestamp_settlement = int((datetime.strptime("20" + self.symbol.split("-")[-1], "%Y%m%d") + timedelta(
                hours=self.settlement_hour)).replace(tzinfo=timezone.utc).timestamp() * 1000)
            # Close the websocket when the settlement of the symbol is reached.
            if timestamp_now > timestamp_settlement:
                self.logger.info(f"Handling settlement of {self.symbol}")
                self.ws.close()

            time.sleep(self.KEEP_ALIVE)
            # pingping the websocket if the websocket is connected
            if self.ws.sock and self.ws.sock.connected:
                self.ws.send('{"op": "ping"}')

    def on_open(self, ws):
        """
         @brief Called when Okex connects to the server. This is the place to do things like log the open and close the connection
         @param ws WebSocket that was used to
        """
        self.logger.info(f"Okex {self.symbol} connection opened...")
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def subscribe_to_quotes(self):
        """
         @brief Subscribe to quotes. This is called by : meth : ` tickers_quote `
         @return True if successful False
        """
        # Subscribe to the websocket.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"op": "subscribe", "args": [{"channel": "tickers", "instId": "%s"}]}' % (self.symbol))
        # Returns true if the market_listeners have been subscribed to quotes.
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f"tickers.{self.symbol}"] = self.on_quote

    def on_message(self, ws, message):
        """
         @brief Called when a message is received from the websocket. This is the callback for websocket. on_message
         @param ws The websocket that received the message
         @param message The message that was
        """
        jmessage = json.loads(message)
        try:
            self.message_handlers[f"{jmessage['arg']['channel']}.{jmessage['arg']['instId']}"](jmessage)
            self.update_timer()
        except:
            pass

    def on_quote(self, message):
        """
         @brief Called when a QUOTATION message is received. Stores the ask and bid in self. quotes
         @param message The message that triggered the event. Should contain the instrument identifier
         @return None Side effects :
        """
        # If message is not a message
        if not message['data']:
            return
        symbol = message['arg']['instId']
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = float(message['data'][0]['askPx'])
        current_quotes['bid'] = float(message['data'][0]['bidPx'])
        self.quotes[symbol] = current_quotes

    def current_price(self, symbol=None):
        """
         @brief Get the current price of the quote. If no symbol is given it will return the current price of the symbol
         @param symbol Symbol to get the price for
         @return Current price of the quote or the current price of
        """
        return self.quotes[self.symbol if symbol is None else symbol]['ask']

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

# This is the OkexQuarterly exchange
class OkexQuarterly(OkexWeekly):
    instance = None
    name = "OkexQuarterly"
    uly = 'BTC-USD'
    alias = 'quarter'
    url = "wss://ws.okex.com:8443/ws/v5/public"
    rest_api_url = 'https://www.okex.com/'
    symbol = None

    def __init__(self):
        """
         @brief Initialize OkexQuarterly. This is called before __init__ to avoid problems with class attributes
        """
        super(OkexQuarterly, self).__init__()

    @staticmethod
    def get_instance():
        """
         @brief Get the singleton instance of OkexQuarterly. This is a singleton so we don't have to worry about lazy instantiation.
         @return an instance of OkexQuarterly or None if there is
        """
        # This method is used to create a new instance of the class.
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
        """
         @brief Initialize class and set quotes to empty dict. @ In None @ Out __init__ instance
        """
        super(HuobiDMCoinSwap, self).__init__()
        self.quotes = {}

    @staticmethod
    def get_instance():
        """
         @brief Singleton to get HuobiDMCoinSwap instance. This is needed because we need to create a class in the context of the test class which is the same for all tests.
         @return The instance of the class ( or None if not initialized
        """
        # HuobiDMCoinSwap. instance is not used.
        if HuobiDMCoinSwap.instance is None:
            HuobiDMCoinSwap.instance = HuobiDMCoinSwap()
        return HuobiDMCoinSwap.instance


    def on_open(self, ws):
        """
         @brief Called when connection is opened. This is the place to do things like log the open and close the connection
         @param ws WebSocket that was used to
        """
        self.logger.info(f"{self.symbol} connection opened...")
        # Calls all registered listeners for all market listeners.
        for listener in self.market_listeners:
            listener()

    def on_ping_(self, ping_msg):
        """
         @brief Called when a ping message is received. Sends a pong to the server
         @param ping_msg The number of pings
        """
        self.ws.send('{"pong": %d}' % (ping_msg))
        #print(f"Received ping, not replying") #

    def subscribe_to_quotes(self):
        """
         @brief Subscribe to quotes. This is called by Market when it is ready to receive quotes.
         @return True if subscription was successful False if it was already
        """
        # Send a bbo message to the websocket.
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"sub": "market.' + self.symbol + '.bbo"}')
        # Returns true if the market_listeners have been subscribed to quotes.
        if self.subscribe_to_quotes in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_quotes)
        self.message_handlers[f'market.{self.symbol}.bbo'] = self.on_quote

    def subscribe_to_taker_trades(self):
        """
         @brief Subscribe to trades from the market. This is called when you subscribe to trade events from the market.
         @return A list of : class : ` MessageHandler `
        """
        # Send a trade detail message to the websocket
        if self.ws.sock and self.ws.sock.connected:
            self.ws.send('{"sub": "market.' + self.symbol + '.trade.detail"}')
        # If subscription_to_taker_trades is in market_listeners return true.
        if self.subscribe_to_taker_trades in self.market_listeners:
            return
        self.market_listeners.append(self.subscribe_to_taker_trades)
        self.message_handlers[f"market.{self.symbol}.trade.detail"] = self.on_taker_trade

    def on_message(self, ws, message):
        """
         @brief Called when a message is received from the websocket. This is the callback for websocket. py
         @param ws The websocket that received the message
         @param message The message that was received
         @return True if the message was handled False if it was
        """
        message = gzip.decompress(message).decode("utf-8")
        jmessage = json.loads(message)
        # ping callback if ping is set
        if 'ping' in jmessage.keys():
            self.on_ping_(jmessage['ping'])
            return
        try:
            self.message_handlers[jmessage['ch']](jmessage)
            self.update_timer()
        except:
            pass

    def on_quote(self, message):
        """
         @brief When a quote is received update the quotes dictionary. This is called by L { on_tick } and L { on_tick_response }
         @param message The message that triggered the event.
         @return None Side effects : Stores the quotes in a dictionary
        """
        # If tick is not set return nothing.
        if not message['tick']:
            return
        symbol = message['ch'].split(".")[1]
        current_quotes = self.quotes.get(symbol, {"ask": 0, "bid": 0})
        current_quotes['ask'] = message['tick']['ask'][0]
        current_quotes['bid'] = message['tick']['bid'][0]
        self.quotes[symbol] = current_quotes

    def on_taker_trade(self, message):
        """
         @brief When trading is done this method will be called. The message contains information about the trade that was done
         @param message Dictionary that contains the trade
        """
        trades = []
        # Trades are the trade data.
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
        """
         @brief Returns the current price of a symbol. This is an unfortunate way to get the current price of a symbol without having to re - ask the quote
         @param symbol Symbol to look up ( BTC - USD )
         @return Current price or None if there is no quote for
        """
        return self.quotes[symbol]['ask'] if len(self.quotes) > 0 else None

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

# This is the HuobiDMUSDTSwap exchange
class HuobiDMUSDTSwap(HuobiDMCoinSwap):
    instance = None
    symbol = 'BTC-USDT'
    url = "wss://api.hbdm.com/linear-swap-ws"
    name = "HuobiDMUSDTSwap"

    def __init__(self):
        """
         @brief Initialize HuobiDMUSDTSwap. Subclasses must implement __init__
        """
        super(HuobiDMUSDTSwap, self).__init__()

    @staticmethod
    def get_instance():
        """
         @brief Singleton to get HuobiDMUSDTSwap instance. This is needed because we need to create a class in __init__. py
         @return The instance of HuobiDMUSDTSwap
        """
        # Create a new instance of HuobiDMUSDTSwap.
        if HuobiDMUSDTSwap.instance is None:
            HuobiDMUSDTSwap.instance = HuobiDMUSDTSwap()
        return HuobiDMUSDTSwap.instance

    def current_price(self, symbol="BTC-USDT"):
        """
         @brief Get the current price of a symbol. This is an unfortunate way to get the current price without having to re - ask the quote
         @param symbol The symbol you want to get the current price for
         @return The current price of the symbol or None if there is
        """
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

# This is the HuobiDMCoinFuturesQuarterly exchange
class HuobiDMCoinFuturesQuarterly(HuobiDMCoinSwap):
    instance = None
    symbol = 'BTC_CQ'
    url = "wss://api.hbdm.com/ws"
    name = "HuobiDMFuturesQuarterly"

    def __init__(self):
        """
         @brief Initialize method. @ In None @ Out __init__ tuple ( int int
        """
        super(HuobiDMCoinFuturesQuarterly, self).__init__()

    @staticmethod
    def get_instance():
        """
         @brief Get HuobiDMCoinFuturesQuarterly instance. This is a singleton so we don't have to create a new instance every time we need it.
         @return The singleton instance of HuobiDMCoinFutures
        """
        # Singleton singleton instance of HuobiDMCoinFuturesQuarterly.
        if HuobiDMCoinFuturesQuarterly.instance is None:
            HuobiDMCoinFuturesQuarterly.instance = HuobiDMCoinFuturesQuarterly()
        return HuobiDMCoinFuturesQuarterly.instance

    def current_price(self, symbol="BTC_CQ"):
        """
         @brief Get the current price of a symbol. This is a function that takes a symbol and returns the current price of that symbol
         @param symbol Symbol to look up. Default BTC_CQ
         @return Current price of the symbol or None if not found
        """
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass

# This is the HuobiDMCoinFuturesWeekly exchange
class HuobiDMCoinFuturesWeekly(HuobiDMCoinFuturesQuarterly):
    instance = None
    symbol = 'BTC_CW'
    name = "HuobiDMFuturesQuarterly"

    def __init__(self):
        """
         @brief Initialize method. @ In None @ Out __init__ tuple ( bool str
        """
        super(HuobiDMCoinFuturesWeekly, self).__init__()

    @staticmethod
    def get_instance():
        """
         @brief Get the HuobiDMCoinFuturesWeekly instance. This is a singleton so we don't have to create a new instance every time we need it.
         @return The HuobiDMCoinFuturesWeekly instance
        """
        # Singleton method to use for instance of HuobiDMCoinFuturesWeekly.
        if HuobiDMCoinFuturesWeekly.instance is None:
            HuobiDMCoinFuturesWeekly.instance = HuobiDMCoinFuturesWeekly()
        return HuobiDMCoinFuturesWeekly.instance

    def current_price(self, symbol="BTC_CW"):
        """
         @brief Get the current price of a symbol. This is a function that takes a symbol and returns the current price of that symbol
         @param symbol Symbol to look up. Default BTC_CW ( Celsius )
         @return Current price of the symbol or None if not found
        """
        return self.quotes[symbol]['ask']

    def on_pong(self, ws):
        """
         @brief Called when we receive a PONG. This is the place where we'll be able to respond to the client's PONG message
         @param ws WebSocket that sent the
        """
        pass
