import numpy as np
import pandas as pd
from influxdb import DataFrameClient
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

timestamp_column = 1
prediction_2_column = 1
ten_seconds = 1000 * 10
one_minute = 1000 * 60
five_minutes = one_minute * 5
one_hour = one_minute * 60
twelve_hours = one_hour * 12
three_minutes = one_minute * 3
one_day = one_hour * 24
one_second = 1000

WANDB_ENTITY = "zeger"
HYPERPARAMETERS_TUNING_PROJECT = "hyperparameters_tuning"
DATA_GENERATION_PROJECT = "data generation"
TAKER_MAKER_PROJECT = "taker_maker_simulations"
TAKER_MAKER_MULTICOIN_PROJECT = "taker_maker_multicoin_simulations"
TAKER_MAKER_PROJECT_2023 = "taker_maker_simulations_2023"
MAKER_MAKER_PROJECT = "maker_maker_simulations"
MAKER_MAKER_PERCENTAGE_PROJECT = "maker_maker_percentage_and_displacement"
TAKER_MAKER_AT_DEPTH_PROJECT = "taker_maker_at_depth"

client = DataFrameClient('influxdb.equinoxai.com',
                         443,
                         os.getenv("DB_USERNAME"),
                         os.getenv("DB_PASSWORD"),
                         'spotswap', ssl=True)

client1 = DataFrameClient('influxdb',
                          8086,
                          os.getenv("DB_USERNAME"),
                          os.getenv("DB_PASSWORD"),
                          'spotswap')


def set_latencies_auto(exchange_swap, exchange_spot):
    """
    @brief Determines optimal latencies for swap and spot exchanges based on location.

    This function calculates and returns the optimal websocket and API latencies for a given swap and spot exchange. 
    It utilizes a predefined latency structure for various exchanges across multiple geographical locations.

    @param exchange_swap The name of the swap exchange. Supported exchanges include:
    - 'Bybit'
    - 'Deribit'
    - 'BitMEX'
    - 'Okex'
    - 'HuobiDMSwap'
    - 'Binance'
    - 'Bitflyer'
    - 'FTX'
    - 'Bitstamp'
    - 'WOO'
    - 'DYDX'
    - 'LMAX'
    - 'Kraken'
    @param exchange_spot The name of the spot exchange. Supported exchanges are the same as for exchange_swap.

    @return A tuple containing:
    - ws_swap: The optimal websocket latency for the swap exchange.
    - api_swap: The optimal API latency for the swap exchange.
    - ws_spot: The optimal websocket latency for the spot exchange.
    - api_spot: The optimal API latency for the spot exchange.

    @note The function automatically adjusts the exchange name from 'HoubiDM' to 'HuobiDMSwap' for consistency.

    @throws KeyError If the specified exchange is not found in the predefined latency structures.
    @throws ValueError If any of the input parameters are invalid.

    @example
    @code
    ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto('Bybit', 'Binance')
    @endcode
    """

    if exchange_swap == 'HoubiDM':
        exchange_swap = 'HuobiDMSwap'
    if exchange_spot == 'HuobiDM':
        exchange_spot = 'HuobiDMSwap'

    latencies = {
        'Bybit': {'tokyo_ws': 42, 'tokyo_api': 94, 'hk_ws': 49, 'hk_api': 56, 'london_ws': 86, 'london_api': 245,
                  'frankfurt_ws': 84, 'frankfurt_api': 275, 'dublin_ws': 91, 'dublin_api': 193, 'singapore_ws': 2,
                  'singapore_api': 21, 'california_ws': 89, 'california_api': 189},
        'Deribit': {'tokyo_ws': 120, 'tokyo_api': 110, 'hk_ws': 90, 'hk_api': 100, 'london_ws': 6, 'london_api': 35,
                    'frankfurt_ws': 12, 'frankfurt_api': 40, 'dublin_ws': 9, 'dublin_api': 54, 'singapore_ws': 107,
                    'singapore_api': 107, 'california_ws': 75, 'california_api': 75},
        'BitMEX': {'tokyo_ws': 100, 'tokyo_api': 290, 'hk_ws': 125, 'hk_api': 230, 'london_ws': 13, 'london_api': 70,
                   'frankfurt_ws': 25, 'frankfurt_api': 150, 'dublin_ws': 8, 'dublin_api': 70, 'singapore_ws': 90,
                   'singapore_api': 249, 'california_ws': 73, 'california_api': 203},
        'Okex': {'tokyo_ws': 43, 'tokyo_api': 35, 'hk_ws': 8, 'hk_api': 15, 'london_ws': 110, 'london_api': 240,
                 'frankfurt_ws': 105, 'frankfurt_api': 260, 'dublin_ws': 125, 'dublin_api': 220, 'singapore_ws': 20,
                 'singapore_api': 69, 'california_ws': 83, 'california_api': 189},
        'HuobiDMSwap': {'tokyo_ws': 2, 'tokyo_api': 14, 'hk_ws': 27, 'hk_api': 65, 'london_ws': 107, 'london_api': 930,
                        'frankfurt_ws': 108, 'frankfurt_api': 930, 'dublin_ws': 100, 'dublin_api': 320,
                        'singapore_ws': 37, 'singapore_api': 110, 'california_ws': 55, 'california_api': 159},
        'Binance': {'tokyo_ws': 10, 'tokyo_api': 25, 'hk_ws': 30, 'hk_api': 70, 'london_ws': 115, 'london_api': 235,
                    'frankfurt_ws': 115, 'frankfurt_api': 310, 'dublin_ws': 105, 'dublin_api': 225, 'singapore_ws': 36,
                    'singapore_api': 88, 'california_ws': 57, 'california_api': 122},
        'Bitflyer': {'tokyo_ws': 70, 'tokyo_api': 200, 'hk_ws': 100, 'hk_api': 245, 'london_ws': 200, 'london_api': 425,
                     'frankfurt_ws': 200, 'frankfurt_api': 525, 'dublin_ws': 210, 'dublin_api': 505,
                     'singapore_ws': 101, 'singapore_api': 105, 'california_ws': 10 ** 6, 'california_api': 10 ** 6},
        'FTX': {'tokyo_ws': 15, 'tokyo_api': 45, 'hk_ws': 35, 'hk_api': 110, 'london_ws': 120, 'london_api': 270,
                'frankfurt_ws': 125, 'frankfurt_api': 320, 'dublin_ws': 140, 'dublin_api': 280, 'singapore_ws': 47,
                'singapore_api': 124, 'california_ws': 67, 'california_api': 150},
        'Bitstamp': {'tokyo_ws': 125, 'tokyo_api': 270, 'hk_ws': 105, 'hk_api': 270, 'london_ws': 15, 'london_api': 40,
                     'frankfurt_ws': 15, 'frankfurt_api': 80, 'dublin_ws': 15, 'dublin_api': 60, 'singapore_ws': 82,
                     'singapore_api': 194, 'california_ws': 10 ** 6, 'california_api': 10 ** 6},
        'WOO': {'tokyo_ws': 10, 'tokyo_api': 40, 'hk_ws': 32, 'hk_api': 90, 'london_ws': 130, 'london_api': 300,
                'frankfurt_ws': 130, 'frankfurt_api': 320, 'dublin_ws': 115, 'dublin_api': 255, 'singapore_ws': 41,
                'singapore_api': 97, 'california_ws': 10 ** 6, 'california_api': 10 ** 6},
        'DYDX': {'tokyo_ws': 110, 'tokyo_api': 210, 'hk_ws': 130, 'hk_api': 250, 'london_ws': 60, 'london_api': 135,
                 'frankfurt_ws': 70, 'frankfurt_api': 190, 'dublin_ws': 65, 'dublin_api': 120, 'singapore_ws': 142,
                 'singapore_api': 282, 'california_ws': 10 ** 6, 'california_api': 10 ** 6},
        'LMAX': {'tokyo_ws': 120, 'tokyo_api': 110, 'hk_ws': 90, 'hk_api': 100, 'london_ws': 5, 'london_api': 3,
                 'frankfurt_ws': 12, 'frankfurt_api': 40, 'dublin_ws': 9, 'dublin_api': 30, 'singapore_ws': 107,
                 'singapore_api': 107, 'california_ws': 10 ** 6, 'california_api': 10 ** 6},
        "Kraken": {'tokyo_ws': 78, 'tokyo_api': 194, 'hk_ws': 90, 'hk_api': 10 ** 6, 'london_ws': 89,
                   'london_api': 10 ** 6,
                   'frankfurt_ws': 104, 'frankfurt_api': 10 ** 6, 'dublin_ws': 95, 'dublin_api': 10 ** 6,
                   'singapore_ws': 106,
                   'singapore_api': 10 ** 6, 'california_ws': 23, 'california_api': 82}
    }

    latencies_df = pd.DataFrame(latencies)
    location_min_index = latencies_df[exchange_swap].idxmin().split('_')[0]
    ws_swap = latencies_df.loc[latencies_df[exchange_swap].index.str.contains(location_min_index), exchange_swap].iloc[
        0]
    api_swap = latencies_df.loc[latencies_df[exchange_swap].index.str.contains(location_min_index), exchange_swap].iloc[
        1]

    ws_spot = latencies_df.loc[latencies_df[exchange_spot].index.str.contains(location_min_index), exchange_spot].iloc[
        0]
    api_spot = latencies_df.loc[latencies_df[exchange_spot].index.str.contains(location_min_index), exchange_spot].iloc[
        1]

    return ws_swap, api_swap, ws_spot, api_spot


def exchange_fees(swap_exchange, swap_symbol, spot_exchange, spot_symbol):
    """
    @brief Retrieves the maker and taker fees for specific trading pairs on given exchanges.

    This function looks up and returns the maker and taker fees associated with specific trading pairs on the 
    specified swap and spot exchanges. It utilizes predefined fee structures for various exchanges and symbols.

    @param swap_exchange The name of the exchange for the swap trading pair. Supported exchanges include:
    - 'Binance'
    - 'Okex'
    - 'HuobiDMSwap'
    - 'HuobiDMOTC'
    - 'HuobiDM'
    - 'FTX'
    - 'FTX_OTC'
    - 'Deribit'
    - 'BitMEX'
    - 'Bitflyer'
    - 'Bitstamp'
    - 'Bybit'
    - 'DYDX'
    - 'KrakenFutures'
    - 'LMAX'
    - 'WOO'
    - 'AAX'
    - 'B2C2'
    - 'BTSE'
    - 'Kraken'
    @param swap_symbol The trading symbol for the swap exchange. Supported symbols are specific to each exchange,
    and may include pairs like 'binance_futures_btcusdt', 'okex_btc-usd-swap', 'ADA-USD', etc.
    @param spot_exchange The name of the exchange for the spot trading pair. This can be the same as swap_exchange or different.
    @param spot_symbol The trading symbol for the spot exchange. Supported symbols are specific to each exchange,
    and may include pairs like 'binance_spot_btcusdt', 'okex_eth-usdt', 'BTC-USD', etc.

    @return A tuple consisting of:
    - The maker fee for the specified swap trading pair.
    - The taker fee for the specified spot trading pair.

    @note Fees may be negative, indicating a rebate, or positive, indicating a cost.

    @throws KeyError If the specified exchange or symbol is not found in the predefined fee structures.
    @throws ValueError If any of the input parameters are invalid.

    @example
    @code
    double maker_fee, taker_fee;
    maker_fee, taker_fee = exchange_fees('Binance', 'binance_futures_btcusdt', 'Okex', 'okex_btc-usdt');
    @endcode
    """
    maker_fees = {
        'Binance': {'binance_futures_adausdt': -0.00002, 'binance_futures_avaxusdt': -0.00002,
                    'binance_futures_bnbusd': -0.00014, 'binance_futures_btcbusd': -0.00014,
                    'binance_futures_btcusdt': -0.00002, 'binance_futures_ethbusd': -0.00014,
                    'binance_futures_ethusdt': -0.00002, 'binance_futures_solusdt': -0.00002,
                    'binance_spot_btcbusd': 0.00015, 'binance_spot_btcusdt': 0.00015, 'binance_spot_ethusdt': 0.00015,
                    'binance_spot_ethbtc': 0.00015, 'binance_spot_ethbusd': 0.00015, 'binance_spot_linkusdt': 0.00015,
                    'binance_spot_ltcusdt': 0.00015, 'binance_spot_maticbtc': 0.00015, 'binance_spot_solbtc': 0.00015,
                    'binance_swap_adausd_perp': -0.00006, 'binance_swap_btcusd_perp': -0.00006,
                    'binance_swap_ethusd_perp': -0.00006, 'binance_swap_linkusd_perp': -0.00006,
                    'binance_swap_solusd': -0.00006, 'btcusd_perp': np.nan, 'btcusdt': np.nan, 'ethusd_perp': np.nan,
                    'ethsudt': np.nan, 'linkusd_perp': np.nan, 'linkusdt': np.nan, 'ltcusdt': np.nan},
        'Okex': {'okex_ada-usd-swap': -0.00005, 'okex_btc-usd-swap': -0.00005, 'okex_btc-usdt': -0.00002,
                 'okex_btc-usdt-swap': -0.0002, 'okex_btc_bi_quarter': -0.00005, 'okex_btc_next_quarter': -0.00005,
                 'okex_btc_next_week': -0.00005, 'okex_btc_quarter': -0.00005, 'okex_btc_this_week': -0.00005,
                 'okex_dot-usd-swap': -0.00005, 'okex_eos-usd-swap': -0.00005, 'okex_eth-usd-swap': -0.00005,
                 'okex_eth-usdt': -0.00002, 'okex_eth-usdt-swap': -0.0002, 'okex_eth_bi_quarter': -0.00005,
                 'okex_eth_next_quarter': -0.00005, 'okex_eth_this_quarter': -0.00005, 'okex_eth_next_week': -0.00005,
                 'okex_eth_quarter': -0.00005, 'okex_eth_this_week': -0.00005, 'okex_link-usd-swap': -0.00005,
                 'okex_ltc-usd-swap': -0.00005, 'okex_sol-usd-swap': -0.00005, 'okex_sol-usdt-swap': -0.0002,
                 'okex_xrp-usd-swap': -0.00005, 'time_adjusted_okex_btc_this_week': -0.00005},
        'HuobiDMSwap': {'ADA-USD': -0.00015, 'BTC-USD': -0.00015, 'BTC-USDT': -0.00015, 'DOGE-USD': -0.00015,
                        'DOT-USD': -0.00015, 'EOS-USD': -0.00015, 'ETH-USD': -0.00015, 'ETH-USDT': -0.00015,
                        'LINK-USD': -0.00015, 'LTC-USD': -0.00015, 'XRP-USD': -0.00015},
        'HuobiDMOTC': {'BTCHUSD': np.nan},
        'HuobiDM': {'BCH_CQ': np.nan, 'BTC_CQ': np.nan, 'BTC_CW': np.nan, 'BTC_NW': np.nan, 'ETH_CQ': np.nan,
                    'ETH_CW': np.nan, 'ETH_NW': np.nan, 'LTC_CQ': np.nan, 'XRP_CQ': np.nan, 'btchusd': np.nan,
                    'btcusdt': np.nan, 'ethhusd': np.nan, 'ethusdt': np.nan},
        'FTX': {'ADA-0930': -0.00002, 'ADA-PERP': -0.00002, 'AVAX-0930': -0.00002, 'AVAX-PERP': -0.00002,
                'AVAX/USD': -0.00002,
                'BNB-0930': -0.00002, 'BNB-PERP': -0.00002, 'BNB/USD': -0.00002, 'BTC-0930': -0.00002,
                'BTC-1230': -0.00002,
                'BTC-PERP': -0.00002, 'BTC/USD': -0.00002, 'DOT-0930': -0.00002, 'DOT-PERP': -0.00002,
                'DOT/USD': -0.00002,
                'ETH-0930': -0.00002, 'ETH-1230': -0.00002, 'ETH-PERP': -0.00002, 'ETH/USD': -0.00002,
                'ETH/BTC': -0.00002,
                'LTC-PERP': -0.00002, 'SOL-0930': -0.00002, 'SOL-PERP': -0.00002, 'SOL/USD': -0.00002,
                'XRP-PERP': -0.00002},
        'FTX_OTC': {'FTX_OTC_BTC-USD': np.nan},
        'Deribit': {'BTC-PERPETUAL': -0.0001, 'ETH-PERPETUAL': -0.0001, 'SOL-PERPETUAL': -0.0002},
        'BitMEX': {'DOGEUSD': -0.00011, 'DOGEUSDT': -0.00011, 'ETHUSD': -0.00011, 'ETHUSDT': -0.00011,
                   'ETHUSDU20': -0.00011, 'LINKUSD': -0.00011,
                   'LINKUSDT': -0.00011, 'LTCUSD': -0.00011, 'LTCUSDT': -0.00011, 'XBTUSD': -0.00011,
                   'XBTUSDT': -0.00011, 'XRPUSD': -0.00011,
                   'backfilled_XBTUSD': -0.00011},
        'Bitflyer': {'BTC_JPY': 0, 'BTC_JPY_IN_USD': np.nan, 'BTC_USD': np.nan, 'ETH_JPY': np.nan,
                     'FX_BTC_JPY': np.nan, 'FX_BTC_USD': np.nan, 'FX_TO_SPOT': np.nan, 'JPY_USD': np.nan},
        'Bitstamp': {'btcusd': np.nan, 'ethusd': np.nan},
        'Bybit': {'bybit_ADAUSDT': 0.0, 'bybit_AVAXUSDT': 0.0, 'bybit_BTCUSD': 0.0, 'bybit_BTCUSDT': 0.0,
                  'bybit_EOSUSD': 0.0, 'bybit_ETHUSD': 0.0, 'bybit_ETHUSDT': 0.0, 'bybit_SOLUSD': 0.0,
                  'bybit_SOLUSDT': 0.0, 'bybit_XRPUSD': 0.0, 'bybit_spot_BTCUSDT': 0.0,
                  'bybit_spot_EOSUSDT': 0.0, 'bybit_spot_ETUSDT': 0.0},
        'DYDX': {'BTC-USD': 0, 'ETH-USD': 0},
        'KrakenFutures': {'PI-ETUHSD': np.nan, 'PI_XBTUSD': np.nan},
        'LMAX': {'BTCUSD': np.nan, 'ETHUSD': np.nan},
        'WOO': {'PERP_BTC_USDT': 0.0001, 'PERP_ETH_USDT': 0.0001, 'PERP_SOL_USDT': 0.0001, 'PERP_XPR_USDT': 0.0001,
                'SPOT_BTC_USDT': 0.0001, 'SPOT_ETH_USDT': 0.0001, 'SPOT_SOL_USDT': 0.0001, 'SPOT_XRP_USDT': 0.0001},
        'AAX': {'BTCUSDFP': np.nan, 'BTCUSDTFP': np.nan, 'ETHUSDFP': np.nan},
        'B2C2': {'BTCEUR.SPOT': np.nan, 'BTCUSD.SPOT': np.nan, 'ETHUSD.SPOT': np.nan},
        'BTSE': {'btse_BTCPFC': np.nan},
        'LMAX': {'BTCUSD': 0.0, 'ETHUSD': 0.0},
        'Kraken': {'ETH/USD': np.nan, 'ETH/EUR': np.nan, 'EUR/USD': np.nan, 'SOL/USD': np.nan, 'XBT/EUR': np.nan,
                   'XBT/USD': np.nan, 'ETH/EUR_IN_USD': 0.0}
    }

    taker_fees = {
        'Binance': {'binance_futures_adausdt': 0.000225, 'binance_futures_avaxusdt': 0.000225,
                    'binance_futures_bnbusd': 0.000207, 'binance_futures_btcbusd': 0.000207,
                    'binance_futures_btcusdt': 0.000225, 'binance_futures_ethbusd': 0.000207,
                    'binance_futures_ethusdt': 0.000225, 'binance_futures_solusdt': 0.000225,
                    'binance_spot_btcbusd': 0.0003, 'binance_spot_btcusdt': 0.0003, 'binance_spot_ethusdt': 0.0003,
                    'binance_spot_ethbtc': 0.0003, 'binance_spot_ethbusd': 0.0003, 'binance_spot_linkusdt': 0.0003,
                    'binance_spot_ltcusdt': 0.0003, 'binance_spot_maticbtc': 0.0003, 'binance_spot_solbtc': 0.0003,
                    'binance_swap_adausd_perp': 0.00024, 'binance_swap_btcusd_perp': 0.00024,
                    'binance_swap_ethusd_perp': 0.00024, 'binance_swap_linkusd_perp': 0.00024,
                    'binance_swap_solusd': 0.00024, 'btcusd_perp': 0.00024, 'btcusdt': 0.00024, 'ethusd_perp': 0.00024,
                    'ethsudt': np.nan, 'linkusd_perp': np.nan, 'linkusdt': np.nan, 'ltcusdt': np.nan},
        'Okex': {'okex_ada-usd-swap': 0.00025, 'okex_btc-usd-swap': 0.00025, 'okex_btc-usdt': 0.0003,
                 'okex_btc-usdt-swap': 0.0002, 'okex_btc_bi_quarter': 0.00025, 'okex_btc_next_quarter': 0.00025,
                 'okex_btc_next_week': 0.00025, 'okex_btc_quarter': 0.00025, 'okex_btc_this_week': 0.00025,
                 'okex_dot-usd-swap': 0.00025, 'okex_eos-usd-swap': 0.00025, 'okex_eth-usd-swap': 0.00025,
                 'okex_eth-usdt': 0.0003, 'okex_eth-usdt-swap': 0.0002, 'okex_eth_bi_quarter': 0.00025,
                 'okex_eth_next_quarter': 0.00025, 'okex_eth_this_quarter': 0.00025, 'okex_eth_next_week': 0.00025,
                 'okex_eth_quarter': 0.00025, 'okex_eth_this_week': 0.00025, 'okex_link-usd-swap': 0.00025,
                 'okex_ltc-usd-swap': 0.00025, 'okex_sol-usd-swap': 0.00025, 'okex_sol-usdt-swap': 0.0002,
                 'okex_xrp-usd-swap': 0.00025, 'time_adjusted_okex_btc_this_week': np.nan},
        'HuobiDMSwap': {'ADA-USD': 0.0003, 'BTC-USD': 0.0003, 'BTC-USDT': 0.000275, 'DOGE-USD': 0.0003,
                        'DOT-USD': 0.0003, 'EOS-USD': 0.0003, 'ETH-USD': 0.0003, 'ETH-USDT': 0.000275,
                        'LINK-USD': 0.0003, 'LTC-USD': 0.0003, 'XRP-USD': 0.0003},
        'HuobiDMOTC': {'BTCHUSD': np.nan},
        'HuobiDM': {'BCH_CQ': np.nan, 'BTC_CQ': np.nan, 'BTC_CW': np.nan, 'BTC_NW': np.nan, 'ETH_CQ': np.nan,
                    'ETH_CW': np.nan, 'ETH_NW': np.nan, 'LTC_CQ': np.nan, 'XRP_CQ': np.nan, 'btchusd': np.nan,
                    'btcusdt': 0.00046, 'ethhusd': np.nan, 'ethusdt': np.nan},
        'FTX': {'ADA-0930': 0.00026, 'ADA-PERP': 0.00026, 'AVAX-0930': 0.00026, 'AVAX-PERP': 0.00026,
                'AVAX/USD': 0.00026,
                'BNB-0930': 0.00026, 'BNB-PERP': 0.00026, 'BNB/USD': 0.00026, 'BTC-0930': 0.00026, 'BTC-1230': 0.00026,
                'BTC-PERP': 0.00026, 'BTC/USD': 0.00026, 'DOT-0930': 0.00026, 'DOT-PERP': 0.00026, 'DOT/USD': 0.00026,
                'ETH-0930': 0.00026, 'ETH-1230': 0.00026, 'ETH-PERP': 0.00026, 'ETH/USD': 0.00026, 'ETH/BTC': 0.00026,
                'LTC-PERP': 0.00026, 'SOL-0930': 0.00026, 'SOL-PERP': 0.00026, 'SOL/USD': 0.00026, 'XRP-PERP': 0.00026},
        'FTX_OTC': {'FTX_OTC_BTC-USD': np.nan},
        'Deribit': {'BTC-PERPETUAL': 0.0003, 'ETH-PERPETUAL': 0.0003, 'SOL-PERPETUAL': 0.0003},
        'BitMEX': {'DOGEUSD': 0.000228, 'DOGEUSDT': 0.000228, 'ETHUSD': 0.000228, 'ETHUSDT': 0.000228,
                   'ETHUSDU20': 0.000228, 'LINKUSD': 0.000228,
                   'LINKUSDT': 0.000228, 'LTCUSD': 0.000228, 'LTCUSDT': 0.000228, 'XBTUSD': 0.000228,
                   'XBTUSDT': 0.000228, 'XRPUSD': 0.000228,
                   'backfilled_XBTUSD': 0.00024},
        'Bitflyer': {'BTC_JPY': 0.0001, 'BTC_JPY_IN_USD': np.nan, 'BTC_USD': np.nan, 'ETH_JPY': np.nan,
                     'FX_BTC_JPY': np.nan, 'FX_BTC_USD': np.nan, 'FX_TO_SPOT': np.nan, 'JPY_USD': np.nan},
        'Bitstamp': {'btcusd': 0.0003, 'ethusd': 0.0003},
        'Bybit': {'bybit_ADAUSDT': 0.0003, 'bybit_AVAXUSDT': 0.0003, 'bybit_BTCUSD': 0.0003, 'bybit_BTCUSDT': 0.0003,
                  'bybit_EOSUSD': 0.0003, 'bybit_ETHUSD': 0.0003, 'bybit_ETHUSDT': 0.0003, 'bybit_SOLUSD': 0.0003,
                  'bybit_SOLUSDT': 0.0003, 'bybit_XRPUSD': 0.0003, 'bybit_spot_BTCUSDT': 0.0002,
                  'bybit_spot_EOSUSDT': 0.0002, 'bybit_spot_ETUSDT': 0.0002},
        'DYDX': {'BTC-USD': 0.00027, 'ETH-USD': 0.00027},
        'KrakenFutures': {'PI-ETUHSD': np.nan, 'PI_XBTUSD': np.nan},
        'LMAX': {'BTCUSD': np.nan, 'ETHUSD': np.nan},
        'WOO': {'PERP_BTC_USDT': 0.0001, 'PERP_ETH_USDT': 0.0001, 'PERP_SOL_USDT': 0.0001, 'PERP_XPR_USDT': 0.0001,
                'SPOT_BTC_USDT': 0.0001, 'SPOT_ETH_USDT': 0.0001, 'SPOT_SOL_USDT': 0.0001, 'SPOT_XRP_USDT': 0.0001},
        'AAX': {'BTCUSDFP': np.nan, 'BTCUSDTFP': np.nan, 'ETHUSDFP': np.nan},
        'B2C2': {'BTCEUR.SPOT': np.nan, 'BTCUSD.SPOT': np.nan, 'ETHUSD.SPOT': np.nan},
        'BTSE': {'btse_BTCPFC': np.nan},
        'LMAX': {'BTCUSD': 0.0003, 'ETHUSD': 0.0003},
        'Kraken': {'ETH/USD': np.nan, 'ETH/EUR': np.nan, 'EUR/USD': np.nan, 'SOL/USD': np.nan, 'XBT/EUR': np.nan,
                   'XBT/USD': np.nan, 'ETH/EUR_IN_USD': np.nan}
    }

    return maker_fees[swap_exchange][swap_symbol], taker_fees[spot_exchange][spot_symbol]


def exchange_tick_size(swap_exchange, swap_symbol):
    """
     @brief Returns the tick size for a swap exchange and swap symbol. This is based on the number of bits per second and in the case of BitMEX the size is used to determine the length of the tick.
     @param swap_exchange The exchange to use. Can be one of BTC ETH PERPETUAL SOFTWARE or BITMEX.
     @param swap_symbol The swap symbol to use. Can be one of DRAFT_CHAN CURSOR BTC_SYM or SOL_CH
    """
    tick_sizes = {
        'Deribit': {'BTC-PERPETUAL': 0.5, 'ETH-PERPETUAL': 0.05, 'SOL-PERPETUAL': 0.01},
        'BitMEX': {'DOGEUSD': 0.00001, 'DOGEUSDT': 0.00001, 'ETHUSD': 0.05, 'ETHUSDU20': 0.05, 'LINKUSD': 0.001,
                   'LINKUSDT': 0.001, 'LTCUSD': 0.01, 'LTCUSDT': 0.01, 'XBTUSD': 0.5, 'XRPUSD': 0.0001,
                   'backfilled_XBTUSD': 0.5},
    }
    return tick_sizes[swap_exchange][swap_symbol]
