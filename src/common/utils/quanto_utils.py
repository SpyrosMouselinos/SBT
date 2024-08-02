import numpy as np
import pandas as pd
from src.common.queries.queries import get_price
from src.common.connections.DatabaseConnections import InfluxConnection
from dotenv import load_dotenv, find_dotenv
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


def bitmex_btc_prices(t0, t1, environment, split_data=True):
    #  download the bitmex xbtusd prices
    bitmex_btc_ask = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='XBTUSD', side='Ask',
                               environment=environment, split_data=split_data)
    bitmex_btc_ask['Time'] = bitmex_btc_ask.index
    bitmex_btc_bid = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='XBTUSD', side='Bid',
                               environment=environment, split_data=split_data)
    bitmex_btc_bid['Time'] = bitmex_btc_bid.index

    bitmex_btc_price = pd.merge_ordered(bitmex_btc_ask, bitmex_btc_bid, on='Time', suffixes=['_ask', '_bid'])
    # drop the lines where the ask price is lower than the bid price
    bitmex_btc_price.reset_index(drop=True, inplace=True)
    bitmex_btc_price['price'] = bitmex_btc_price[['price_ask', 'price_bid']].mean(axis=1)
    return bitmex_btc_price


def bitmex_btc_ask_prices(t0, t1, environment):
    bitmex_btc_ask = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='XBTUSD', side='Ask',
                               environment=environment)
    bitmex_btc_ask['Time'] = bitmex_btc_ask.index
    return bitmex_btc_ask


def bitmex_eth_prices(t0, t1, environment, split_data=True):
    # download the bitmex ethusd prices
    bitmex_eth_ask = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='ETHUSD', side='Ask',
                               environment=environment, split_data=split_data)
    bitmex_eth_ask['Time'] = bitmex_eth_ask.index
    bitmex_eth_bid = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='ETHUSD', side='Bid',
                               environment=environment, split_data=split_data)
    bitmex_eth_bid['Time'] = bitmex_eth_bid.index

    bitmex_eth_price = pd.merge_ordered(bitmex_eth_ask, bitmex_eth_bid, on='Time', suffixes=['_ask', '_bid'])
    bitmex_eth_price.reset_index(drop=True, inplace=True)
    bitmex_eth_price['price'] = bitmex_eth_price[['price_ask', 'price_bid']].mean(axis=1)
    return bitmex_eth_price


def bitmex_eth_ask_prices(t0, t1, environment):
    # download the bitmex ethusd prices
    bitmex_eth_ask = get_price(t_start=t0, t_end=t1, exchange='BitMEX', symbol='ETHUSD', side='Ask',
                               environment=environment)
    bitmex_eth_ask['Time'] = bitmex_eth_ask.index
    return bitmex_eth_ask


def track_average_price(cum_volume, sum_vol_price, volume, price):
    # find the weighted average of price
    if cum_volume != 0:
        sum_vol_price = sum_vol_price + volume * price
        w_avg_price = sum_vol_price / cum_volume
        return w_avg_price
    else:
        return 0


def coin_volume_func(cum_volume, weighted_average):
    # find the volume of the underlying coin
    return cum_volume / weighted_average


def quanto_pnl_func(price_eth, avg_price_eth, price_btc, avg_price_btc, coin_volume):
    # compute the quanto loss
    return (price_eth - avg_price_eth) * (price_btc - avg_price_btc) * 0.000001 * coin_volume


def quanto_pnl_func_exp(price_eth, avg_price_eth, price_btc, avg_price_btc, coin_volume, exp1, exp2):
    # compute the quanto loss
    return pow(abs(price_eth - avg_price_eth), exp1) * pow(abs(price_btc - avg_price_btc), exp2) * 0.000001 * \
        coin_volume * np.sign(price_btc - avg_price_btc) * np.sign(price_eth - avg_price_eth)


def recompute_band_value(band_value, quanto_loss):
    # recompute the exit band value
    if quanto_loss < 0:
        return band_value + abs(quanto_loss)
    else:
        return band_value


def get_boxes(df, bp):
    center_price = df.price.iloc[0]
    upper_limit = center_price + bp / 10000 * center_price
    lower_limit = center_price - bp / 10000 * center_price
    start_time_box = df.index[0]
    end_time_box = None
    volumes_per_box = []
    ask_volume_per_box = []
    bid_volume_per_box = []
    number_of_trades_per_box = []
    cum_ask_volume_box = 0
    cum_bid_volume_box = 0
    number_trades_ask = 0
    number_trades_bid = 0
    for j in range(1, len(df)):
        price = df.price.iloc[j]
        if price > upper_limit or price < lower_limit:
            end_time_box = df.index[j - 1]
            current_box = [start_time_box, end_time_box, lower_limit, upper_limit, center_price,
                           cum_ask_volume_box + cum_bid_volume_box, cum_ask_volume_box, cum_bid_volume_box,
                           number_trades_ask, number_trades_bid]
            start_time_box = df.index[j]
            if abs(price - center_price) > 100 * bp / 10000 * center_price:
                number_of_boxes = ((price - center_price) / center_price - 4) % 2
                start_timestamp = int(start_time_box.timestamp() * 1000) - 10
                end_timestamp = int(end_time_box.timestamp() * 1000) + 10
                increment = (end_timestamp - start_timestamp) / number_of_boxes
                if pd.isna(number_of_boxes):
                    number_of_boxes = 0
                for k in range(int(number_of_boxes)):
                    volumes_per_box.append([end_time_box, end_time_box, lower_limit, upper_limit,
                                            center_price, 0, 0, 0, 0, 0])
            volumes_per_box.append(current_box)
            cum_ask_volume_box = 0
            cum_bid_volume_box = 0
            number_trades_ask = 0
            number_trades_bid = 0
            center_price = df.price.iloc[j]
            upper_limit = center_price + bp / 10000 * center_price
            lower_limit = center_price - bp / 10000 * center_price
        if df.side.iloc[j] == "Bid":
            cum_bid_volume_box += df["size"].iloc[j]
            number_trades_ask += 1
        if df.side.iloc[j] == "Ask":
            cum_ask_volume_box += df["size"].iloc[j]
            number_trades_bid += 1
    column_names = ["start_time", "end_time", "lower_limit", "upper_limit", "center_price", "volume_box",
                    "ask_volume_box", "bid_volume_box", "ask_number_trades", "bid_number_trades"]
    return pd.DataFrame(data=volumes_per_box, columns=column_names)


def normalise(df, window=1):
    normalized_df = df.copy(deep=True)
    for j in range(len(df)):
        start_date = df.iloc[j].end_time - timedelta(hours=window)
        end_date = df.iloc[j].end_time
        avg_box_count = df[(df.end_time >= start_date) & (df.end_time <= end_date)].counts.median()
        std_box_count = df[(df.end_time >= start_date) & (df.end_time <= end_date)].counts.std()
        col_filter = [x for x in df.columns if "ema_counts" in x]
        for col in col_filter:
            normalized_df[col].iloc[j] = (df[col].iloc[j] - avg_box_count)
    return normalized_df


def get_price_box_signal(t0, t1, basis_point, aggr_window, span):
    influx = InfluxConnection.getInstance()
    influx_staging = influx.archival_client_spotswap_dataframe
    trades = influx_staging.query(f"SELECT price, size, side FROM trade WHERE "
                                  f"(exchange = 'BitMEX' AND symbol = 'ETHUSD') AND time >= {t0}ms "
                                  f"and time <= {t1}ms")['trade']
    trades = trades[trades.price != 0]
    boxes_df = get_boxes(trades, basis_point)
    boxes_df["box_durations"] = boxes_df.end_time.apply(lambda x: int(x.timestamp())) - boxes_df.start_time.apply(
        lambda x: int(x.timestamp()))
    boxes_df_aggregated = boxes_df
    boxes_df_aggregated = boxes_df_aggregated.set_index('end_time'). \
        groupby(pd.Grouper(freq=aggr_window)).size().reset_index(name='counts')
    boxes_df_aggregated[f"signal"] = boxes_df_aggregated["counts"].ewm(span=span).mean()
    return boxes_df_aggregated
