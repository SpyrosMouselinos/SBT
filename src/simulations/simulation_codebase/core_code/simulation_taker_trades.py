from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.queries.queries import Takers
import os
import time
import pandas as pd
import numpy as np


def get_taker_trades(t0, t1, swapMarket, swapSymbol):
    """
     @brief Get trades from Taker. This is a function to get trades from Taker. It takes two parameters t0 and t1 which are datetime objects and swapMarket and swapSymbol
     @param t0 datetime object of start date
     @param t1 datetime object of end date ( inclusive )
     @param swapMarket string of swap market ( BitMEX HuobiDMSwap Okex Deribit )
     @param swapSymbol string of swap symbol ( BTCUSD etc. )
     @return pandas. DataFrame with trades from Taker. Each row corresponds to a time stamp and trade type
    """
    influx_connection = InfluxConnection.getInstance()
    start = time.time()
    # This function will return a Takers object for the swap market.
    if swapMarket == 'BitMEX' or swapMarket == 'Binance' or swapMarket == 'HuobiDMSwap':
        swap_takers_querier = Takers(influx_connection.archival_client_spotswap_dataframe,
                                     [swapMarket], [swapSymbol])

    elif swapMarket == 'Okex' or swapMarket == "HuobiDM" or swapMarket == "Binance" or swapMarket == 'Deribit':
        swap_takers_querier = Takers(influx_connection.staging_client_spotswap_dataframe,
                                     [swapMarket], [swapSymbol])
    else:
        try:
            swap_takers_querier = Takers(influx_connection.archival_client_spotswap_dataframe,
                                         [swapMarket], [swapSymbol])
        except:
            swap_takers_querier = Takers(influx_connection.staging_client_spotswap_dataframe,
                                         [swapMarket], [swapSymbol])
    try:
        day_in_millis = 1000 * 60 * 60 * 24
        dfs = []
        # Load trades from influx.
        if t1 - t0 >= day_in_millis:
            t_start = t0
            t_end = t0 + day_in_millis

            # Load trades from influx.
            while t_end <= t1:
                # Set t_end to the end of the time range.
                if t1 - day_in_millis <= t_end <= t1:
                    t_end = t1

                base_dir = f"/home/equinoxai/data"
                # Returns the base directory where simulations_management data is stored.
                if not os.path.isdir(base_dir):
                    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../",
                                            "simulations_management", "data")
                local_dir_swap = f"{base_dir}/trades/{swapMarket}/{swapSymbol}/{swapMarket}_{swapSymbol}_{pd.to_datetime(t_start, unit='ms', utc=True).date()}.parquet.br"
                # Load taker trades from local file and load them into a DFS.
                if os.path.exists(local_dir_swap):
                    # print(f"Loading taker trades from local file {local_dir_swap}")
                    try:
                        df = pd.read_parquet(local_dir_swap, engine="pyarrow")
                    except:
                        df = pd.read_parquet(local_dir_swap)

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
                    df['timems'] = np.array((df['timestamp'].view(np.int64) + 1) / 1000000).astype(np.int64)
                    df = df.set_index("timestamp")
                    df['price'] = df['price'].astype(np.float64)
                    df['size'] = df['size'].astype(np.float64)
                    dfs.append(df)
                else:
                    print(f"Loading taker trades from influx. Couldn't find {local_dir_swap}")
                    df = swap_takers_querier.query_data(t_start, t_end).get_data(t_start, t_end)
                    dfs.append(df)
                    time.sleep(1)
                t_start = t_start + day_in_millis
                t_end = t_end + day_in_millis
        else:
            df = swap_takers_querier.query_data(t0, t1).get_data(t0, t1)
            dfs.append(df)
            time.sleep(1)
        # print(f"It took {time.time() - start}s to query the taker trades")
        return pd.concat(dfs)
    except KeyError:
        return pd.DataFrame(columns=['side'])
