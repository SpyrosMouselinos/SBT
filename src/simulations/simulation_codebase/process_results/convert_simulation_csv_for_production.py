import pandas as pd
import datetime
import numpy as np


def convert_simulation_csv_to_production_csv(simulation_csv_path: str = None,
                                             production_csv_path: str = None,
                                             strategies_list: list = None,
                                             local_path: str = None,
                                             num_of_params: int = None,
                                             has_generic_funding_parameters: bool = False):
    df = pd.read_csv(simulation_csv_path)
    # create a dictionary to place generic funding parameters
    index_len = len(strategies_list)

    renamed_columns, original_columns = production_csv_column_list(production_csv_path)
    prod_df = pd.DataFrame(columns=renamed_columns)

    if "bitmex_perp_deribit_perp_btc" in strategies_list[0]:
        strategy_group = "xbtusd_deribit"
    elif "XBTUSD_maker_perpetual" in strategies_list[0]:
        strategy_group = "xbtusd_bitmex"
    elif "ETH_perpetual_bitmex_ETHUSD" in strategies_list[0]:
        strategy_group = "ethusd_bitmex"
    elif "bitmex_perp_deribit_perp_eth" in strategies_list[0]:
        strategy_group = "ethusd_deribit"
    else:
        return

    if 'deribit' in strategy_group and has_generic_funding_parameters:
        swap_funding_weights, spot_funding_weights = create_generic_funding_parameters(df, num_of_params, "deribit")
    elif 'bitmex' in strategy_group and has_generic_funding_parameters:
        spot_funding_weights, swap_funding_weights = create_generic_funding_parameters(df, num_of_params, "bitmex")
    else:
        spot_funding_weights = pd.Series([np.nan] * index_len)
        swap_funding_weights = pd.Series([np.nan] * index_len)

    if "slow_funding_window" not in df.columns:
        df["slow_funding_window"] = np.nan

    if "funding_periods_lookback" not in df.columns:
        df["funding_periods_lookback"] = np.nan

    if "strategy" in df.columns and 'generic' not in df["strategy"].iloc[0]:
        prod_df["strategy"] = df["strategy"]
        index_len = len(df)
    else:
        prod_df["strategy"] = strategies_list

    if "Name" in df.columns:
        prod_df["batch_id"] = df["Name"].iloc[:index_len]

    if df.columns.str.contains("ROR Annualized").any():
        ror_annualized = "ROR Annualized"
    elif df.columns.str.contains("ROR", case=False).any():
        ror_annualized = \
            df.loc[:,
            df.columns.str.contains("ROR", case=False) & ~ df.columns.str.contains("STD", case=False)].columns[-1]
    else:
        ror_annualized = None

    if ror_annualized in df.columns:
        prod_df["simulated_ROR"] = df[ror_annualized].iloc[:index_len].round(2)
    else:
        prod_df["simulated_ROR"] = [0.0] * index_len

    prod_df["blended"] = ["false"] * index_len
    if 'max_trade_volume' in df.columns:
        prod_df["max_trade_volume_entry"] = df["max_trade_volume"].iloc[:index_len]
        prod_df["max_trade_volume_exit"] = df["max_trade_volume"].iloc[:index_len]
    else:
        prod_df["max_trade_volume_entry"] = [3000] * index_len
        prod_df["max_trade_volume_exit"] = [3000] * index_len

    prod_df["funding_system"] = df["band_funding_system"].iloc[:index_len]
    if "band_funding_system2" in df.columns:
        prod_df["funding_system_close"] = df["band_funding_system2"].iloc[:index_len]

    prod_df["window_size"] = df["window_size"].iloc[:index_len]
    prod_df["funding_window"] = df["funding_window"].iloc[:index_len]
    prod_df["funding_periods_lookback"] = df["funding_periods_lookback"].iloc[:index_len]
    prod_df["funding_periods_lookback.1"] = df["funding_periods_lookback"].iloc[:index_len]
    prod_df["entry_delta_spread"] = df["entry_delta_spread"].iloc[:index_len].round(4)
    prod_df["exit_delta_spread"] = df["exit_delta_spread"].iloc[:index_len].round(4)
    prod_df["use_max_current_next_funding"] = ["true"] * index_len

    prod_df["slow_funding_window"] = df["slow_funding_window"].iloc[:index_len]
    prod_df["spot_funding_weights_exit"] = spot_funding_weights.iloc[:index_len]
    prod_df["swap_funding_weights_exit"] = swap_funding_weights.iloc[:index_len]
    prod_df["spot_funding_weights_entry"] = spot_funding_weights.iloc[:index_len]
    prod_df["swap_funding_weights_entry"] = swap_funding_weights.iloc[:index_len]

    if strategy_group in ["xbtusd_deribit", "xbtusd_bitmex"]:
        if "window_size2" not in df.columns:
            df["window_size2"] = df['window_size']
        prod_df["window_size.1"] = df["window_size2"].iloc[:index_len]
        prod_df["funding_window.1"] = df["funding_window"].iloc[:index_len]
        prod_df["entry_delta_spread.1"] = df["entry_delta_spread2"].iloc[:index_len].round(4)
        prod_df["exit_delta_spread.1"] = df["exit_delta_spread2"].iloc[:index_len].round(4)
        prod_df["use_max_current_next_funding.1"] = ["true"] * index_len
        prod_df["slow_funding_window.1"] = df["slow_funding_window"].iloc[:index_len]
        prod_df["spot_funding_weights_exit.1"] = spot_funding_weights.iloc[:index_len]
        prod_df["swap_funding_weights_exit.1"] = swap_funding_weights.iloc[:index_len]
        prod_df["spot_funding_weights_entry.1"] = spot_funding_weights.iloc[:index_len]
        prod_df["swap_funding_weights_entry.1"] = swap_funding_weights.iloc[:index_len]
    elif strategy_group in ["ethusd_bitmex", "ethusd_deribit"]:
        prod_df["current_ratio"] = df["current_r"].iloc[:index_len].round(4)
        prod_df["high_volatility_ratio"] = df["high_r"].iloc[:index_len].round(4)
        prod_df["minimum_between_bands.1"] = [0.4] * index_len
        prod_df["rolling_time_window_size.1"] = df["rolling_time_window_size"].iloc[:index_len]
        prod_df["high_volatility_threshold"] = - df["quanto_threshold"].iloc[:index_len].round(4)
        prod_df["rolling_time_window_size_ratio"] = df["ratio_entry_band_mov_ind"].iloc[:index_len].round(4)
        prod_df["rolling_time_window_size_enabled"] = ["true"] * index_len
        prod_df["hours_to_stop_going_short_after_hr"] = df["hours_to_stop"].iloc[:index_len]
    else:
        pass

    prod_df.rename(columns={renamed_columns[x]: original_columns[x] for x in range(len(original_columns))},
                   inplace=True)
    year_month_day = datetime.datetime.now().strftime("%Y_%m_%d")
    prod_df.to_csv(f"{local_path}{strategy_group}_as_maker_{year_month_day}.csv", index=False, quotechar='"')


def create_generic_funding_parameters(df: pd.DataFrame, num_of_params: int = 3, swap_exchange: str = "deribit"):
    if swap_exchange == "deribit":
        suffix1 = "Swap"
        suffix2 = "Spot"
    elif swap_exchange == "bitmex":
        suffix1 = "Spot"
        suffix2 = "Swap"

    if not any(df.columns.str.contains(f"hoursBefore{suffix1}")):
        for ix in range(num_of_params):
            df[f"hoursBefore{suffix1}{ix}"] = df[f"hoursBefore{suffix2}{ix}"]

    if pd.isna(df[f'slowWeight{suffix1}0'].iloc[0]):
        for ix in range(num_of_params):
            df[f"slowWeight{suffix1}{ix}"] = df[f"slowWeight{suffix2}{ix}"]
    if pd.isna(df[f'slowWeight{suffix2}0'].iloc[0]):
        for ix in range(num_of_params):
            df[f"slowWeight{suffix2}{ix}"] = df[f"slowWeight{suffix1}{ix}"]

    funding_weights1 = pd.Series([[{"fastWeight": df[f"fastWeight{suffix1}{ix}"].iloc[x].round(4),
                                    "slowWeight": df[f"slowWeight{suffix1}{ix}"].iloc[x].round(4),
                                    "hoursBefore": df[f"hoursBefore{suffix1}{ix}"].iloc[x].round(4)}
                                   for ix in range(num_of_params)] for x in df.index])

    if any(df.columns.str.contains(f"fastWeight{suffix2}")) and any(
            df.columns.str.contains(f"slowWeight{suffix2}")) and any(
        df.columns.str.contains(f"hoursBefore{suffix2}")):
        funding_weights2 = pd.Series([[{"fastWeight": df[f"fastWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "slowWeight": df[f"slowWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "hoursBefore": df[f"hoursBefore{suffix2}{ix}"].iloc[x].round(4)}
                                       for ix in range(num_of_params)] for x in df.index])
    elif any(df.columns.str.contains(f"fastWeight{suffix2}")) and any(df.columns.str.contains(f"slowWeight{suffix2}")):
        funding_weights2 = pd.Series([[{"fastWeight": df[f"fastWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "slowWeight": df[f"slowWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "hoursBefore": df[f"hoursBefore{suffix1}{ix}"].iloc[x].round(4)}
                                       for ix in range(num_of_params)] for x in df.index])
    elif any(df.columns.str.contains(f"fastWeight{suffix2}")):
        funding_weights2 = pd.Series([[{"fastWeight": df[f"fastWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "slowWeight": df[f"slowWeight{suffix1}{ix}"].iloc[x].round(4),
                                        "hoursBefore": df[f"hoursBefore{suffix1}{ix}"].iloc[x].round(4)}
                                       for ix in range(num_of_params)] for x in df.index])
    elif any(df.columns.str.contains(f"slowWeight{suffix2}")):
        funding_weights2 = pd.Series([[{"fastWeight": df[f"fastWeight{suffix1}{ix}"].iloc[x].round(4),
                                        "slowWeight": df[f"slowWeight{suffix2}{ix}"].iloc[x].round(4),
                                        "hoursBefore": df[f"hoursBefore{suffix1}{ix}"].iloc[x].round(4)}
                                       for ix in range(num_of_params)] for x in df.index])
    else:
        funding_weights2 = funding_weights1

    return funding_weights1, funding_weights2


def production_csv_column_list(production_csv_path: str = None):
    df = pd.read_csv(production_csv_path, on_bad_lines='skip')
    renamed_columns = df.columns.tolist()
    duplicate_columns = df.loc[:, df.columns.str.contains(".1")].columns.tolist()
    df.rename(columns={col: col[:-2] for col in duplicate_columns}, inplace=True)
    original_columns = df.columns.tolist()
    return renamed_columns, original_columns


def btc_deribit_strategies_list():
    return [
        # "netopia_bitmex_perp_deribit_perp_btc_9",
        # "netopia_bitmex_perp_deribit_perp_btc_8",
        # "netopia_bitmex_perp_deribit_perp_btc_7",
        # "netopia_bitmex_perp_deribit_perp_btc_6",
        # "netopia_bitmex_perp_deribit_perp_btc_5",
        # "netopia_bitmex_perp_deribit_perp_btc_4",
        "netopia_bitmex_perp_deribit_perp_btc_3",
        "netopia_bitmex_perp_deribit_perp_btc_20",
        "netopia_bitmex_perp_deribit_perp_btc_2",
        "netopia_bitmex_perp_deribit_perp_btc_19",
        "netopia_bitmex_perp_deribit_perp_btc_18",
        # "netopia_bitmex_perp_deribit_perp_btc_17",
        "netopia_bitmex_perp_deribit_perp_btc_16",
        "netopia_bitmex_perp_deribit_perp_btc_15",
        # "netopia_bitmex_perp_deribit_perp_btc_14",
        "netopia_bitmex_perp_deribit_perp_btc_13",
        # "netopia_bitmex_perp_deribit_perp_btc_12",
        # "netopia_bitmex_perp_deribit_perp_btc_11",
        # "netopia_bitmex_perp_deribit_perp_btc_10",
        # "netopia_bitmex_perp_deribit_perp_btc_1",
        # "bequant_bitmex_perp_deribit_perp_btc_9",
        # "bequant_bitmex_perp_deribit_perp_btc_8",
        # "bequant_bitmex_perp_deribit_perp_btc_7",
        # "bequant_bitmex_perp_deribit_perp_btc_6",
        # "bequant_bitmex_perp_deribit_perp_btc_5",
        # "bequant_bitmex_perp_deribit_perp_btc_4",
        # "bequant_bitmex_perp_deribit_perp_btc_3",
        # "bequant_bitmex_perp_deribit_perp_btc_2",
        # "bequant_bitmex_perp_deribit_perp_btc_10",
        # "bequant_bitmex_perp_deribit_perp_btc_1"
    ]


def btc_bitmex_strategies_list():
    return [
        # "deribit_XBTUSD_maker_perpetual_2",
        "deribit_XBTUSD_maker_perpetual_16",
        # "deribit_XBTUSD_maker_perpetual_22",
        # "deribit_XBTUSD_maker_perpetual_23",
        # "deribit_XBTUSD_maker_perpetual_24",
        "deribit_XBTUSD_maker_perpetual_25",
        "deribit_XBTUSD_maker_perpetual_26",
        "deribit_XBTUSD_maker_perpetual_27",
        "deribit_XBTUSD_maker_perpetual_28",
        "deribit_XBTUSD_maker_perpetual_29",
        "deribit_XBTUSD_maker_perpetual_30",
        "deribit_XBTUSD_maker_perpetual_31",
        "deribit_XBTUSD_maker_perpetual_32",
        "deribit_XBTUSD_maker_perpetual_33",
        "deribit_XBTUSD_maker_perpetual_34",
        "deribit_XBTUSD_maker_perpetual_35",
        "deribit_XBTUSD_maker_perpetual_36",
        "deribit_XBTUSD_maker_perpetual_37",
        "deribit_XBTUSD_maker_perpetual_38",
        "deribit_XBTUSD_maker_perpetual_39",
        "deribit_XBTUSD_maker_perpetual_40"
    ]


def eth_bitmex_strategies_list():
    return [
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_9",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_8",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_7",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_6",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_5",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_4",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_3",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_25",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_24",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_23",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_22",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_21",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_20",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_2",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_19",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_18",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_17",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_16",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_15",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_14",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_13",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_12",
        "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_11",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_10",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_netopia_1",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_9",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_8",
        "deribit_ETH_perpetual_bitmex_ETHUSD_7",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_6",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_5",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_40",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_4",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_39",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_38",
        "deribit_ETH_perpetual_bitmex_ETHUSD_37",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_36",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_35",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_34",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_33",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_32",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_31",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_30",
        "deribit_ETH_perpetual_bitmex_ETHUSD_20",
        "deribit_ETH_perpetual_bitmex_ETHUSD_19",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_18",
        # "deribit_ETH_perpetual_bitmex_ETHUSD_15"
    ]


def eth_deribit_strategies_list():
    return ["equinox_bitmex_perp_deribit_perp_eth_1"]


if __name__ == "__main__":
    simulation_csv_path = "wandb_export_2024-06-27T12_23_21.243+02_00.csv"
    production_csv_path = "bands_from_postgres-5.csv"
    strategies_list = eth_bitmex_strategies_list()
    local_path = "/Users/konstantinospapastamatiou/Downloads/"
    num_of_params = 0
    has_generic_funding_parameters = False
    convert_simulation_csv_to_production_csv(simulation_csv_path=simulation_csv_path,
                                             production_csv_path=production_csv_path,
                                             strategies_list=strategies_list,
                                             local_path=local_path,
                                             num_of_params=num_of_params,
                                             has_generic_funding_parameters=has_generic_funding_parameters)
