import pandas as pd
import numpy as np
from src.scripts.wandb_sweeps.process_sweep_results.download_wandb_results import AutomateParameterSelection


def mapping_real_params_to_simulation_params(df):
    rename_variables = {"window_size.1": "window_size2",
                        "entry_delta_spread.1": "entry_delta_spread2",
                        "exit_delta_spread.1": "exit_delta_spread2",
                        "current_ratio": "current_r",
                        "high_volatility_ratio": "high_r",
                        "high_volatility_threshold": "quanto_threshold",
                        "rolling_time_window_size_ratio": "ratio_entry_band_mov_ind",
                        "hours_to_stop_going_short_after_hr": "hours_to_stop",
                        "minimum_between_bands.1": "minimum_distance",
                        "rolling_time_window_size.1": "rolling_time_window_size"
                        }
    df.drop(columns=['rolling_time_window_size'], inplace=True)
    df.rename(columns=rename_variables, inplace=True)
    df["quanto_threshold"] = - df["quanto_threshold"]
    return df


def convert_df_to_wandb_format(df):
    result = []
    for idx in df.index:
        dictionary = {}
        for col in df.columns:
            if type(df.loc[idx, col]) == np.bool_:
                dictionary[col] = {"value": str(df.loc[idx, col])}
            elif type(df.loc[idx, col]) == np.int64:
                dictionary[col] = {"value": int(df.loc[idx, col])}
            elif type(df.loc[idx, col]) == str:
                dictionary[col] = {"value": df.loc[idx, col]}
            else:
                print(col, df.loc[idx, col])
                dictionary[col] = {"value": float(df.loc[idx, col])}
        result.append(dictionary)
    return result


def params_for_ethusd_short_go_long(df, band_funding_system: str = None):
    df = df.loc[(df["strategy"].str.contains("ETHUSD")) & (df.blended == False),
    ["window_size", "exit_delta_spread", "entry_delta_spread",
     "current_r", "high_r", "rolling_time_window_size", "ratio_entry_band_mov_ind",
     "hours_to_stop", "quanto_threshold"]]
    if band_funding_system is not None:
        df["band_funding_system"] = "funding_adjusted_band_swap_spot_with_drop"
    else:
        df["band_funding_system"] = band_funding_system
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


def params_for_xbtusd(df, band_funding_system: str = None, band_funding_system2: str = None):
    df = df.loc[df["strategy"].str.contains("XBTUSD") & (~df["strategy"].str.contains("_t")) & (df.blended == True),
    ["window_size", "exit_delta_spread", "entry_delta_spread",
     "window_size2", "exit_delta_spread2", "entry_delta_spread2"]]
    if band_funding_system is not None:
        df["band_funding_system"] = band_funding_system
        df["band_funding_system2"] = band_funding_system2
    else:
        df["band_funding_system"] = "funding_both_sides_no_netting_worst_case"
        df["band_funding_system2"] = "funding_both_sides_no_netting_worst_case"
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


def params_for_btc_deribit_maker(df, band_funding_system: str = None, band_funding_system2: str = None):
    df = df.loc[(df["strategy"].str.contains("bitmex_perp_deribit_perp_btc")) & (~df["strategy"].str.contains("_t")) & (
                df.blended == False),
    ["window_size", "exit_delta_spread", "entry_delta_spread",
     "window_size2", "exit_delta_spread2", "entry_delta_spread2"]]
    if band_funding_system is not None:
        df["band_funding_system"] = band_funding_system
        df["band_funding_system2"] = band_funding_system2
    else:
        df["band_funding_system"] = "funding_both_sides_no_netting_worst_case"
        df["band_funding_system2"] = "funding_both_sides_no_netting_worst_case"
    df["exchange_swap"] = "Deribit"
    df["swap_instrument"] = "BTC-PERPETUAL"
    df["exchange_spot"] = "BitMEX"
    df["spot_instrument"] = "XBTUSD"
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


def params_ethusd_combined_results(df):
    try:
        df = df[AutomateParameterSelection().ethusd_params_list()]
    except:
        df = df[["window_size", "exit_delta_spread", "entry_delta_spread",
                 "current_r", "high_r", "rolling_time_window_size", "ratio_entry_band_mov_ind",
                 "hours_to_stop", "quanto_threshold", "band_funding_system"]]
    if "funding_window" not in df.columns:
        df['funding_window'] = df['funding_window'].fillna(90).astype(int)
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


def generic_columns(column_names: list = None):
    generic_swap = AutomateParameterSelection().generic_funding_swap_params_list()
    generic_spot = AutomateParameterSelection().generic_funding_spot_params_list()
    if 'fastWeightSwap2' not in column_names:
        generic_swap.remove('fastWeightSwap2')

    if 'fastWeightSpot2' not in column_names:
        generic_spot.remove("fastWeightSpot2")

    if 'slowWeightSwap2' not in column_names:
        generic_swap.remove("slowWeightSwap2")

    if 'slowWeightSpot2' not in column_names:
        generic_spot.remove("slowWeightSpot2")

    if 'hoursBeforeSwap0' not in column_names:
        generic_swap.remove("hoursBeforeSwap0")

    if 'hoursBeforeSpot0' not in column_names:
        generic_spot.remove("hoursBeforeSpot0")

    if 'hoursBeforeSwap1' not in column_names:
        generic_swap.remove("hoursBeforeSwap1")

    if 'hoursBeforeSpot1' not in column_names:
        generic_spot.remove("hoursBeforeSpot1")

    if 'hoursBeforeSwap2' not in column_names:
        generic_swap.remove("hoursBeforeSwap2")

    if 'hoursBeforeSpot2' not in column_names:
        generic_spot.remove("hoursBeforeSpot2")

    if 'use_same_slowSwap_slowSpot_generic_funding' in column_names:
        generic_swap.append('use_same_slowSwap_slowSpot_generic_funding')

    if 'use_same_window_size' in column_names:
        generic_swap.append('use_same_window_size')

    if 'adjust_pnl_automatically' in column_names:
        generic_swap.append('adjust_pnl_automatically')

    if 'maximum_quality' in column_names:
        generic_swap.append('maximum_quality')

    return generic_swap + generic_spot


def check_additional_columns(df: pd.DataFrame = None):
    if "window_size2" not in df.columns:
        df['window_size2'] = df['window_size']

    if "max_trade_volume" not in df.columns:
        df['adjust_pnl_automatically'] = int(3000)

    return df


def params_xbtusd_combined_results(df, band_funding_system: str = "funding_both_sides_no_netting_worst_case",
                                   band_funding_system2: str = "funding_both_sides_no_netting_worst_case"):
    df = check_additional_columns(df)

    try:
        df = df[AutomateParameterSelection().xbtusd_params_list() + generic_columns(column_names=df.columns)]
    except:
        df = df[AutomateParameterSelection().xbtusd_params_list()]
    if "band_funding_system" not in df.columns:
        df["band_funding_system"] = band_funding_system
    if "band_funding_system2" not in df.columns:
        df["band_funding_system2"] = band_funding_system2
    if "funding_window" not in df.columns:
        df['funding_window'] = df['funding_window'].fillna(90).astype(int)
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


def params_btc_deribit_maker_combined_results(df, band_funding_system: str = "funding_continuous_weight_concept",
                                              band_funding_system2: str = "funding_continuous_weight_concept"):
    df = check_additional_columns(df)

    try:
        df = df[AutomateParameterSelection().xbtusd_params_list() + generic_columns(column_names=df.columns)]
    except:
        df = df[AutomateParameterSelection().xbtusd_params_list()]

    if 'band_funding_system' not in df.columns:
        df["band_funding_system"] = band_funding_system
        df["band_funding_system2"] = band_funding_system2

    df["exchange_swap"] = "Deribit"
    df["swap_instrument"] = "BTC-PERPETUAL"
    df["exchange_spot"] = "BitMEX"
    df["spot_instrument"] = "XBTUSD"
    df.dropna(inplace=True)
    res = convert_df_to_wandb_format(df)
    return res


if __name__ == "__main__":
    # df = pd.read_csv("/Users/konstantinospapastamatiou/Downloads/bands_from_postgres-6.csv")
    # df1 = mapping_real_params_to_simulation_params(df)
    # df1 = params_for_ethusd_short_go_long(df1)
    df = pd.read_csv("XBTUSD_deribit_generic_database_25_06_2024.csv")
    df1 = params_btc_deribit_maker_combined_results(df)
    a = 3
