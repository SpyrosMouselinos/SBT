import argparse
import time
from src.common.utils.utils import parse_args
from src.common.equinox_api_call import DatalinkCreateBands

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--t_start', default=1670108400000, type=int)
    parser.add_argument('--t_end', default=1673737200000, type=int)
    parser.add_argument('--exchange_spot', default="Binance", type=str)
    parser.add_argument('--exchange_swap', default="Binance", type=str)
    parser.add_argument('--spot_instrument', default="okex_chz-usdt-swap", type=str)
    parser.add_argument('--swap_instrument', default="binance_futures_chzusdt", type=str)
    parser.add_argument('--token_list', default="binance_futures_btcusdt~binance_swap_btcusd_perp", type=str)
    parser.add_argument('--broker_name', default="hidden_road", type=str)
    parser.add_argument('--window_sizes', default="500,800,1000,1200,1500,1800,2000,3000", type=str)
    parser.add_argument('--band_type', default="bogdan_bands", type=str)
    parser.add_argument('--environment', default="staging", type=str)
    parser.add_argument('--band_funding_system', default='No', type=str)
    parser.add_argument('--funding_window', default= 90, type=int)
    parser.add_argument('--move_bogdan_band', default='No', type=str)
    parser.add_argument('--funding_system', default='No', type=str)
    parser.add_argument('--generate_percentage_bands', default='False', type=str)
    parser.add_argument('--lookback', default=None, type=str)
    parser.add_argument('--recomputation_time', default=None, type=str)
    parser.add_argument('--target_percentage_exit', default=None, type=str)
    parser.add_argument('--target_percentage_entry', default=None, type=str)
    parser.add_argument('--entry_opportunity_source', default=None, type=str)
    parser.add_argument('--exit_opportunity_source', default=None, type=str)
    parser.add_argument('--use_bp', default='True', type=str)
    parser.add_argument('--use_stored_bands', default='True', type=str)

    params = vars(parse_args(parser))

    tokens_to_use = [item for item in params.get('token_list', '').split(',') if len(item) > 0]
    params['family'] = 'Other'
    params['strategy'] = ''
    params['area_spread_threshold'] = 0
    params['exit_delta_spread'] = 4
    params['entry_delta_spread'] = 4
    params['force_band_creation'] = True
    params['move_bogdan_band'] = 'No'


    for pair_name in tokens_to_use:
        instruments = pair_name.split("~")
        params["spot_instrument"] = instruments[0]
        params["swap_instrument"] = instruments[1]
        if params['broker_name'] == "hidden_road":
            swap_fee_usdt, swap_fee_busd, swap_fee_coin = -0.00005, -0.00014, -0.00015
            spot_fee_usdt, spot_fee_busd, spot_fee_coin = 0.0001275, 0.0002185, 0.00024
        elif params['broker_name'] == "bitmain":
            swap_fee_usdt, swap_fee_busd, swap_fee_coin = -0.00004, -0.00014, -0.00009
            spot_fee_usdt, spot_fee_busd, spot_fee_coin = 0.00017, 0.00023, 0.00024
        spot_fee, swap_fee = spot_fee_usdt, swap_fee_coin
        if "usdt" in instruments[1]:
            swap_fee = swap_fee_usdt
        elif "busd" in instruments[1]:
            swap_fee = swap_fee_busd
        elif "perp" in instruments[1]:
            swap_fee = swap_fee_coin
        if "usdt" in instruments[0]:
            spot_fee = spot_fee_usdt
        elif "busd" in instruments[0]:
            spot_fee = spot_fee_busd
        elif "perp" in instruments[0]:
            spot_fee = spot_fee_coin
        params['spot_fee'] = spot_fee
        params['swap_fee'] = swap_fee
        for window_size_ in params['window_sizes'].split(","):
            window_size = int(window_size_)
            params['window_size'] = window_size
            if params['band_type'] == 'bogdan_bands':
                start = time.time()
                if (params['force_band_creation'] or params['strategy'] == ''):
                    datalink = DatalinkCreateBands(t_start=params['t_start'] - 1000 * 60 * (window_size + 10),
                                                   t_end=params['t_end'], swap_exchange=params['exchange_swap'],
                                                   swap_symbol=params['swap_instrument'],
                                                   spot_exchange=params['exchange_spot'],
                                                   spot_symbol=params['spot_instrument'], window_size=window_size,
                                                   entry_delta_spread=params['entry_delta_spread'],
                                                   exit_delta_spread=params['exit_delta_spread'], swap_fee=swap_fee,
                                                   spot_fee=spot_fee,
                                                   generate_percentage_bands=params['generate_percentage_bands'],
                                                   funding_system=params['band_funding_system'],
                                                   funding_window=params['funding_window'],
                                                   environment=params['environment'],
                                                   recomputation_time=params['recomputation_time'],
                                                   entry_opportunity_source=params['entry_opportunity_source'],
                                                   exit_opportunity_source=params['exit_opportunity_source'],
                                                   target_percentage_entry=params['target_percentage_entry'],
                                                   target_percentage_exit=params['target_percentage_exit'],
                                                   lookback=params['lookback'], minimum_target=None,
                                                   use_aggregated_opportunity_points=None, ending=None,
                                                   use_bps=params['use_bp'])
                    if params['use_stored_bands']:
                        band_values = datalink.load_bands_from_disk()
                        print(f"Done bands with window size {window_size} for pair {pair_name}")
