import wandb
import numpy as np
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from src.common.queries.queries import get_entry_exit_bands
import datetime
import random
from src.common.utils.utils import get_data_for_trader
from src.simulations.taker_maker.TakerMakerDeeperLevelSpread import TakerMakerDeeperLevelSpread
from src.streamlit.streamlit_page_taker_maker_at_depth import ConstantDepthPosting

load_dotenv(find_dotenv())


def simulation_trader_taker_maker_at_depth(params):
    band = params.get('band', None)
    lookback = params.get('lookback', None)
    recomputation_time = params.get('recomputation_time', None)
    target_percentage_exit = params.get('target_percentage_exit', None)
    target_percentage_entry = params.get('target_percentage_entry', None)
    entry_opportunity_source = params.get('entry_opportunity_source', None)
    exit_opportunity_source = params.get('exit_opportunity_source', None)
    # start - end time
    t_start = params['t_start']
    t_end = params['t_end']

    # family and environment
    family = params['family']
    environment = params['environment']
    strategy = params['strategy']

    # exchanges
    exchange_spot = params['exchange_spot']
    exchange_swap = params['exchange_swap']
    instrument_spot = params['instrument_spot']
    instrument_swap = params['instrument_swap']

    entry_delta_spread = params['entry_delta_spread']
    exit_delta_spread = params['exit_delta_spread']

    # fees
    taker_fee_spot = params['taker_fee_spot']
    maker_fee_spot = params['maker_fee_spot']
    taker_fee_swap = params['taker_fee_swap']
    maker_fee_swap = params['maker_fee_swap']
    area_spread_threshold = params['area_spread_threshold']

    # latencies
    latency_spot = params['latency_spot']
    latency_try_post_spot = params[
        "latency_try_post_spot"]  # latency_col_2.number_input('Latency Trying to Post Spot', min_value=0, value=40, max_value=1000)
    latency_cancel_spot = params['latency_cancel_spot']
    latency_balance_spot = params['latency_balance_spot']
    latency_swap = params['latency_swap']
    latency_try_post_swap = params['latency_try_post_swap']
    latency_cancel_swap = params['latency_cancel_swap']
    latency_balance_swap = params['latency_balance_swap']

    # trade volume
    max_trade_volume = params['max_trade_volume']
    max_position = params['max_position']
    funding_system = params['funding_system']
    minimum_distance = params['minimum_distance']
    minimum_value = params['minimum_value']
    disable_when_below = params['disable_when_below']
    area_spread_threshold = params['area_spread_threshold']
    trailing_value = params['trailing_value']
    swap_market_tick_size = params['swap_market_tick_size']

    # configs
    use_backblaze = params.get("use_backblaze", False)
    use_wandb = params.get("use_wandb", False)
    send_alerts = params.get("send_alerts", False)

    # convert milliseconds to datetime
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)

    if send_alerts:
        # send message for simulation initialization
        now = datetime.datetime.now()
        dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
        data = {
            "message": f"Simulation (MM) of {strategy} from {date_start} to {date_end} Started at {dt_string_start} UTC",
        }

        # requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(data), headers={
        #     "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})

    depth_posting_predictor = ConstantDepthPosting(params['constant_depth'])

    file_id = random.randint(10 ** 6, 10 ** 7)

    # convert milliseconds to datetime
    date_start = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc)
    date_end = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc)

    # send message for simulation initialization
    now = datetime.datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
    data = {
        "message": f"Simulation (TM) of {strategy} from {date_start} to {date_end} Started at {dt_string_start} UTC",
    }

    band_values = get_entry_exit_bands(t0=t_start, t1=t_end, strategy=strategy, entry_delta_spread=entry_delta_spread,
                                       exit_delta_spread=exit_delta_spread, btype='central_band',
                                       environment=environment)
    band_values.rename(columns={'Band': 'Central Band'}, inplace=True)

    df, _ = get_data_for_trader(t_start, t_end, exchange_spot=exchange_spot, spot_instrument=instrument_spot,
                                exchange_swap=exchange_swap, swap_instrument=instrument_swap,
                                swap_fee=maker_fee_swap, spot_fee=taker_fee_spot, strategy=strategy,
                                area_spread_threshold=area_spread_threshold, environment=environment,
                                band_type="bogdan_bands", window_size=0, exit_delta_spread=exit_delta_spread,
                                entry_delta_spread=entry_delta_spread, band_funding_system=funding_system,
                                generate_percentage_bands=False,
                                lookback=None, recomputation_time=None, target_percentage_exit=None,
                                target_percentage_entry=None, entry_opportunity_source=None,
                                exit_opportunity_source=None,
                                minimum_target=None, use_aggregated_opportunity_points=None, ending=None,
                                force_band_creation=False, move_bogdan_band="No")

    model = TakerMakerDeeperLevelSpread(df=df, spot_fee=taker_fee_spot, swap_fee=maker_fee_swap,
                                        area_spread_threshold=area_spread_threshold, latency_spot=latency_spot,
                                        latency_swap=latency_swap,
                                        latency_try_post=latency_try_post_spot,
                                        latency_cancel=latency_cancel_swap, latency_spot_balance=latency_balance_spot,
                                        max_position=max_position, max_trade_volume=max_trade_volume,
                                        environment=environment, exchange_swap=exchange_swap,
                                        swap_instrument=instrument_swap,
                                        spot_instrument=instrument_spot, funding_system=funding_system,
                                        minimum_distance=minimum_distance, minimum_value=minimum_value,
                                        trailing_value=trailing_value, disable_when_below=disable_when_below,
                                        depth_posting_predictor=depth_posting_predictor,
                                        swap_market_tick_size=swap_market_tick_size, verbose=False)

    print("Starting simulation")
    while model.timestamp < t_end - 1000 * 60 * 5:
        for trigger in model.machine.get_triggers(model.state):
            if not trigger.startswith('to_'):
                if model.trigger(trigger):
                    break
    print("Done!")

    if len(model.executions) == 0:
        simulated_executions = pd.DataFrame(
            columns=["timems", "timestamp_swap_executed", "timestamp_spot_executed", "executed_spread",
                     "targeted_spread", "order_depth", "volume_executed", "entry_band", "exit_band", "price"
                                                                                                     "was_trying_to_cancel_swap",
                     "source_at_execution_swap", "dest_at_execution_swap", "side"])
    else:
        simulated_executions = pd.DataFrame(model.executions)

    if len(model.cancelled_orders_swap) == 0:
        simulated_cancellations = pd.DataFrame(
            columns=["Time posted", "Time cancelled", "timestamp_posted", "timestamp_cancelled",
                     "cancelled_to_post_deeper", "targeted_spread", "price", "max_targeted_depth", "side"])
        simulated_cancellations_entry = simulated_cancellations
        simulated_cancellations_exit = simulated_cancellations
    else:
        temp_cancellations = []
        for order in model.cancelled_orders_swap:
            temp_cancellations.append(
                [order.timestamp_posted, order.timestamp_cancelled, order.cancelled_to_post_deeper,
                 order.targeted_spread, order.price, order.max_targeted_depth, order.side])
        simulated_cancellations = pd.DataFrame(temp_cancellations, columns=["timestamp_posted", "timestamp_cancelled",
                                                                            "cancelled_to_post_deeper",
                                                                            "targeted_spread",
                                                                            "price", "max_targeted_depth", "side"])
        simulated_cancellations['Time posted'] = pd.to_datetime(simulated_cancellations['timestamp_posted'], unit='ms')
        simulated_cancellations['Time cancelled'] = pd.to_datetime(simulated_cancellations['timestamp_cancelled'],
                                                                   unit='ms')
        simulated_cancellations_entry = simulated_cancellations[simulated_cancellations.side == 'entry']
        simulated_cancellations_exit = simulated_cancellations[simulated_cancellations.side == 'exit']

    band_values['timems'] = band_values['Time'].view(np.int64) // 10 ** 6

    if use_wandb:
        band_values = band_values[~band_values["Central Band"].isnull()]
        simulated_executions = pd.merge_ordered(band_values[['Central Band', 'Entry Band', 'Exit Band', 'timems']],
                                                simulated_executions, on='timems')
        simulated_executions['Central Band'].ffill(inplace=True)
        simulated_executions['Entry Band'].ffill(inplace=True)
        simulated_executions['Exit Band'].ffill(inplace=True)
        simulated_executions['time_diff'] = simulated_executions['timems'].diff()

        if len(simulated_executions) == 0:
            return
        sim_ex_entry_mask = simulated_executions.side == 'entry'
        sim_ex_exit_mask = simulated_executions.side == 'exit'

        simulated_executions['execution_quality'] = np.nan
        simulated_executions['execution_quality'][sim_ex_entry_mask] = simulated_executions[
                                                                           sim_ex_entry_mask].executed_spread - \
                                                                       simulated_executions[sim_ex_entry_mask][
                                                                           'Entry Band']
        simulated_executions['execution_quality'][sim_ex_exit_mask] = -(
                    simulated_executions[sim_ex_exit_mask].executed_spread - simulated_executions[sim_ex_exit_mask][
                'Exit Band'])
        simulated_executions['Time'] = pd.to_datetime(simulated_executions['timems'], unit='ms')
        entries = simulated_executions[sim_ex_entry_mask]
        exits = simulated_executions[sim_ex_exit_mask]
        # Number of cancelled per side, number of executed per side
        n_executions_entry = len(entries)
        n_executions_exit = len(exits)
        avg_executed_spread_entry = round(entries.executed_spread.sum() / len(entries), 2)
        avg_executed_spread_exit = round(exits.executed_spread.sum() / len(exits), 2)
        avg_execution_quality_entry = round(entries.execution_quality.sum() / len(entries), 2)
        avg_execution_quality_exit = round(exits.execution_quality.sum() / len(exits), 2)
        total_volume_traded = simulated_executions.volume_executed.sum()
        total_volume_traded_in_token = (entries.volume_executed / entries.price).sum() + (
                    exits.volume_executed / exits.price).sum()
        n_cancelled_entry = len(simulated_cancellations_entry)
        n_cancelled_exit = len(simulated_cancellations_exit)
        n_trades_executed_both_sides = min(len(entries), len(exits))

        weighted_fixed_spread = round(
            (entries.executed_spread[:n_trades_executed_both_sides].sum() / (n_trades_executed_both_sides + 0.001)) -
            (exits.executed_spread[:n_trades_executed_both_sides].sum() / (n_trades_executed_both_sides + 0.001)), 2)
        funding_system = params['funding_system']
        minimum_distance = params['minimum_distance']
        minimum_value = params['minimum_value']
        disable_when_below = params['disable_when_below']
        area_spread_threshold = params['area_spread_threshold']
        output = {
            "stremlit_url": None,
            'file id': file_id,
            'timestamp_start': t_start,
            'start_date_dt': date_start.strftime("%Y-%m-%d"),
            'end_date_dt': date_end.strftime("%Y-%m-%d"),
            'lookback': lookback,
            'recomputation_time': recomputation_time,
            'trailing_value': trailing_value,
            'funding_system': funding_system,
            'minimum_distance': minimum_distance,
            'minimum_value': minimum_value,
            'disable_when_below': disable_when_below,
            'family': family,
            'environment': environment,
            'strategy': strategy,
            'exchange_spot': exchange_spot,
            'exchange_swap': exchange_swap,
            'instrument_spot': instrument_spot,
            'instrument_swap': instrument_swap,
            'maker_fee_spot': maker_fee_spot,
            'taker_fee_spot': taker_fee_spot,
            'maker_fee_swap': maker_fee_swap,
            'taker_fee_swap': taker_fee_swap,
            'area_spread_threshold': area_spread_threshold,
            'latency_spot': latency_spot,
            'latency_swap': latency_swap,
            'latency_try_post_spot': latency_try_post_spot,
            'latency_try_post_swap': latency_try_post_swap,
            'latency_cancel_spot': latency_cancel_spot,
            'latency_cancel_swap': latency_cancel_swap,
            'latency_balance_spot': latency_balance_spot,
            'latency_balance_swap': latency_balance_swap,
            'max_trade_volume': max_trade_volume,
            'max_position': max_position,
            'function': 'simulation_trader',
            'avg_executed_spread_entry': avg_executed_spread_entry,
            'avg_executed_spread_exit': avg_executed_spread_exit,
            'avg_execution_quality_entry': avg_execution_quality_entry,
            'avg_execution_quality_exit': avg_execution_quality_exit,
            'n_executions_entry': n_executions_entry,
            'n_executions_exit': n_executions_exit,
            'total_volume_traded': total_volume_traded,
            'total_volume_traded_in_token': total_volume_traded_in_token,
            'n_cancellations_entry': n_cancelled_entry,
            'n_cancellations_exit': n_cancelled_exit,
            'weighted_fixed_spread': weighted_fixed_spread
        }
        wandb.log(output)
        # wandb.finish()
        return output
