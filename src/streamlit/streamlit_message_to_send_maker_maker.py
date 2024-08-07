import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.common.constants.constants import exchange_fees, set_latencies_auto
from src.common.queries.queries import get_strategy_families, get_symbol, get_strategy_influx, get_exhange_names
from src.common.clients.rpc_client import RpcClient
import datetime


load_dotenv(find_dotenv())


def streamlit_trader_message():

    if st.button("Clear All"):
        # Clears all singleton caches:
        st.experimental_singleton.clear()
    # if os.path.exists('logs_trader.txt'):
    #     os.remove("logs_trader.txt")
    st.sidebar.write("Trader Simulator in order to discover new markets and trading opportunities")
    st.title('Trading Simulator')
    actual_exists = st.sidebar.selectbox('Are there any real executions recorded? :', ('no', 'yes'))
    st.sidebar.write('"Yes": if there are real executions, "No":  if there are no real executions')
    band = st.sidebar.selectbox('Select the type of bands you want to use', ('bogdan_bands', 'percentage_bogdan_bands', 'quanto_profit', 'quanto_profit_additional', 'percentage_band'))
    if band == 'percentage_band':
        lookback = st.sidebar.text_input('lookback')
        recomputation_time = st.sidebar.text_input('recomputation_time')
        target_percentage_exit = st.sidebar.number_input('target_percentage_exit')
        target_percentage_entry = st.sidebar.number_input('target_percentage_entry')
        entry_opportunity_source = st.sidebar.selectbox('entry_opportunity_source', ('0', 'entry_with_takers', 'entry_with_takers_latency_200'))
        exit_opportunity_source = st.sidebar.selectbox('exit_opportunity_source', ('1', 'exit_with_takers', 'exit_with_takers_latency_200'))
    else:
        lookback = None,
        recomputation_time = None,
        target_percentage_exit = None,
        target_percentage_entry = None,
        entry_opportunity_source = None,
        exit_opportunity_source = None

    date_range = st.date_input("Input a range of time report", [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
    t_start = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month, day=date_range[0].day).timestamp() * 1000)
    t_end = int(datetime.datetime(year=date_range[1].year, month=date_range[1].month, day=date_range[1].day).timestamp() * 1000)
    #t_end = t_start + 1000 * 60 * 60 * 2
    st.write('The ending time in milliseconds', t_end)
    st.text('Default time-range is 1 day')
    st.write('Select the strategies family and strategy you want to review')
    st.write('If you want to review a new combination of exchanges select "Other"')

    col1, col2, col3 = st.columns(3)
    family = col1.selectbox('Strategy family', ('deribit_xbtusd', 'deribit_eth', 'Other'))
    environment = col2.selectbox('Environment from where data are downloaded', ('production', 'staging', 'server'))
    if family == 'Other':
        strategy = col3.selectbox('Give the strategy name:', get_strategy_influx(environment=environment))
    elif family == 'deribit_xbtusd':
        strategy = col3.selectbox('Select the strategy', get_strategy_families(t0=t_start, environment='production')[family], index=17)
    else:
        strategy = col3.selectbox('Select the strategy', get_strategy_families(t0=t_start, environment='production')[family])


    col_1, col_2, col_3, col_4 = st.columns(4)

    if family == 'Other':
        exchange_spot = col_1.selectbox('ExchangeSpot', get_exhange_names(t0=t_start, t1=t_end, environment=environment))
        exchange_swap = col_2.selectbox('Exchange Swap', get_exhange_names(t0=t_start, t1=t_end, environment=environment))
        spot_instrument = col_3.selectbox('Spot Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot,
                                                                        environment=environment))
        swap_instrument = col_4.selectbox('Swap Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap,
                                                                        environment=environment))
    else:
        exchange_spot = col_1.selectbox('Exchange Spot', ('Deribit', 'BitMEX'))
        exchange_swap = col_2.selectbox('Exchange Swap', ('BitMEX', 'Deribit'))
        spot_instrument = col_3.selectbox('Spot Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_spot, environment=environment))
        swap_instrument = col_4.selectbox('Swap Instrument', get_symbol(t0=t_start, t1=t_end, exchange=exchange_swap, environment=environment), index=3)

    fee_col_1, fee_col_2, fee_col_3, fee_col_4 = st.columns(4)
    default_maker_fee_swap, default_taker_fee_swap = exchange_fees(exchange_swap, swap_instrument, exchange_swap, swap_instrument)
    default_maker_fee_spot, default_taker_fee_spot = exchange_fees(exchange_spot, spot_instrument, exchange_spot, spot_instrument)
    taker_fee_spot = fee_col_1.number_input('Spot Fee Taker', min_value=-1.0, value=default_taker_fee_spot, max_value=1.0, step=0.00001, format="%.6f")
    maker_fee_spot = fee_col_2.number_input('Spot Fee Maker', min_value=-1.0, value=default_maker_fee_spot, max_value=1.0, step=0.00001, format="%.6f")
    taker_fee_swap = fee_col_3.number_input('Swap Fee Taker', min_value=-1.0, value=default_taker_fee_swap, max_value=1.0, step=0.00001, format="%.6f")
    maker_fee_swap = fee_col_4.number_input('Swap Fee Maker', min_value=-1.0, value=default_maker_fee_swap, max_value=1.0, step=0.00001, format="%.6f")


    # latencies default values
    ws_swap, api_swap, ws_spot, api_spot = set_latencies_auto(exchange_swap, exchange_spot)
    # latencies
    latency_col_1, latency_col_2, latency_col_3, latency_col_4 = st.columns(4)
    latency_spot = latency_col_1.number_input('Latency Spot', min_value=0, value=ws_spot, max_value=1000)
    latency_try_post_spot = latency_col_2.number_input('Latency Trying to Post Spot', min_value=0, value=api_spot, max_value=1000)
    latency_cancel_spot = latency_col_3.number_input('Latency Cancel Spot', min_value=0, value=api_spot, max_value=1000)
    latency_balance_spot = latency_col_4.number_input('Latency Balance Spot', min_value=0, value=api_swap, max_value=1000)
    latency_col_5, latency_col_6, latency_col_7, latency_col_8 = st.columns(4)
    latency_swap = latency_col_5.number_input('Latency Swap', min_value=0, value=ws_swap, max_value=1000)
    latency_try_post_swap = latency_col_6.number_input('Latency Trying to Post Swap', min_value=0, value=api_swap, max_value=1000)
    latency_cancel_swap = latency_col_7.number_input('Latency Cancel Swap', min_value=0, value=api_swap, max_value=1000)
    latency_balance_swap = latency_col_8.number_input('Latency Balance Swap', min_value=0, value=api_spot, max_value=1000)

    slippage_col_1, slippage_col_2, displacement_col_1, area_spread_col_1 = st.columns(4)
    taker_slippage_spot = slippage_col_1.number_input('Slippage Spot', min_value=0.0, value=4, max_value=100.0)
    taker_slippage_swap = slippage_col_2.number_input('Slippage Swap', min_value=0.0, value=4, max_value=100.0)
    displacement = displacement_col_1.number_input('Displacement', min_value=0, value=5, max_value=100.0)
    area_spread_threshold = area_spread_col_1.number_input('Area Spread Threshold', min_value=0, value=0, max_value=100)


    col_ts1, col_ts2 = st.columns(2)
    max_trade_volume = col_ts1.number_input('Max Trade Volume', min_value=0, value=4000, max_value=100000)
    max_position = col_ts2.number_input('Max Position', min_value=0, value=100000, max_value=1000000)

    st.subheader('When ready with parameter input click the button')
    check = st.checkbox('Click Here')
    st.write('State of the checkbox: ', check)

    if check:
        body = {'t_start': t_start, 't_end': t_end, 'band': band,
                'lookback': lookback, 'recomputation_time': recomputation_time,
                'target_percentage_entry': target_percentage_entry, 'target_percentage_exit': target_percentage_exit,
                'entry_opportunity_source': entry_opportunity_source, 'exit_opportunity_source': exit_opportunity_source,
                'family': family, 'environment': environment, 'strategy': strategy, 'exchange_spot': exchange_spot,
                'exchange_swap': exchange_swap, 'spot_instrument': spot_instrument, 'swap_instrument': swap_instrument,
                'taker_fee_spot': taker_fee_spot, 'maker_fee_spot': maker_fee_spot, 'taker_fee_swap': taker_fee_swap,
                'maker_fee_swap': maker_fee_swap, 'area_spread_threshold': area_spread_threshold,
                'latency_spot': latency_spot, 'latency_try_post_spot': latency_try_post_spot,
                'latency_cancel_spot': latency_cancel_spot, 'latency_balance_spot': latency_balance_spot,
                'latency_swap': latency_swap, 'latency_try_post_swap': latency_try_post_swap,
                'latency_cancel_swap': latency_cancel_swap, 'latency_balance_swap': latency_balance_swap,
                'taker_slippage_spot': taker_slippage_spot, 'taker_slippage_swap': taker_slippage_swap,
                'displacement': displacement,
                'max_trade_volume': max_trade_volume, 'max_position': max_position,
                'function': 'simulation_trader_maker_maker'}

        client = RpcClient()
        client.call(body, routing_key='simulation_rpc_queue_maker_maker')
        now = datetime.datetime.now()
        dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")





