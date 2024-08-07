import traceback
import pika
import os
import json
import threading
import functools
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function import simulation_trader

functions = {'simulation_trader': simulation_trader
             # 'simulation_trader_maker_maker': simulation_trader_maker_maker
             }


class TakerMakerWorker:

    def __init__(self, queue="simulation_rpc_queue"):
        """
         @brief Initialize the connection to RabbitMQ. This is called by __init__ and should not be called directly
         @param queue The name of the
        """
        self.credentials = pika.PlainCredentials(os.getenv("RABBITMQ_USERNAME"), os.getenv("RABBITMQ_PASSWORD"))
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOSTNAME"), credentials=self.credentials, heartbeat=5))

        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=queue)
        self.threads = []

    def on_message(self, ch, method, props, body):
        delivery_tag = method.delivery_tag

        t = threading.Thread(target=self.work, args=(self.connection, self.channel, delivery_tag, body))
        t.start()
        self.threads.append(t)

    def work(self, connection, channel, delivery_tag, body):
        thread_id = threading.get_ident()
        print(f'Thread id: {thread_id} Delivery tag: {delivery_tag}')

        params = json.loads(body)
        print(f"Received simulation with params: {json.dumps(params, indent=2)}")

        try:
            functions[params['function']](params=params)
        except Exception as e:
            traceback.print_exc()
            # message = {
            #     "message": f"Simulation with params {json.dumps(params, indent=2)} failed, Error : {e}",
            # }
            # requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message), headers={
            #     "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})

        cb = functools.partial(self.ack_message, channel, delivery_tag)
        connection.add_callback_threadsafe(cb)

    def on_debug(self, body):
        params = json.loads(body)
        print(f"Received simulation with params: {json.dumps(params, indent=2)}")

        functions[params['function']](params=params)

    def ack_message(self, channel, delivery_tag):

        if channel.is_open:
            channel.basic_ack(delivery_tag)
        else:
            # Channel is already closed, so we can't ACK this message;
            pass


if __name__ == '__main__':
    worker = TakerMakerWorker()
    # body = {
    #     "t_start": 1657497600000,
    #     "t_end": 1657584000000,
    #     "band": "bogdan_bands",
    #     "lookback": [
    #         None
    #     ],
    #     "recomputation_time": [
    #         None
    #     ],
    #     "target_percentage_entry": [
    #         None
    #     ],
    #     "target_percentage_exit": [
    #         None
    #     ],
    #     "entry_opportunity_source": [
    #         None
    #     ],
    #     "exit_opportunity_source": None,
    #     "family": "deribit_xbtusd",
    #     "environment": "production",
    #     "strategy": "deribit_XBTUSD_maker_perpetual_20",
    #     "exchange_spot": "Deribit",
    #     "exchange_swap": "BitMEX",
    #     "spot_instrument": "BTC-PERPETUAL",
    #     "swap_instrument": "XBTUSD",
    #     "spot_fee": 0.0003,
    #     "swap_fee": -0.0001,
    #     "area_spread_threshold": 0.0,
    #     "latency_spot": 150,
    #     "latency_swap": 60,
    #     "latency_try_post": 115,
    #     "latency_cancel": 120,
    #     "latency_spot_balance": 40,
    #     "max_trade_volume": 3000,
    #     "max_position": 275000,
    #     "function": "simulation_trader"
    # }

    # worker.on_debug(body=json.dumps(body))
    worker.channel.basic_qos(prefetch_count=1)
    worker.channel.basic_consume(queue='simulation_rpc_queue', on_message_callback=worker.on_message)

    print(" [x] Awaiting RPC requests")
    worker.channel.start_consuming()
