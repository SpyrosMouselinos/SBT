import os
import math
import requests
import wandb
import json
import pandas as pd
import subprocess
from time import sleep
from b2sdk.v2 import *
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
# -------------------------------------------#
from src.common.constants.simulation_constants import *
from src.common.connections.DatabaseConnections import InfluxConnection, PostgresConnection
from src.common.utils.utils import Util

load_dotenv(find_dotenv())


def is_training_running(host, sweep_id):
    my_sweep_cmd = ''' 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep "sweep_''' + sweep_id + '''" | awk "NR==1{print $1}"' '''
    # Check if there is any sweep running
    cmd = f"ssh {host} {my_sweep_cmd}"
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    print(out.decode('utf-8').strip())
    if "sweep" in out.decode('utf-8').strip():
        print(f"Training sweep with id {out.decode('utf-8').strip().split('_')[1]} is still running at host {host}")
        return True
    return False


def is_controller_running(sweep_id, host):
    controller_sweep_cmd = f''' 'ps -aux | grep "sweep_{sweep_id}" | wc -l' '''
    # Check if there is any sweep running
    cmd = f"ssh {host} {controller_sweep_cmd}"
    print(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    print(out.decode('utf-8').strip())
    if int(out.decode('utf-8').strip()) > 2:
        print(f"Controller with sweep_id {sweep_id} is still running at host {host}")
        return True
    return False


class DataFetcher:
    def __init__(self):
        self.client = InfluxConnection.getInstance().staging_client_spotswap
        self.client_archival = InfluxConnection.getInstance().archival_client_spotswap

    def fetch_and_upload(self, base_path, exchange, symbol, start, end):

        t_start = start
        t_end = start + timedelta(days=1)
        while t_end <= end:
            print(t_start, t_end)
            if base_path == "prices":
                print("Upload Prices")
                # PRICES
                result = self.client.query(f'''SELECT "price","side" FROM "price" WHERE ("exchange" = '{exchange}' AND
                                         symbol = '{symbol}') AND time > '{t_start}' AND time < '{t_end}' ''')
                points = result.get_points()
                prices = pd.DataFrame(Util.create_array_from_generator(points), columns=['timestamp', 'price', 'side'])
                if len(prices.index) < 3:
                    message = {
                        "message": f"No prices found in influx from {t_start} to {t_end} for {symbol}@{exchange}",
                    }
                    requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                                  headers={
                                      "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
                    continue
                day = t_start
                print(day, type(day))
                filename = f"{exchange}_{symbol}_{day.strftime('%Y-%m-%d')}.parquet.br"
                prices.to_parquet(filename, engine="pyarrow", compression='brotli')
                self.upload_prices_to_backblaze(filename=filename, exchange=exchange, symbol=symbol)
            # TRADES
            elif base_path == "trades":
                print("Upload Trades")
                if exchange == 'Deribit':
                    result = self.client.query(
                        f'''SELECT "price","size","side" FROM "trade" WHERE ("exchange" = '{exchange}' AND
                                                        symbol = '{symbol}') AND time > '{t_start}' AND time < '{t_end}' ''')
                else:
                    result = self.client_archival.query(
                        f'''SELECT "price","size","side" FROM "trade" WHERE ("exchange" = '{exchange}' AND
                                                                 symbol = '{symbol}') AND time > '{t_start}' AND time < '{t_end}' ''')
                points = result.get_points()
                arr = Util.create_array_from_generator(points)
                if len(arr) == 0:
                    message = {
                        "message": f"No trades found in influx from {t_start} to {t_end} for {symbol}@{exchange}",
                    }
                    requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                                  headers={
                                      "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
                    t_start = t_start + timedelta(days=1)
                    t_end = t_end + timedelta(days=1)
                    continue
                trades = pd.DataFrame(arr, columns=['timestamp', 'price', 'size', 'side'])
                day = t_start
                filename = f"{exchange}_{symbol}_{day.strftime('%Y-%m-%d')}.parquet.br"
                trades.to_parquet(filename, engine="pyarrow", compression='brotli')
                self.upload_trades_to_backblaze(filename=filename, exchange=exchange, symbol=symbol)
            # REAL TIME FUNDING
            elif base_path == "real_time_funding":
                print("Upload Real Time Funding")
                denormalized_factor = 8 * 60 * 60

                result = self.client.query(
                    f'''
                SELECT mean("funding")/{denormalized_factor} as "funding" 
                FROM "real_time_funding" WHERE ("exchange" = '{exchange}' AND symbol = '{symbol}') 
                AND time > '{t_start}' AND time < '{t_end}' GROUP BY time(1s) 
                '''
                )
                points = result.get_points()
                funding = pd.DataFrame(Util.create_array_from_generator(points), columns=['time', 'funding'])
                if len(funding.index) < 3:
                    message = {
                        "message": f"No real time funding found in influx from {t_start} to {t_end} for {symbol}@{exchange}",
                    }
                    requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                                  headers={
                                      "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
                    continue
                day = t_start
                print(day, type(day))
                filename = f"{exchange}_{symbol}_{day.strftime('%Y-%m-%d')}.parquet.br"
                funding.to_parquet(filename, engine="pyarrow", compression='brotli')
                self.upload_real_time_funding_to_backblaze(filename=filename, exchange=exchange, symbol=symbol)
                # FUNDING
            elif base_path == "funding":
                print("Upload Funding")
                result = self.client.query(
                    f'''
                        SELECT "funding"
                        FROM "funding" WHERE ("exchange" = '{exchange}' AND symbol = '{symbol}') 
                        AND time > '{t_start}' AND time < '{t_end}' 
                    '''
                )
                points = result.get_points()
                funding = pd.DataFrame(Util.create_array_from_generator(points), columns=['time', 'funding'])
                if len(funding.index) < 2:
                    message = {
                        "message": f"No funding found in influx from {t_start} to {t_end} for {symbol}@{exchange}",
                    }
                    requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                                  headers={
                                      "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
                    continue
                day = t_start
                print(day, type(day))
                filename = f"{exchange}_{symbol}_{day.strftime('%Y-%m-%d')}.parquet.br"
                funding.to_parquet(filename, engine="pyarrow", compression='brotli')
                self.upload_funding_to_backblaze(filename=filename, exchange=exchange, symbol=symbol)
            # date +1 day
            t_start = t_start + timedelta(days=1)
            t_end = t_end + timedelta(days=1)
        return

    def upload_prices_to_backblaze(self, filename, exchange, symbol):
        return self.upload_to_backblaze("prices", filename, exchange, symbol)

    def upload_trades_to_backblaze(self, filename, exchange, symbol):
        return self.upload_to_backblaze("trades", filename, exchange, symbol)

    def upload_real_time_funding_to_backblaze(self, filename, exchange, symbol):
        if exchange == 'Deribit':
            return self.upload_to_backblaze("real_time_funding", filename, exchange, symbol)

    def upload_funding_to_backblaze(self, filename, exchange, symbol):
        if exchange != 'Deribit':
            return self.upload_to_backblaze("funding", filename, exchange, symbol)

    def upload_to_backblaze(self, base_path, filename, exchange, symbol):
        print(f"Ready to upload parquet file {filename}")
        # backblaze
        b2_api = B2Api()
        application_key_id = os.getenv("BACKBLAZE_KEY_ID")
        application_key = os.getenv("BACKBLAZE_KEY_SECRET")
        b2_api.authorize_account("production", application_key_id, application_key)
        backblaze_bucket = b2_api.get_bucket_by_name(os.getenv('BACKBLAZE_BUCKET_NAME'))
        backblaze_folder = f"{base_path}/{exchange}/{symbol}"
        if os.path.getsize(filename) < 2000:
            message = {
                "message": f"File {filename} size is {os.path.getsize(filename) / 1024}kB something went wrong",
            }
            requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                          headers={
                              "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
            return
        try:
            backblaze_bucket.upload_local_file(
                local_file=filename,
                file_name=f"{backblaze_folder}/{filename}"
            )
            print('Successfully uploaded to backblaze')
            os.remove(filename)
        except Exception as e:
            message = {
                "message": f"Failed to upload {filename} to backblaze {e}",
            }
            requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                          headers={
                              "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
        return

    def download_prices(self, exchange, symbol, start_ms, end_ms):
        return self.download("prices", exchange, symbol, start_ms, end_ms)

    def download_trades(self, exchange, symbol, start_ms, end_ms):
        return self.download("trades", exchange, symbol, start_ms, end_ms)

    def download_real_time_funding(self, exchange, symbol, start_ms, end_ms):
        if exchange == 'Deribit':
            return self.download("real_time_funding", exchange, symbol, start_ms, end_ms)

    def download_funding(self, exchange, symbol, start_ms, end_ms):
        if exchange != 'Deribit':
            return self.download("funding", exchange, symbol, start_ms, end_ms)

    def download(self, base_path, exchange, symbol, start_ms, end_ms):
        ##---------------##
        start = (datetime.fromtimestamp(start_ms / 1000)).date()
        end = (datetime.fromtimestamp(end_ms / 1000))
        # If the end date does not end on 12 AM we need to add one day to the end date
        if not (end.hour == 0 and end.minute == 0 and end.second == 0):
            end = end + timedelta(days=1)
        end = end.date()
        ##---------------##



        t_start = start
        t_end = start + timedelta(days=1)
        while t_end <= end:
            print(t_start, t_end)
            day = t_start

            status = self.download_from_backblaze(base_path, f"{exchange}_{symbol}_{day}.parquet.br", exchange=exchange,
                                                  symbol=symbol)
            if not status:
                self.fetch_and_upload(base_path=base_path, exchange=exchange, symbol=symbol, start=t_start, end=t_end)
                retry_status = self.download_from_backblaze(base_path, f"{exchange}_{symbol}_{day}.parquet.br",
                                                            exchange=exchange, symbol=symbol)
                if not retry_status:
                    print()
                    message = {
                        "message": f"Error while downloading {exchange}_{symbol}_{day}.parquet.br from backblaze. "
                                   f"Exiting...",
                    }
                    requests.post(f"https://nodered.equinoxai.com/simulation_alerts", data=json.dumps(message),
                                  headers={
                                      "Content-Type": "application/json", "Cookie": os.getenv("AUTHELIA_COOKIE")})
                    continue

            t_start = t_start + timedelta(days=1)
            t_end = t_end + timedelta(days=1)
        return True

    def download_from_backblaze(self, base_path, filename, exchange, symbol):
        print(f"Ready to download parquet file {base_path}/{filename}")
        # backblaze
        b2_api = B2Api()
        application_key_id = os.getenv("BACKBLAZE_KEY_ID")
        application_key = os.getenv("BACKBLAZE_KEY_SECRET")
        b2_api.authorize_account("production", application_key_id, application_key)
        backblaze_bucket = b2_api.get_bucket_by_name(os.getenv('BACKBLAZE_BUCKET_NAME'))
        backblaze_folder = f"{base_path}/{exchange}/{symbol}"
        base_dir = f"/home/equinoxai/data"
        if not os.path.isdir(base_dir):
            base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        base_dir = f"{base_dir}/{base_path}/{exchange}/{symbol}"
        try:
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            if os.path.exists(f"{base_dir}/{filename}"):
                print(f"File {filename} already exists.")
                return True
            downloaded_file = backblaze_bucket.download_file_by_name(f"{backblaze_folder}/{filename}")
            with open(f"{base_dir}/{filename}", 'wb') as f:
                downloaded_file.save(f)
                print(f"Successfully downloaded file: {filename}")
            return True
        except Exception as e:
            print(f"Failed to download from backblaze {e}")
            return False


class SimulationsDataQuality:

    def __init__(self, t_start, t_end, exchange, symbol):
        self.t_start = t_start
        self.t_end = t_end
        self.exchange = exchange
        self.symbol = symbol
        self.client = InfluxConnection.getInstance().staging_client_spotswap

    def start(self):
        print("Test")
        self.check_price_data()

    def check_price_data(self):
        result = self.client.query(f'''SELECT "price","side" FROM "price" WHERE ("exchange" = '{self.exchange}' AND
                                             symbol = '{self.symbol}') AND time > '{self.t_start}' AND time < '{self.t_end}' ''')
        points = result.get_points()
        prices = pd.DataFrame(Util.create_array_from_generator(points), columns=['timestamp', 'price', 'side'])
        print("Prices")
        print(prices.head())
        prices['datetime'] = pd.to_datetime(prices['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True,
                                            errors="coerce")
        prices.loc[prices["datetime"].isna(), "datetime"] = pd.to_datetime(
            prices.loc[prices["datetime"].isna(), "timestamp"], utc=True)
        prices["epoch"] = prices["datetime"].astype(int) // 1000000
        no_data = prices.loc[(prices["epoch"].diff() > (60 * 60 * 1000))]
        print(no_data.head())

    def check_taker_trades_data(self):
        result = self.client.query(f'''SELECT "price","size" FROM "trade" WHERE ("exchange" = '{self.exchange}' AND
                                             symbol = '{self.symbol}') AND time > '{self.t_start}' AND time < '{self.t_end}' ''')
        points = result.get_points()
        prices = pd.DataFrame(Util.create_array_from_generator(points), columns=['timestamp', 'price', 'size'])
        print("Prices")
        print(prices.head())

    def check_funding_data(self):
        result = self.client.query(f'''SELECT "funding" FROM "funding" WHERE ("exchange" = '{self.exchange}' AND
                                             symbol = '{self.symbol}') AND time > '{self.t_start}' AND time < '{self.t_end}' ''')
        points = result.get_points()
        prices = pd.DataFrame(Util.create_array_from_generator(points), columns=['timestamp', 'funding'])
        print("Prices")
        print(prices.head())


class AutomatedSimulation:
    api = None
    symbol = None
    training_hosts = []
    training_hosts_len = 2
    total_controllers = 4
    controllers_state = []

    def __init__(self, symbol, sweep_id, t_start, t_end, name="Estimated PNL with Realized Quanto_profit"):
        print("Automated Simulation Started...")
        self.project = "automation_test"
        self.symbol = symbol
        self.sweep_id = sweep_id
        self.t_start = t_start
        self.t_end = t_end
        self.name = name
        self.wandb_key = os.getenv("WANDB_API_KEY")
        self.wandb_host = os.getenv("WANDB_HOST")
        self.postgres_connection = PostgresConnection()

    def connect_to_wandb(self):
        wandb.login(key=self.wandb_key, host=self.wandb_host)
        self.api = wandb.Api()

    def start(self):
        self.connect_to_wandb()
        self.init_confirmations()
        self.start_training(agents=30)
        self.get_training_results_and_start_controllers()
        self.wait_for_controllers_and_start_confirmations()
        self.merge_results()

    def init_confirmations(self):
        return

    def start_training(self, agents):
        print(f"----- [Step1]: Start training for sweep {self.sweep_id}")
        started = False
        while not started:
            for host in HOSTS:

                if self.is_host_available(host) and self.training_hosts_len > len(self.training_hosts):
                    self.training_hosts.append(host)
                    cmd = f"ssh {host} 'python src/simulations_management/start_sweep.py " \
                          f"--sweep_id={self.sweep_id} --num_agents={agents} --maximum_results={math.ceil(MAXIMUM_RESULTS / self.training_hosts_len)}'"

                    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    # out, err = process.communicate()
                    # print(out.decode('utf-8').strip())
                    print(cmd)
                    started = True
                    print(f"----- [Step1]: Training for sweep {self.sweep_id} started...")
                    self.postgres_connection.update_simulation(sweep_id=self.sweep_id, status="training")
                    self.postgres_connection.update_simulation_host(host=host, attached=False, available=False)
                    if self.training_hosts_len == len(self.training_hosts):
                        started = True
                        break
            sleep(60)

        return

    def get_training_results_and_start_controllers(self):
        print(f"----- [Step2]: Get training results from wandb")
        training_finished = 0
        while training_finished < len(self.training_hosts):
            for host in self.training_hosts:
                if is_training_running(host, self.sweep_id):
                    print(f"----- [Step2]: Training for sweep {self.sweep_id} is still running at host {host}")
                else:
                    training_finished += 1
                if self.training_hosts_len == training_finished:
                    break
            sleep(60)
        print(f"----- [Step2]: Training for sweep {self.sweep_id} finished")
        self.postgres_connection.update_simulation(sweep_id=self.sweep_id, status="running_controllers")
        for index, controller in enumerate(self.controllers_state):
            self.start_local_controller(sweep_id=controller["sweep_id"], index=index)

    @staticmethod
    def setup_and_run(symbol, file, t_start, t_end):
        return

    @staticmethod
    def init_sweep_and_start_container(name, time_from, time_to):
        return

    def start_local_controller(self, sweep_id, index):
        print(f"----- [Step3]: Start local controller {sweep_id}")
        started = False
        while not started:
            for host in HOSTS:
                if self.is_host_available(host):
                    self.controllers_state[index]['host'] = host
                    cmd = f""" ssh {host} 'python src/scripts/wandb_sweeps/controller_specific_combinations.py --sweep_id="{sweep_id}" --source_sweep_id="{self.sweep_id}" --custom_filter="global_filter" --project_name={self.project} ' """
                    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    # out, err = process.communicate()
                    # print(out.decode('utf-8').strip())
                    print(cmd)
                    started = True
                    print(f"----- [Step3]: Controller with sweep {sweep_id} started...")
                    self.postgres_connection.update_simulation_host(host=host, attached=False, available=False)
                    break
            sleep(60)

    def wait_for_controllers_and_start_confirmations(self):
        print(f"----- [Step4]: Wait for controllers")
        for controller in self.controllers_state:
            finished = False
            while not finished:
                finished = not is_controller_running(controller["sweep_id"], controller["host"])
                sleep(60)
            self.postgres_connection.update_simulation_host(host=controller["host"], attached=False, available=True)
        self.postgres_connection.update_simulation(sweep_id=self.sweep_id, status="running_confirmations")

    def start_confirmation(self, sweep_id, host):
        print(f"----- [Step5]: Start confirmation {sweep_id}")
        started = False
        while not started:
            if self.is_host_available(host):
                cmd = f""" ssh {host} 'python src/simulations_management/start_sweep.py --sweep_id {sweep_id} --num_agents 30' """
                # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                # out, err = process.communicate()
                # print(out.decode('utf-8').strip())
                print(cmd)
                started = True
                print(f"----- [Step5]: Confirmation with sweep {sweep_id} started...")
                self.postgres_connection.update_simulation_host(host=host, attached=False, available=False)
                self.postgres_connection.insert_into_sweeps_and_hosts(sweep_id=sweep_id, host=host)
            sleep(60)

    def merge_results(self):
        print(f"----- [Step6]: Wait for confirmations")
        for confirmation in self.controllers_state:
            finished = False
            while not finished:
                finished = not is_training_running(sweep_id=confirmation["sweep_id"], host=confirmation["host"])
                sleep(60)
            self.postgres_connection.update_simulation_host(host=confirmation["host"], attached=False, available=True)
        self.postgres_connection.update_simulation(sweep_id=self.sweep_id, status="merge_results")
        sleep(60)
        print(f"----- [Step6]: Merge results")
        cmd = f"python src/scripts/wandb_sweeps/process_sweep_results/download_wandb_results.py --symbol={self.symbol} --sweep_id_training={self.sweep_id} --sweep_id_confirm={','.join(c['sweep_id'] for c in self.controllers_state)}"
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # out, err = process.communicate()
        print(cmd)

    def is_host_available(self, host):
        attached = True
        available = False
        retries = 0
        while attached and retries < 3:
            sleep(10)
            result = self.postgres_connection.query_simulation_host(host=host)
            attached = result[1]
            available = result[2]
            retries += 1
        if not attached:
            self.postgres_connection.update_simulation_host(host=host, attached=True, available=available)
            # Check if there is any sweep running
            cmd = f"ssh {host} {IS_ANY_SWEEP_RUNNING}"
            # print(cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            # print(out, err)
            print(out.decode('utf-8').strip())
            if "sweep" in out.decode('utf-8').strip():
                print(
                    f"A sweep with id {out.decode('utf-8').strip().split('_')[1]} is currently running at host {host}")
                self.postgres_connection.update_simulation_host(host=host, attached=False, available=False)
                return False

            # Check is if there is any controller running
            cmd = f"ssh {host} {IS_ANY_CONTROLLER_RUNNING}"
            # print(cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            # print(out, err)
            print(out.decode('utf-8').strip())
            if int(out.decode('utf-8').strip()) > 2:
                print(f"A controller is currently running at host {host}")
                self.postgres_connection.update_simulation_host(host=host, attached=False, available=False)
                return False
            return True
        return False


class AutomatedSimulationETHUSDMultiperiod(AutomatedSimulation):
    total_controllers = 4
    controllers_state = []

    @staticmethod
    def setup_and_run(symbol, file, t_start, t_end):

        build_command = f"cd ../../ && docker build -f Dockerfile.automated_simulation -t {image} ."
        process = subprocess.Popen(build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        out = process.communicate()
        if f"naming to docker.io/{image} done" not in (out[0].decode('utf-8').strip()):
            print(out[0].decode('utf-8').strip())
            exit(1)
        print(out[0].decode('utf-8').strip())
        print(f"Building done.")
        AutomatedSimulationETHUSDMultiperiod.init_sweep_and_start_containers(
            name="Estimated PNL with Realized Quanto_profit_P4",
            t_start=t_start,
            t_end=t_end)

    @staticmethod
    def init_sweep_and_start_containers(name, t_start, t_end):
        sweep_configuration = {
            "method": "bayes",
            "metric": {
                "goal": "maximize",
                "name": name
            },
            "parameters": {
                "band_funding_system": {
                    "distribution": "constant",
                    "value": "funding_adjusted_band_swap_spot_with_drop"
                },
                "current_r": {
                    "distribution": "uniform",
                    "max": 3,
                    "min": 0
                },
                "entry_delta_spread": {
                    "distribution": "uniform",
                    "max": 2,
                    "min": 0.1
                },
                "exit_delta_spread": {
                    "distribution": "uniform",
                    "max": 2,
                    "min": 0.3
                },
                "funding_system": {
                    "distribution": "constant",
                    "value": "Quanto_both"
                },
                "high_r": {
                    "distribution": "uniform",
                    "max": 5,
                    "min": 0.5
                },
                "hours_to_stop": {
                    "distribution": "uniform",
                    "max": 96,
                    "min": 8
                },
                "move_bogdan_band": {
                    "distribution": "constant",
                    "value": "No"
                },
                "quanto_threshold": {
                    "distribution": "uniform",
                    "max": 10,
                    "min": 0.5
                },
                "ratio_entry_band_mov": {
                    "distribution": "constant",
                    "value": 1
                },
                "ratio_entry_band_mov_ind": {
                    "distribution": "uniform",
                    "max": 10,
                    "min": 0
                },
                "rolling_time_window_size": {
                    "distribution": "int_uniform",
                    "max": 4000,
                    "min": 500
                },
                "window_size": {
                    "distribution": "uniform",
                    "max": 4000,
                    "min": 500
                },
                "t_end_period": {
                    "distribution": "constant",
                    "value": f"1662930000000~1667944800000~1672794000000~1675040400000~{t_end}"
                },
                "t_start_period": {
                    "distribution": "constant",
                    "value": f"1660510800000~1665262800000~1668474000000~1672794000000~{t_start}"
                }

            },
            "program": "src/scripts/wandb_sweeps/maker_taker_quanto_contracts_multiperiods.py"
        }
        print(json.dumps(sweep_configuration, indent=2))
        wandb.login(key=os.getenv("WANDB_API_KEY"), host=os.getenv("WANDB_HOST"))
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="automation_test")
        print(f"Starting local container for sweep: {sweep_id}")
        sweep_local_container_command = f"docker run -d " \
                                        f"--name automated_simulation_multiperiod_{sweep_id} " \
                                        f"--network staging " \
                                        f"-v /root/.ssh:/root/.ssh " \
                                        f"{image} python -u AutomatedSimulationETHUSDMultiperiod.py  --sweep_id {sweep_id} --t_start={t_start} --t_end={t_end}"
        process = subprocess.Popen(sweep_local_container_command, shell=True)
        result = process.communicate()
        return

    def init_confirmations(self):

        for i in range(self.total_controllers):
            CONTROLLER_CONF["parameters"]["t_end"]["value"] = CONTROLLER_CONFS_TIMESTAMPS[i]["t_end"]
            CONTROLLER_CONF["parameters"]["t_start"]["value"] = CONTROLLER_CONFS_TIMESTAMPS[i]["t_start"]
            sweep_id = wandb.sweep(sweep=CONTROLLER_CONF, project="automation_test")
            self.controllers_state.append({})
            self.controllers_state[i]["sweep_id"] = sweep_id

        self.postgres_connection.insert_simulation(
            creation_date=datetime.now(),
            status="initiated",
            training_sweep_id=self.sweep_id,
            conf_sweep_ids=",".join(c["sweep_id"] for c in self.controllers_state)
        )


class AutomatedSimulationSinglePeriod(AutomatedSimulation):
    total_controllers = 4
    controllers_state = []

    @staticmethod
    def setup_and_run(symbol, file, time_from, time_to):

        build_command = f"cd ../../ && docker build -f Dockerfile.automated_simulation -t {image} ."
        print(build_command)
        process = subprocess.Popen(build_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        out = process.communicate()
        if f"naming to docker.io/{image} done" not in (out[0].decode('utf-8').strip()):
            print(out[0].decode('utf-8').strip())
            exit(1)
        print(out[0].decode('utf-8').strip())
        print(f"Building done.")
        params_df = AutomatedSimulationSinglePeriod.get_parameters_from_csv(filename=file)
        AutomatedSimulationSinglePeriod.init_sweep_and_start_containers(
            name="Estimated PNL with Realized Quanto_profit",
            symbol=symbol,
            params_df=params_df,
            t_start=time_from,
            t_end=time_to)

    @staticmethod
    def get_parameters_from_csv(filename):
        print(f"----- [Step1]: Get parameters from csv file {filename}")
        df = None
        try:
            df = pd.read_csv(filename, usecols=COLS)
            df = df[df["window_size"].notna()]
        except BaseException as e:
            print(f"Error: {e}")
            exit(1)

        return df

    @staticmethod
    def init_sweep_and_start_containers(name, symbol, t_start, t_end, params_df=None):
        for index, row in params_df.iterrows():

            sweep_configuration = {
                "method": "bayes",
                "metric": {
                    "goal": "maximize",
                    "name": name
                },
                "parameters": {
                    "entry_delta_spread": {
                        "distribution": "uniform",
                        "max": 2,
                        "min": 0.1
                    },
                    "exit_delta_spread": {
                        "distribution": "uniform",
                        "max": 2,
                        "min": 0.3
                    },
                    "move_bogdan_band": {
                        "distribution": "constant",
                        "value": "No"
                    },
                    "ratio_entry_band_mov": {
                        "distribution": "constant",
                        "value": 1
                    }
                }
            }

            for col, value in row.items():
                sweep_configuration["parameters"][f"{col}"] = {
                    "distribution": "constant",
                    "value": value
                }
            sweep_configuration["parameters"]["t_end"] = {
                "distribution": "constant",
                "value": t_end
            }
            sweep_configuration["parameters"]["t_start"] = {
                "distribution": "constant",
                "value": t_start
            }
            sweep_configuration["program"] = SYMBOL_PROGRAM[symbol]
            print(json.dumps(sweep_configuration, indent=2))
            wandb.login(key=os.getenv("WANDB_API_KEY"), host=os.getenv("WANDB_HOST"))
            sweep_id = wandb.sweep(sweep=sweep_configuration, project="automation_test")
            # for each row start a local container

            print(f"Starting local container for sweep: {sweep_id}")
            sweep_local_container_command = f"docker run -d " \
                                            f"--name automated_simulation_{sweep_id} " \
                                            f"--network staging " \
                                            f"-v /root/.ssh:/root/.ssh " \
                                            f"{image} python AutomatedSimulationSinglePeriod.py --symbol {symbol} --sweep_id {sweep_id} --t_start={t_start} --t_end={t_end}"
            process = subprocess.Popen(sweep_local_container_command, shell=True)
            result = process.communicate()
        return

    def init_confirmations(self):
        for i in range(self.total_controllers):
            CONTROLLER_CONFS[i]["program"] = SYMBOL_PROGRAM[self.symbol]
            # print(json.dumps(CONTROLLER_CONFS[i]["program"], indent=2))
            sweep_id = wandb.sweep(sweep=CONTROLLER_CONFS[i], project="automation_test")
            self.controllers_state[i]["sweep_id"] = sweep_id

        self.postgres_connection.insert_simulation(
            creation_date=datetime.now(),
            status="initiated",
            training_sweep_id=self.sweep_id,
            conf_sweep_ids=",".join(c["sweep_id"] for c in self.controllers_state)
        )
