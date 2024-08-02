import time
import os
import psycopg2
import pandas as pd
import clickhouse_connect
from typing import List
from dotenv import load_dotenv, find_dotenv
from influxdb import InfluxDBClient, DataFrameClient
from influxdb_client import InfluxDBClient as InfluxDBClientFlux
from requests.exceptions import ConnectionError
from src.common.utils.utils import Util

load_dotenv(find_dotenv())


# Not a connection class but simply a type
class Strategy:
    def __init__(self, spot_name, spot_symbol, swap_name, swap_symbol, description) -> None:
        self.spot_name = spot_name
        self.spot_symbol = spot_symbol
        self.swap_name = swap_name
        self.swap_symbol = swap_symbol
        self.description = description


class InfluxConnection:
    instance = None

    def __init__(self):
        self._staging_client_spotswap = None
        self._staging_client_spotswap_dataframe = None
        self._archival_client_spotswap = None
        self._archival_client_spotswap_dataframe = None
        self._archival_client_secondary_ai = None
        self._staging_client_ai = None
        self._prod_client_spotswap_dataframe = None
        self._prod_client_spotswap = None
        self._prod_client_connection_monitoring_dataframe = None
        self._archival_client_secondary_ai_dataframe = None
        self._staging_client_ai_dataframe = None
        self._staging_flux_client_df = None
        self._local_client_spotswap = None
        self._local_client_spotswap_dataframe = None
        self._prod_flux_client_df = None

    @staticmethod
    def getInstance():
        if InfluxConnection.instance is None:
            InfluxConnection.instance = InfluxConnection()
        return InfluxConnection.instance

    def query_influx(self, client, query, epoch="ms"):
        res = []
        retries = 0
        while True:
            try:
                if "influxdb_client" in str(type(client)):
                    res = client.query_api().query_data_frame(query)
                else:
                    res = client.query(query, epoch=epoch)
            except IndexError:
                print("Retrying...")
                time.sleep(1)
                retries += 1
                continue
            except ConnectionError:
                return []
            except Exception as e:
                print("InfluxDB threw error: ", e)
                pass
            break
        return res

    @property
    def staging_client_spotswap(self):
        if self._staging_client_spotswap is None:
            self._staging_client_spotswap = InfluxDBClient('influxdb.staging.equinoxai.com',
                                                           443,
                                                           os.getenv("DB_USERNAME"),
                                                           os.getenv("DB_PASSWORD"),
                                                           'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._staging_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_spotswap

    @property
    def staging_client_spotswap_dataframe(self):
        if self._staging_client_spotswap_dataframe is None:
            self._staging_client_spotswap_dataframe = DataFrameClient('influxdb.staging.equinoxai.com',
                                                                      443,
                                                                      os.getenv("DB_USERNAME"),
                                                                      os.getenv("DB_PASSWORD"),
                                                                      'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._staging_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_spotswap_dataframe

    @property
    def local_client_spotswap(self):
        if self._local_client_spotswap is None:
            self._local_client_spotswap = InfluxDBClient('localhost',
                                                         8086,
                                                         os.getenv("DB_USERNAME"),
                                                         os.getenv("DB_PASSWORD"),
                                                         'spotswap')
        return self._local_client_spotswap

    @property
    def local_client_spotswap_dataframe(self):
        if self._local_client_spotswap_dataframe is None:
            self._local_client_spotswap_dataframe = DataFrameClient('localhost',
                                                                    8086,
                                                                    os.getenv("DB_USERNAME"),
                                                                    os.getenv("DB_PASSWORD"),
                                                                    'spotswap', retries=1)
        return self._local_client_spotswap_dataframe

    @property
    def prod_client_spotswap_dataframe(self):
        if self._prod_client_spotswap_dataframe is None:
            self._prod_client_spotswap_dataframe = DataFrameClient('influxdb.equinoxai.com',
                                                                   443,
                                                                   os.getenv("DB_USERNAME"),
                                                                   os.getenv("DB_PASSWORD"),
                                                                   'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._prod_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE")
        return self._prod_client_spotswap_dataframe

    @property
    def prod_client_spotswap(self):
        if self._prod_client_spotswap is None:
            self._prod_client_spotswap = InfluxDBClient('influxdb.equinoxai.com',
                                                        443,
                                                        os.getenv("DB_USERNAME"),
                                                        os.getenv("DB_PASSWORD"),
                                                        'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._prod_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE")
        return self._prod_client_spotswap

    @property
    def prod_client_connection_monitoring_dataframe(self):
        if self._prod_client_connection_monitoring_dataframe is None:
            self._prod_client_connection_monitoring_dataframe = DataFrameClient('influxdb.equinoxai.com',
                                                                                443,
                                                                                os.getenv("DB_USERNAME"),
                                                                                os.getenv("DB_PASSWORD"),
                                                                                'connection_monitoring', ssl=True,
                                                                                verify_ssl=True, gzip=True)
            self._prod_client_connection_monitoring_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE")
        return self._prod_client_connection_monitoring_dataframe

    @property
    def archival_client_spotswap(self):
        if self._archival_client_spotswap is None:
            self._archival_client_spotswap = InfluxDBClient('simulations-influxdb.staging.equinoxai.com',
                                                            443,
                                                            os.getenv("DB_USERNAME"),
                                                            os.getenv("DB_PASSWORD"),
                                                            'spotswap', ssl=True, gzip=True)
            self._archival_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_spotswap

    @property
    def archival_client_secondary_ai(self):
        if self._archival_client_secondary_ai is None:
            self._archival_client_secondary_ai = InfluxDBClient('simulations-influxdb.staging.equinoxai.com',
                                                                443,
                                                                os.getenv("DB_USERNAME"),
                                                                os.getenv("DB_PASSWORD"),
                                                                'secondary_ai', ssl=True, verify_ssl=True, gzip=True)
            self._archival_client_secondary_ai._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_secondary_ai

    @property
    def staging_flux_client_df(self):
        if self._staging_flux_client_df is None:
            self._staging_flux_client_df = InfluxDBClientFlux('https://influxdb.staging.equinoxai.com:443',
                                                              f'{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}',
                                                              verify_ssl=True,
                                                              org='-')

            self._staging_flux_client_df.api_client.cookie = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_flux_client_df

    @property
    def prod_flux_client_df(self):
        if self._prod_flux_client_df is None:
            self._prod_flux_client_df = InfluxDBClientFlux('https://influxdb.equinoxai.com:443',
                                                           f'{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}',
                                                           verify_ssl=True,
                                                           org='-')

            self._prod_flux_client_df.api_client.cookie = os.getenv("AUTHELIA_COOKIE")
        return self._prod_flux_client_df

    @property
    def archival_client_secondary_ai_dataframe(self):
        if self._archival_client_secondary_ai_dataframe is None:
            self._archival_client_secondary_ai_dataframe = DataFrameClient('simulations-influxdb.staging.equinoxai.com',
                                                                           443,
                                                                           os.getenv("DB_USERNAME"),
                                                                           os.getenv("DB_PASSWORD"),
                                                                           'secondary_ai', ssl=True, verify_ssl=True,
                                                                           gzip=True)
            self._archival_client_secondary_ai_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_secondary_ai_dataframe

    @property
    def staging_client_ai(self):
        if self._staging_client_ai is None:
            self._staging_client_ai = InfluxDBClient('influxdb.staging.equinoxai.com',
                                                     443,
                                                     os.getenv("DB_USERNAME"),
                                                     os.getenv("DB_PASSWORD"),
                                                     'ai', ssl=True, verify_ssl=True, gzip=True)

            self._staging_client_ai._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_ai

    @property
    def staging_client_ai_dataframe(self):
        if self._staging_client_ai_dataframe is None:
            self._staging_client_ai_dataframe = DataFrameClient('influxdb.staging.equinoxai.com',
                                                                443,
                                                                os.getenv("DB_USERNAME"),
                                                                os.getenv("DB_PASSWORD"),
                                                                'ai', ssl=True, verify_ssl=True, gzip=True)

            self._staging_client_ai_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_ai_dataframe

    @property
    def archival_client_spotswap_dataframe(self):
        if self._archival_client_spotswap_dataframe is None:
            self._archival_client_spotswap_dataframe = DataFrameClient('simulations-influxdb.staging.equinoxai.com',
                                                                       443,
                                                                       os.getenv("DB_USERNAME"),
                                                                       os.getenv("DB_PASSWORD"),
                                                                       'spotswap', ssl=True, gzip=True)
            self._archival_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_spotswap_dataframe


class InfluxMeasurements:
    def __init__(self, single_measurement=None, many_measurements=None):
        self.single_measurement = single_measurement
        self.many_measurements = many_measurements
        self.influx_connection = InfluxConnection.getInstance()
        self.influxclient = {
            "prod": self.influx_connection.prod_client_spotswap
        }

    def get_active_strategies(self):
        query = f'SHOW TAG VALUES FROM {self.single_measurement} WITH KEY = "strategy" WHERE  time > now() - 14d'
        array = self.influx_connection.query_influx(self.influxclient[os.getenv("ENVIRONMENT")], query,
                                                    epoch='ns')
        strategies = pd.DataFrame(array.raw['series'][0]['values'], columns=["name", "Strategies"])
        strategies.drop("name", axis=1, inplace=True)
        strategies = strategies.squeeze()
        return strategies

    def get_field_from_measurement(self, measurement, t0, t1):
        query = f"SELECT * FROM {measurement} WHERE time >= {t0}ms and time <= {t1}ms GROUP BY *"
        array = self.influx_connection.query_influx(self.influxclient[os.getenv("ENVIRONMENT")], query,
                                                    epoch='ns')
        return array


class PostgresConnection:

    def __init__(self) -> None:
        """

        :rtype: object
        """
        self.logger = Util.get_logger('PostgresConnection')
        self.logger.info("Connecting to postgres...")
        self.connection = psycopg2.connect(user=os.getenv("POSTGRES_USERNAME"),
                                           password=os.getenv("POSTGRES_PASSWORD"),
                                           host="pgbouncer",
                                           port=6432)
        try:
            self.connection_ta = psycopg2.connect(user=os.getenv("POSTGRES_USERNAME"),
                                                  password=os.getenv("POSTGRES_PASSWORD"),
                                                  host="ta_db",
                                                  port=5432)
        except:
            print(
                "psycopg2.OperationalError: could not translate host name 'ta_db' to address: Name or service not known");
        self.logger.info("Connected!")

    def close_connection(self):
        self.connection.close()
        self.connection_ta.close()

        self.logger.info("Connection closed")

    def connect(self):
        self.connection = psycopg2.connect(user=os.getenv("POSTGRES_USERNAME"),
                                           password=os.getenv("POSTGRES_PASSWORD"),
                                           host="pgbouncer",
                                           port=6432)

    def query_strategies(self) -> List[Strategy]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT spot_name, spot_symbol, swap_name, swap_symbol, description from strategy")
        records = cursor.fetchall()
        cursor.close()

        strategies = []
        for strategy in records:
            prefix = "hybrid"
            description = strategy[4]
            if "cross" in description:
                prefix = "cross"

            strategies.append(
                Strategy(strategy[0], f"{prefix}_{strategy[1]}", strategy[2], f"{prefix}_{strategy[3]}", description))
        return strategies

    def query_account_per_strategy(self, query):
        # @TODO maybe improve
        cursor = self.connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        self.logger.info(f"Queried account {records}")
        return records[0]

    def query_daily_transfers(self, query):
        # @TODO maybe improve
        cursor = self.connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        self.logger.info(f"Queried transfers. type {type(records)}")
        return records

    def insert_ai_experiment_result(self, model_name, experiment_name, price_movement, max_accuracy_ask,
                                    max_accuracy_bid, total_predictions_ask, total_predictions_bid,
                                    number_high_accuracy_predictions_ask, number_high_accuracy_predictions_bid, from_,
                                    to_):
        cursor = self.connection.cursor()
        cursor.execute('''INSERT INTO "ai_experiment_results" 
          ("model_name", "experiment_name", "price_movement", "max_accuracy_ask", "max_accuracy_bid", "total_predictions_ask", "total_predictions_bid", "number_high_accuracy_predictions_ask", "number_high_accuracy_predictions_bid", "from", "to") VALUES 
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                       (model_name, experiment_name, price_movement, max_accuracy_ask, max_accuracy_bid,
                        total_predictions_ask, total_predictions_bid, number_high_accuracy_predictions_ask,
                        number_high_accuracy_predictions_bid, from_, to_))
        self.connection.commit()
        cursor.close()

    def insert_ai_experiment_optimal_regions(self,
                                             model_name,
                                             experiment_name,
                                             price_movement,
                                             starting_threshold_ask,
                                             ending_threshold_ask,
                                             starting_price_movement_ask,
                                             ending_price_movement_ask,
                                             max_average_ask,
                                             number_of_predictions_in_region_ask,
                                             starting_threshold_ask_at_least_n_thresholds,
                                             ending_threshold_ask_at_least_n_thresholds,
                                             starting_price_movement_ask_at_least_n_thresholds,
                                             ending_price_movement_ask_at_least_n_thresholds,
                                             max_average_ask_at_least_n_thresholds,
                                             number_of_predictions_ask_at_least_n_thresholds,
                                             starting_threshold_bid,
                                             ending_threshold_bid,
                                             starting_price_movement_bid,
                                             ending_price_movement_bid,
                                             max_average_bid,
                                             number_of_predictions_in_region_bid,
                                             starting_threshold_bid_at_least_n_thresholds,
                                             ending_threshold_bid_at_least_n_thresholds,
                                             starting_price_movement_bid_at_least_n_thresholds,
                                             ending_price_movement_bid_at_least_n_thresholds,
                                             max_average_bid_at_least_n_thresholds,
                                             number_of_predictions_bid_at_least_n_thresholds,
                                             from_,
                                             to_):
        cursor = self.connection.cursor()
        number_of_predictions_in_region_ask = int(number_of_predictions_in_region_ask)
        number_of_predictions_in_region_bid = int(number_of_predictions_in_region_bid)

        cursor.execute('''INSERT INTO "ai_experiment_optimal_regions"
        ("model_name", "experiment_name", "price_movement", "starting_threshold_ask", "ending_threshold_ask", "starting_price_movement_ask", "ending_price_movement_ask",
         "starting_threshold_ask_at_least_two_thresholds", "ending_threshold_ask_at_least_two_thresholds", "starting_price_movement_ask_at_least_two_thresholds", "ending_price_movement_ask_at_least_two_thresholds",
          "max_average_ask", "number_of_predictions_in_region_ask", "max_average_ask_at_least_two_thresholds", "number_of_predictions_ask_at_least_two_tresholds",
         "starting_threshold_bid", "ending_threshold_bid", "starting_price_movement_bid", "ending_price_movement_bid", 
         "starting_threshold_bid_at_least_two_thresholds", "ending_threshold_bid_at_least_two_thresholds", "starting_price_movement_bid_at_least_two_thresholds", "ending_price_movement_bid_at_least_two_thresholds", 
         "max_average_bid", "number_of_predictions_in_region_bid", "max_average_bid_at_least_two_thresholds", "number_of_predictions_bid_at_least_two_tresholds", "from", "to") VALUES 
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                       (model_name, experiment_name, price_movement, starting_threshold_ask, ending_threshold_ask,
                        starting_price_movement_ask, ending_price_movement_ask,
                        starting_threshold_ask_at_least_n_thresholds, ending_threshold_ask_at_least_n_thresholds,
                        starting_price_movement_ask_at_least_n_thresholds,
                        ending_price_movement_ask_at_least_n_thresholds,
                        max_average_ask, number_of_predictions_in_region_ask, max_average_ask_at_least_n_thresholds,
                        number_of_predictions_ask_at_least_n_thresholds,
                        starting_threshold_bid, ending_threshold_bid, starting_price_movement_bid,
                        ending_price_movement_bid,
                        starting_threshold_bid_at_least_n_thresholds, ending_threshold_bid_at_least_n_thresholds,
                        starting_price_movement_bid_at_least_n_thresholds,
                        ending_price_movement_bid_at_least_n_thresholds,
                        max_average_bid, number_of_predictions_in_region_bid, max_average_bid_at_least_n_thresholds,
                        number_of_predictions_bid_at_least_n_thresholds, from_, to_))
        self.connection.commit()
        cursor.close()

    def query_ai_experiment_results(self):
        cursor = self.connection.cursor()
        cursor.execute(
            '''SELECT "model_name", "experiment_name", "price_movement", "from", "to" from ai_experiment_results''')
        records = cursor.fetchall()
        cursor.close()
        model_names = []
        experiment_names = []
        price_movements = []
        froms = []
        tos = []
        for experiment_result in records:
            model_name = experiment_result[0]
            model_names.append(model_name)
            experiment_name = experiment_result[1]
            experiment_names.append(experiment_name)
            price_movement = experiment_result[2]
            price_movements.append(price_movement)
            from_ = experiment_result[3]
            froms.append(from_)
            to_ = experiment_result[4]
            tos.append(to_)
        cursor.close()
        return model_names, experiment_names, price_movements, froms, tos

    def get_mtm_profit_for_ETH(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z'):
        '''
        Input
        t0: starting time
        t1: ending time

        Output
        strategies: strategy name
        family: family name of strategies
        transfers: the transfers in the selected period
        profits: profit excluding transfers t
        mtm_profits: Mark to Market profit = profit - transfers
        '''
        cursor = self.connection.cursor()
        # t0=datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
        # t1=datetime.utcfromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')

        base_query = f'''with prof as (select description, sum(increase)as mtm, min(fvalue) as fvalue From
                        (select description, time, value -lag(value) over (partition by description order by time) as increase, first_value(value) over (partition by description order by time) as fvalue
                        From(
                        select underlying - coalesce(price * amount,0) as value, time as time, description from (

                        select  
                        s.description as description,
                        m.snapshot_time as time,
                        max(c.price) as price,
                        sum(c.amount) as amount,
                        round(sum(m.underlying_value * m.price), 0) as underlying

                        from strategy as s
                        left JOIN mark_to_market_snapshots as m ON (m.account_name=s.swap_account OR m.account_name=s.spot_account)
                        left join mark_to_market_snapshot_crypto_loan as c on m.snapshot_time = c.snapshot_time and m.account_name = c.account_name

                        where  m.snapshot_time BETWEEN '{t0}' AND '{t1}' and s.description LIKE 'deribit_ETH%' and ((m.exchange = 'BitMEX' and m.currency = 'XBT') or m.exchange = 'Deribit') 
                        group by s.description, m.snapshot_time
                        ) as x) as foo)as fooo
                        group by description)
                        SELECT prof.description as strategy,
                        CASE
                          WHEN transfers IS NULL THEN prof.mtm
                        ELSE
                           ROUND((-transfers+prof.mtm)::numeric,2)
                        END as MtM_profit

                        From(
                        SELECT
                        sum(amount*price) as transfers,description
                        FROM account_transfers
                        LEFT JOIN strategy
                        ON (account_transfers.strategy=description)
                        WHERE timestamp BETWEEN '{t0}' AND '{t1}'
                        AND strategy LIKE '%deribit_ETH%'
                        AND (NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
                        GROUP BY strategy.description
                        ) as x
                        RIGHT JOIN prof ON x.description=prof.description'''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        strategies = []
        mtm_profits = []
        percentual_profit = []
        for profit in records:
            strategy = profit[0]
            strategies.append(strategy)
            mtm_profit = profit[1]
            mtm_profits.append(mtm_profit)

        return strategies, mtm_profits

    def get_mtm_profit_for_XBTUSD(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z'):
        '''
       Input
       t0: starting time
       t1: ending time

       Output
       strategies: strategy name
       family: family name of strategies
       transfers: the transfers in the selected period
       profits: profit excluding transfers t
       mtm_profits: Mark to Market profit = profit - transfers
       '''
        cursor = self.connection.cursor()
        # t0=datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
        # t1=datetime.utcfromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')

        base_query = f'''with prof as (select description, sum(increase)as mtm From
                        (select description, time, value -lag(value) over (partition by description order by time) as increase
                        From(
                        select underlying - coalesce(price * amount,0) as value, time as time, description from (

                        select  
                        s.description as description,
                        m.snapshot_time as time,
                        max(c.price) as price,
                        sum(c.amount) as amount,
                        round(sum(m.underlying_value * m.price), 0) as underlying


                        from strategy as s
                        left JOIN mark_to_market_snapshots as m ON (m.account_name=s.swap_account OR m.account_name=s.spot_account)
                        left join mark_to_market_snapshot_crypto_loan as c on m.snapshot_time = c.snapshot_time and m.account_name = c.account_name

                        where  m.snapshot_time BETWEEN '{t0}' AND '{t1}' and s.description LIKE 'deribit_XBTUSD%' 
                        group by s.description, m.snapshot_time
                        ) as x) as foo)as fooo
                        group by description)
                        SELECT prof.description as strategy,
                        CASE
                          WHEN transfers IS NULL THEN prof.mtm
                        ELSE
                           ROUND((-transfers+prof.mtm)::numeric,2)
                        END as MtM_profit

                        From(
                        SELECT
                        sum(amount*price) as transfers,description
                        FROM account_transfers
                        LEFT JOIN strategy
                        ON (account_transfers.strategy=description)
                        WHERE timestamp BETWEEN '{t0}' AND '{t1}'
                        AND strategy LIKE '%deribit_XBTUSD_maker_perpetual%'
                        AND (NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
                        GROUP BY strategy.description
                        ) as x
                        RIGHT JOIN prof ON x.description=prof.description'''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        strategies = []
        mtm_profits = []
        for profit in records:
            strategy = profit[0]
            strategies.append(strategy)
            mtm_profit = profit[1]
            mtm_profits.append(mtm_profit)

        return strategies, mtm_profits

    def get_perc_mtm_profit(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z',
                            strategy_family='deribit_ETH'):
        '''
        Input
        t0: starting time
        t1: ending time

        Output
        strategies: strategy name
        family: family name of strategies
        transfers: the transfers in the selected period
        profits: profit excluding transfers t
        mtm_profits: Mark to Market profit = profit - transfers
        '''
        cursor = self.connection.cursor()
        # t0=datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
        # t1=datetime.utcfromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')

        if strategy_family == 'deribit_ETH':
            base_query = f'''with prof as (select description, sum(increase)as mtm, min(fvalue) as fvalue 
                                        From
                                        (select description, time, value -lag(value) over (partition by description order by time) as increase, first_value(value) over (partition by description order by time) as fvalue
                                        From(
                                        select underlying - coalesce(price * amount,0) as value, time as time, description from (

                                        select  
                                        s.description as description,
                                        m.snapshot_time as time,
                                        max(c.price) as price,
                                        sum(c.amount) as amount,
                                        round(sum(m.underlying_value * m.price), 0) as underlying

                                        from strategy as s
                                        left JOIN mark_to_market_snapshots as m ON (m.account_name=s.swap_account OR m.account_name=s.spot_account)
                                        left join mark_to_market_snapshot_crypto_loan as c on m.snapshot_time = c.snapshot_time and m.account_name = c.account_name

                                        where  m.snapshot_time BETWEEN '{t0}' AND '{t1}' and s.description LIKE '{strategy_family}%' 
                                        and ((m.exchange = 'BitMEX' and m.currency = 'XBT') or m.exchange = 'Deribit') 
                                        group by s.description, m.snapshot_time
                                        ) as x) as foo)as fooo
                                        group by description)
                                        SELECT prof.description as strategy,
                                        CASE
                                        WHEN transfers IS NULL THEN prof.mtm
                                        ELSE
                                        ROUND((-transfers+prof.mtm)::numeric,2)
                                        END as MtM_profit, 
                                        fvalue,
                                        prof.mtm,
                                        CASE
                                        WHEN transfers IS NULL THEN ROUND((prof.mtm / fvalue)::numeric, 4)*100
                                        ELSE
                                        ROUND(((-transfers+prof.mtm)/fvalue)::numeric,4)*100
                                        END as MtM_profit_perc,
                                        transfers

                                        From(
                                        SELECT
                                        sum(amount*price) as transfers,description
                                        FROM account_transfers
                                        LEFT JOIN strategy
                                        ON (account_transfers.strategy=description)
                                        WHERE timestamp BETWEEN '{t0}' AND '{t1}'
                                        AND strategy LIKE '%{strategy_family}%'
                                        AND (NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
                                        GROUP BY strategy.description
                                        ) as x
                                        RIGHT JOIN prof ON x.description=prof.description
                                        ORDER BY MtM_profit_perc DESC
                                        '''
        else:
            base_query = f'''with prof as (select description, sum(increase)as mtm, min(fvalue) as fvalue 
                            From
                            (select description, time, value -lag(value) over (partition by description order by time) as increase, first_value(value) over (partition by description order by time) as fvalue
                            From(
                            select underlying - coalesce(price * amount,0) as value, time as time, description from (

                            select  
                            s.description as description,
                            m.snapshot_time as time,
                            max(c.price) as price,
                            sum(c.amount) as amount,
                            round(sum(m.underlying_value * m.price), 0) as underlying

                            from strategy as s
                            left JOIN mark_to_market_snapshots as m ON (m.account_name=s.swap_account OR m.account_name=s.spot_account)
                            left join mark_to_market_snapshot_crypto_loan as c on m.snapshot_time = c.snapshot_time and m.account_name = c.account_name

                            where  m.snapshot_time BETWEEN '{t0}' AND '{t1}' and s.description LIKE '{strategy_family}%'
                            group by s.description, m.snapshot_time
                            ) as x) as foo)as fooo
                            group by description)
                            SELECT prof.description as strategy,
                            CASE
                            WHEN transfers IS NULL THEN prof.mtm
                            ELSE
                            ROUND((-transfers+prof.mtm)::numeric,2)
                            END as MtM_profit, 
                            fvalue,
                            prof.mtm,
                            CASE
                            WHEN transfers IS NULL THEN ROUND((prof.mtm / fvalue)::numeric, 4)*100
                            ELSE
                            ROUND(((-transfers+prof.mtm)/fvalue)::numeric,4)*100
                            END as MtM_profit_perc,
                            transfers

                            From(
                            SELECT
                            sum(amount*price) as transfers,description
                            FROM account_transfers
                            LEFT JOIN strategy
                            ON (account_transfers.strategy=description)
                            WHERE timestamp BETWEEN '{t0}' AND '{t1}'
                            AND strategy LIKE '%{strategy_family}%'
                            AND (NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
                            GROUP BY strategy.description
                            ) as x
                            RIGHT JOIN prof ON x.description=prof.description
                            ORDER BY MtM_profit_perc DESC
                            '''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        strategies = []
        mtm_profits = []
        fvalues = []
        transfers = []
        percentual_profit = []
        fdiff = []
        for profit in records:
            strategy = profit[0]
            strategies.append(strategy)
            mtm_profit = profit[1]
            mtm_profits.append(mtm_profit)
            fvalues.append(profit[2])
            fdiff.append(profit[3])
            percentual_profit.append(profit[4])
            transfers.append(profit[5])

        return strategies, mtm_profits, fvalues, fdiff, percentual_profit, transfers

    def get_transfers_XBTUSD(self, t0, t1):
        base_query = f'''SELECT
         sum(amount*price) as transfers,strategy
        FROM account_transfers
        WHERE
            strategy LIKE '%deribit_XBTUSD%' 
            AND(NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
            AND timestamp BETWEEN '{t0}' AND '{t1}' 
        Group by strategy
        ORDER BY 1'''
        cursor = self.connection.cursor()
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()

        strategies = []
        transfers = []
        for ix in records:
            transfers.append(ix[0])
            strategies.append(ix[1])

        return strategies, transfers

    def mapping_strategy_account(self, family):
        if family == 'deribit_xbtusd':
            base_query = f''' SELECT * FROM "strategy" WHERE "description" LIKE '%deribit_XBTUSD_maker_perpetual%' '''
        elif family == 'deribit_eth':
            base_query = f''' SELECT * FROM "strategy" WHERE "description" LIKE 'deribit_ETH_perpetual_bitmex_ETHUSD%' '''
        else:
            return

        cursor = self.connection.cursor()
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()

        identity = []
        spot_name = []
        spot_symbol = []
        swap_name = []
        swap_symbol = []
        identifier = []
        description = []
        window_size = []
        entry_delta_spread = []
        exit_delta_spread = []
        spot_account = []
        swap_account = []
        for ix in records:
            identity.append(ix[0])
            spot_name.append(ix[1])
            spot_symbol.append(ix[2])
            swap_name.append(ix[3])
            swap_symbol.append(ix[4])
            identifier.append(ix[5])
            description.append(ix[6])
            window_size.append(ix[7])
            entry_delta_spread.append(ix[8])
            exit_delta_spread.append(ix[9])
            spot_account.append(ix[10])
            swap_account.append(ix[11])

        return identity, spot_name, spot_symbol, swap_name, swap_symbol, identifier, description, window_size, \
            entry_delta_spread, exit_delta_spread, spot_account, swap_account

    def get_exchange_from_strategy_name(self, strategy):
        base_query = f''' SELECT * FROM "strategy" WHERE "description" ='{strategy}' '''

        cursor = self.connection.cursor()
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()

        identity = []
        spot_name = []
        spot_symbol = []
        swap_name = []
        swap_symbol = []
        identifier = []
        description = []
        window_size = []
        entry_delta_spread = []
        exit_delta_spread = []
        spot_account = []
        swap_account = []
        for ix in records:
            identity.append(ix[0])
            spot_name.append(ix[1])
            spot_symbol.append(ix[2])
            swap_name.append(ix[3])
            swap_symbol.append(ix[4])
            identifier.append(ix[5])
            description.append(ix[6])
            window_size.append(ix[7])
            entry_delta_spread.append(ix[8])
            exit_delta_spread.append(ix[9])
            spot_account.append(ix[10])
            swap_account.append(ix[11])

        return spot_name, spot_symbol, swap_name, swap_symbol, spot_account, swap_account

    def get_maker_taker_traded_volume_from_ta_db(self, t0, t1):
        base_query = f'''SELECT "account_name", split_part(split_part("account_name", '@', 1), ']',2) as swap_account, sum("Taker_Volume")/sum("Maker_Volume") as ratio_taker_maker_total_vol, sum("Taker_Volume") as taker_total_vol, account_name
                        FROM(
                          SELECT  
                            CASE 
                               WHEN "fee" > 0 THEN sum(volume * price) 
                            END as "Taker_Volume" ,
                            CASE 
                               WHEN "fee" < 0 THEN sum(volume * price) 
                            END as "Maker_Volume",
                          account_name, timestamp
                          FROM trades
                          WHERE exchange = 'bitmex' AND "type" = 'trade' 
                          AND "timestamp" BETWEEN '{t0}' AND '{t1}' 
                          GROUP BY timestamp, fee, account_name) as x
                        GROUP BY account_name '''

        cursor = self.connection_ta.cursor()
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()

        account_name = []
        ratio_taker_maker_total_vol = []
        taker_total_vol = []
        swap_account = []
        for ix in records:
            account_name.append(ix[0])
            swap_account.append(ix[1])
            ratio_taker_maker_total_vol.append(ix[2])
            taker_total_vol.append(ix[3])

        return account_name, swap_account, ratio_taker_maker_total_vol, taker_total_vol

    def get_transfers_ETH(self, t0, t1):
        base_query = f'''SELECT
         sum(amount*price) as transfers,strategy
        FROM account_transfers
        WHERE
            strategy LIKE '%deribit_ETH_perpetual_bitmex_ETHUSD%' 
            AND CAST("timestamp" AS text) NOT LIKE '%2022-03-23%'
            AND(NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
            AND timestamp BETWEEN '{t0}' AND '{t1}' 
        Group by strategy
        ORDER BY 1'''
        cursor = self.connection.cursor()
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()

        strategies = []
        transfers = []
        for ix in records:
            transfers.append(ix[0])
            strategies.append(ix[1])

        return strategies, transfers

    def get_percentage_band_params(self):
        '''
        Output:
        params:   a list of parameters in json format for the percentage band
        strategies:  the name of the strategy that has this set of params
        '''

        cursor = self.connection.cursor()
        base_query = '''SELECT parameters ,strategy,enabled FROM bands WHERE  (type='percentage_bogdan_band') '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        params = []
        strategies = []
        enabled = []
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])

        return params, strategies, enabled

    def get_bogdan_band_params(self):
        '''
        Output:
        params:   a list of parameters in json format for the percentage band
        strategies:  the name of the strategy that has this set of params
        '''

        cursor = self.connection.cursor()
        base_query = '''SELECT parameters ,strategy,enabled FROM bands WHERE  (type='bogdan_bands') '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        params = []
        strategies = []
        enabled = []
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])

        return params, strategies, enabled

    def get_band_changed_params(self, t1, type):
        '''
        Output:
        params:   a list of parameters in json format for the percentage band
        strategies:  the name of the strategy that has this set of params
        '''

        cursor = self.connection.cursor()
        base_query = f'''SELECT to_parameters ,strategy,enabled,"timestamp" FROM bands_events WHERE  (type='{type}') 
        AND "timestamp" <= '{t1}' ORDER BY "timestamp" DESC '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        params = []
        strategies = []
        enabled = []
        timestamp = []
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])
            timestamp.append(idx[3])

        return params, strategies, enabled, timestamp

    def get_snapshots(self, t0, t1, family='deribit_eth'):
        '''
        Input
        t0: starting time
        t1: ending time
        family: strategy family name (acceptable names: deribit_eth, deribit_xbtusd


        Output
        Time: the timestamp of the snapshot
        strategies: strategy names
        values: the value of the snapshot (meaning the value of the displayed strategy)
        '''

        cursor = self.connection.cursor()
        if family == 'deribit_eth':
            base_query = f'''SELECT sum(mark_to_market_snapshots.underlying_value * mark_to_market_snapshots.price) as value, 
            mark_to_market_snapshots.snapshot_time as "Time",
            description
            FROM strategy INNER JOIN mark_to_market_snapshots ON (account_name=swap_account OR account_name=spot_account)
            WHERE description LIKE 'deribit_ETH_perpetual_bitmex_ETHUSD%' AND (symbol = 'ETH-PERPETUAL' OR symbol = 'XBTUSD') 
            AND snapshot_time BETWEEN '{t0}' AND '{t1}'
            GROUP BY snapshot_time,description
            ORDER BY snapshot_time
                    '''
        else:
            base_query = f'''SELECT sum(mark_to_market_snapshots.underlying_value * mark_to_market_snapshots.price) as value, 
            mark_to_market_snapshots.snapshot_time as "Time",
            description
            FROM strategy INNER JOIN mark_to_market_snapshots ON (account_name=swap_account OR account_name=spot_account)
            WHERE description LIKE 'deribit_XBTUSD%' AND snapshot_time BETWEEN '{t0}' AND '{t1}'
            GROUP BY snapshot_time,description
            ORDER BY snapshot_time
                               '''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        values = []
        Time = []
        strategies = []
        for profit in records:
            strategy = profit[2]
            strategies.append(strategy)
            timestamp = profit[1]
            Time.append(timestamp)
            value = profit[0]
            values.append(value)

        return Time, strategies, values

    def get_transfers(self, t0, t1, family='deribit_eth'):
        '''
        Input
        t0: starting time
        t1: ending time
        family: strategy family name (acceptable names: deribit_eth, deribit_xbtusd


        Output
        Time: the timestamp of the snapshot
        strategies: strategy names
        transfers: transfers made over this period
        '''

        cursor = self.connection.cursor()
        if family == 'deribit_eth':
            base_query = f'''SELECT amount * price as transfer,timestamp as "time",strategy
            FROM account_transfers
            WHERE strategy LIKE 'deribit_ETH_perpetual_bitmex_ETHUSD%' 
            AND CAST("timestamp" AS text) NOT LIKE '%2022-03-23%'
            AND(NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
            AND timestamp BETWEEN '{t0}' AND '{t1}'
                            '''
        else:
            base_query = f'''SELECT amount * price as transfer,timestamp as "time",strategy
                        FROM account_transfers
                        WHERE strategy LIKE 'deribit_XBTUSD_maker_perpetual%' 
                        AND(NOT("account" LIKE '%[Deribit] EquinoxAIBV%' AND "account" NOT LIKE '%ETH' AND "account" NOT LIKE '%BTC'))
                        AND timestamp BETWEEN '{t0}' AND '{t1}'
                                        '''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        transfers = []
        Time = []
        strategies = []
        for profit in records:
            strategy = profit[2]
            strategies.append(strategy)
            timestamp = profit[1]
            Time.append(timestamp)
            transfer = profit[0]
            transfers.append(transfer)

        return Time, strategies, transfers

    def get_past_microparams_values_postgres(self, t0, t1, family='deribit_eth'):
        '''
        Input
        t0: starting time
        t1: ending time
        family: strategy family name (acceptable names: deribit_eth, deribit_xbtusd


        Output
        Time: the timestamp of the snapshot
        strategies: strategy names
        transfers: transfers made over this period
        '''
        cursor = self.connection.cursor()
        if family == 'deribit_eth':
            base_query = f'''SELECT type ,timestamp as "time",strategy,from_parameters, to_parameters, enabled
            FROM bands_events
            WHERE strategy LIKE 'deribit_ETH_perpetual_bitmex_ETHUSD%' 
            AND timestamp BETWEEN '{t0}' AND '{t1}'
                            '''
        else:
            base_query = f'''SELECT type ,timestamp as "time",strategy,from_parameters, to_parameters, enabled
            FROM bands_events
            WHERE strategy LIKE '%XBTUSD%' 
            AND timestamp BETWEEN '{t0}' AND '{t1}'
                            '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        types = []
        timestamps = []
        strategies = []
        from_params = []
        to_params = []
        status = []
        for idx in records:
            types.append(idx[0])
            timestamps.append(idx[1])
            strategies.append(idx[2])
            from_params.append(idx[3])
            to_params.append(idx[4])
            status.append(idx[5])

        return types, timestamps, strategies, from_params, to_params, status

    def get_strategy_microparams(self, family):
        cursor = self.connection.cursor()

        if family == 'deribit_eth':
            varstrat = 'deribit_ETH%'
        else:
            varstrat = 'deribit_XBTUSD%'

        base_query = f'''WITH jsonvalue AS(SELECT parameters,strategy,type,enabled
         FROM bands),
        bogdan_bands AS(
        SELECT
             strategy,
             parameters->>'window_size' AS window_size,
             parameters->>'entry_delta_spread' AS entry_delta_spread,
             parameters->>'exit_delta_spread' AS exit_delta_spread
                               FROM jsonvalue
                               WHERE type='bogdan_bands'AND enabled='true'),

        quanto_profit AS(
        SELECT
             strategy,
             CASE WHEN enabled='true' THEN 'ON'
             END state,
             parameters->>'trail_value' AS QB_trail_value
                               FROM jsonvalue
                               WHERE type='quanto_profit'AND enabled='true'),
        fast_bands AS(
        SELECT
             strategy,
             CASE WHEN enabled='true' THEN 'ON'
             END state,
             parameters->>'order' AS FB_orders,
             parameters->>'target' AS FB_target
                               FROM jsonvalue
                               WHERE type='fast_band' AND enabled='true'),
        percentage_bogdan_bands AS(
        SELECT
             strategy,
             CASE WHEN enabled='true' THEN 'ON'
             END state,
             parameters->>'lookback' AS PB_lookback,
             parameters->>'minimum_target' AS PB_minimum_target,
             parameters->>'target_percentage_entry' AS PB_target_percentage_entry,
             parameters->>'target_percentage_exit' AS PB_target_percentage_exit,
             parameters->>'recomputation_time' AS PB_recomputation_time,
             parameters->>'exit_opportunity_source' AS PB_exit_opportunity_source,
             parameters->>'entry_opportunity_source' AS PB_entry_opportunity_source
                               FROM jsonvalue
                               WHERE type='percentage_bogdan_band' AND enabled='true')
        SELECT bogdan_bands.strategy,
        CASE 
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='spread' AND quanto_profit.state IS NULL AND percentage_bogdan_bands.state IS NULL THEN 'FB_S'
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='opportunities' AND quanto_profit.state IS NULL AND percentage_bogdan_bands.state IS NULL THEN 'FB_O'
           WHEN fast_bands.state IS NULL AND quanto_profit.state='ON' AND percentage_bogdan_bands.state IS NULL THEN 'QB'
           WHEN fast_bands.state IS NULL AND quanto_profit.state IS NULL AND percentage_bogdan_bands.state='ON' THEN 'PB'
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='spread'AND quanto_profit.state='ON' AND percentage_bogdan_bands.state IS NULL THEN 'FB_S,QB'
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='opportunities'AND quanto_profit.state='ON' AND percentage_bogdan_bands.state IS NULL THEN 'FB_O,QB'
           WHEN fast_bands.state='ON'AND fast_bands.FB_target='spread' AND quanto_profit.state IS NULL AND percentage_bogdan_bands.state='ON' THEN 'FB_S,PB'
           WHEN fast_bands.state='ON'AND fast_bands.FB_target='opportunities' AND quanto_profit.state IS NULL AND percentage_bogdan_bands.state='ON' THEN 'FB_O,PB'
           WHEN fast_bands.state IS NULL AND quanto_profit.state='ON' AND percentage_bogdan_bands.state='ON' THEN 'QB,PB'
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='spread'AND quanto_profit.state='ON' AND percentage_bogdan_bands.state='ON' THEN 'FB_S,QB,PB'
           WHEN fast_bands.state='ON' AND fast_bands.FB_target='opportunities'AND quanto_profit.state='ON' AND percentage_bogdan_bands.state='ON' THEN 'FB_O,QB,PB'
        END status,
        window_size,entry_delta_spread,exit_delta_spread,
        QB_trail_value,FB_orders,FB_target,PB_lookback,PB_minimum_target,PB_target_percentage_entry,PB_target_percentage_exit,PB_recomputation_time,
        PB_exit_opportunity_source,PB_entry_opportunity_source
        FROM bogdan_bands
        FULL JOIN fast_bands ON bogdan_bands.strategy=fast_bands.strategy
        FULL JOIN quanto_profit ON bogdan_bands.strategy=quanto_profit.strategy
        FULL JOIN percentage_bogdan_bands ON bogdan_bands.strategy=percentage_bogdan_bands.strategy
        WHERE (bogdan_bands.strategy like '{varstrat}' ) 
        GROUP BY bogdan_bands.strategy,status,window_size,entry_delta_spread,exit_delta_spread,
        QB_trail_value,FB_orders,FB_target,PB_lookback,PB_minimum_target,PB_target_percentage_entry,PB_target_percentage_exit,PB_recomputation_time,
        PB_exit_opportunity_source,PB_entry_opportunity_source'''

        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        statuses = []
        strategies = []
        window_sizes = [];
        entry_delta_spreads = [];
        exit_delta_spreads = []
        QB_trail_values = [];
        FB_orders_s = [];
        FB_targets = [];
        PB_lookbacks = [];
        PB_minimum_targets = [];
        PB_target_percentages_entry = [];
        PB_target_percentages_exit = [];
        PB_recomputation_times = [];
        PB_exit_opportunity_sources = [];
        PB_entry_opportunity_sources = [];
        for idx in records:
            status = idx[1]
            statuses.append(status)
            strategy = idx[0]
            strategies.append(strategy)
            window_size = idx[2]
            window_sizes.append(window_size)
            entry_delta_spread = idx[3]
            entry_delta_spreads.append(entry_delta_spread)
            exit_delta_spread = idx[4]
            exit_delta_spreads.append(exit_delta_spread)
            QB_trail_value = idx[5]
            QB_trail_values.append(QB_trail_value)
            FB_orders = idx[6]
            FB_orders_s.append(FB_orders)
            FB_target = idx[7]
            FB_targets.append(FB_target)
            PB_lookback = idx[8]
            PB_lookbacks.append(PB_lookback)
            PB_minimum_target = idx[9]
            PB_minimum_targets.append(PB_minimum_target)
            PB_target_percentage = idx[10]
            PB_target_percentages_entry.append(PB_target_percentage)
            PB_target_percentages_exit.append(idx[11])
            PB_recomputation_time = idx[12]
            PB_recomputation_times.append(PB_recomputation_time)
            PB_exit_opportunity_source = idx[13]
            PB_exit_opportunity_sources.append(PB_exit_opportunity_source)
            PB_entry_opportunity_source = idx[14]
            PB_entry_opportunity_sources.append(PB_entry_opportunity_source)

        return strategies, statuses, window_sizes, entry_delta_spreads, exit_delta_spreads, \
            QB_trail_values, FB_orders_s, FB_targets, PB_lookbacks, \
            PB_minimum_targets, PB_target_percentages_entry, PB_target_percentages_exit, PB_recomputation_times, \
            PB_exit_opportunity_sources, PB_entry_opportunity_sources

    def query_ai_experiment_optimal_regions(self):
        cursor = self.connection.cursor()
        cursor.execute(
            '''SELECT "model_name", "experiment_name", "price_movement", "from", "to" from ai_experiment_optimal_regions''')
        records = cursor.fetchall()
        cursor.close()
        model_names = []
        experiment_names = []
        price_movements = []
        froms = []
        tos = []
        for experiment_result in records:
            model_name = experiment_result[0]
            model_names.append(model_name)
            experiment_name = experiment_result[1]
            experiment_names.append(experiment_name)
            price_movement = experiment_result[2]
            price_movements.append(price_movement)
            from_ = experiment_result[3]
            froms.append(from_)
            to_ = experiment_result[4]
            tos.append(to_)
        cursor.close()
        return model_names, experiment_names, price_movements, froms, tos

    def update(self, string):
        cursor = self.connection.cursor()
        cursor.execute(string)
        self.connection.commit()
        cursor.close()

    def query_simulation_host(self, host):
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT hostname, attached, available from simulation_host WHERE hostname = '{host}'")
        records = cursor.fetchall()
        cursor.close()
        return records[0]

    def update_simulation_host(self, host, attached, available):
        cursor = self.connection.cursor()
        cursor.execute('UPDATE simulation_host SET attached = %s, available = %s  WHERE hostname = %s',
                       (attached, available, host))
        self.connection.commit()
        cursor.close()

    def insert_simulation(self, creation_date, status, training_sweep_id, conf_sweep_ids):
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT INTO simulation ("creation_date", "status", "training_sweep_id", "conf_sweep_ids") VALUES (%s, %s, %s, %s)',
            (creation_date, status, training_sweep_id, conf_sweep_ids))
        self.connection.commit()
        cursor.close()

    def update_simulation(self, sweep_id, status):
        cursor = self.connection.cursor()
        cursor.execute('UPDATE simulation SET status = %s WHERE training_sweep_id = %s', (status, sweep_id))
        self.connection.commit()
        cursor.close()

    def insert_into_sweeps_and_hosts(self, sweep_id, host):
        cursor = self.connection.cursor()
        cursor.execute('INSERT INTO sweeps_and_hosts ("sweep_id", "host") VALUES (%s, %s)', (sweep_id, host))
        self.connection.commit()
        cursor.close()


class ClickhouseConnection:
    def __init__(self) -> None:
        """

        :rtype: object
        """
        self.logger = Util.get_logger('ClickhouseConnection')
        self.logger.info("Connecting to Clickhouse...")
        self.client = clickhouse_connect.get_client(host=os.getenv("CLICKHOUSE_HOST"),
                                                    port=8123,
                                                    database="v2data",
                                                    username=os.getenv("CLICKHOUSE_USERNAME"),
                                                    password=os.getenv("CLICKHOUSE_PASSWORD"))
        self.logger.info("Connected!")

    def close_connection(self):
        self.client.close()

        self.logger.info("Connection closed")

    def connect(self):
        self.client = clickhouse_connect.get_client(host=os.getenv("CLICKHOUSE_HOST"),
                                                    port=8123,
                                                    database="v2data",
                                                    username=os.getenv("CLICKHOUSE_USERNAME"),
                                                    password=os.getenv("CLICKHOUSE_PASSWORD"))

    def get_funding_rate(self):
        result = self.client.query('SELECT * FROM funding_rate LIMIT 10')
        rows = result.result_rows
        print(result.result_rows)

