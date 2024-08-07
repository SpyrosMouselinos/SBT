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


# This is a strategy for getting information about a particular resource. It should return a list of resource names
class Strategy:
    def __init__(self, spot_name, spot_symbol, swap_name, swap_symbol, description) -> None:
        """
         @brief Initializes the object with the values provided. This is the place where you can store your data
         @param spot_name The name of the spot in the database. It is used to display the name of the spot in the table.
         @param spot_symbol The symbol of the spot in the database. It is used to display the symbol of the swapth in the table.
         @param swap_name The name of the swap in the database. It is used to display the name of the swap in the table.
         @param swap_symbol The symbol of the swap in the database. It is used to display the symbol of the swapth in the table.
         @param description The description of the swapth in the table.
         @return The instance of the class that was initialized with the values provided
        """
        self.spot_name = spot_name
        self.spot_symbol = spot_symbol
        self.swap_name = swap_name
        self.swap_symbol = swap_symbol
        self.description = description


# Creates connection to InfluxDB and returns it. This is a no - op if there is no connection
class InfluxConnection:
    instance = None

    def __init__(self):
        """
         @brief Initialize the class. This is called by __init__ and should not be called directly
        """
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
    # Returns the instance of the plugin. This is a singleton so you can call it multiple times
    def getInstance():
        """
         @brief Get the instance of InfluxConnection. This is a singleton so we don't have to worry about this in the __init__. py
         @return an instance of : class : ` InfluxConnection
        """
        # This method is used to create a new instance of the InfluxConnection class.
        if InfluxConnection.instance is None:
            InfluxConnection.instance = InfluxConnection()
        return InfluxConnection.instance

    # Return a handle to the influx data store. This is called by the InfluxDB server
    def query_influx(self, client, query, epoch="ms"):
        """
         @brief Queries InfluxDB and returns data. This is a wrapper around the query method that handles retry in case of connection errors
         @param client An instance of the InfluxDB client
         @param query The query to be executed. Can be a string or a DataFrame
         @param epoch The epoch of the data to be returned. Defaults to ms
         @return A list of data returned by the query or an empty list
        """
        res = []
        retries = 0
        # This function is a blocking function that will try to get the data frame from InfluxDB.
        while True:
            try:
                # This function is used to get the data frame from the Elasticsearch API.
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
    # spotswap_staging_client_spotswap is a function that takes a staging client and returns a pointer to it
    def staging_client_spotswap(self):
        """
         @brief Get or create InfluxDB client for spotswap. This is used to make requests to SpotSWAP and should not be used in production.
         @return An : class : ` InfluxDBClient `
        """
        # This method will create a new InfluxDBClient and set the headers to the StagingClient.
        if self._staging_client_spotswap is None:
            self._staging_client_spotswap = InfluxDBClient('influxdb.staging.equinoxai.com',
                                                           443,
                                                           os.getenv("DB_USERNAME"),
                                                           os.getenv("DB_PASSWORD"),
                                                           'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._staging_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_spotswap

    @property
    # Staging client spotswap data. This is a dataframe with one row per user
    def staging_client_spotswap_dataframe(self):
        """
         @brief A : class : ` DataFrameClient ` for Spotswap. It is cached so repeated calls are fast.
         @return A : class : ` DataFrameClient ` for Spot
        """
        # This method will create a pandas DataFrameClient to store the spotswap data.
        if self._staging_client_spotswap_dataframe is None:
            self._staging_client_spotswap_dataframe = DataFrameClient('influxdb.staging.equinoxai.com',
                                                                      443,
                                                                      os.getenv("DB_USERNAME"),
                                                                      os.getenv("DB_PASSWORD"),
                                                                      'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._staging_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_spotswap_dataframe

    @property
    # spotswap_t local_client_spotswap ( char * s ) Returns the client '
    def local_client_spotswap(self):
        """
         @brief Get or create a spotswap InfluxDB client. This is a lazy property so it's only used for testing.
         @return An instance of : class : ` InfluxDBClient
        """
        # Use this method to create a new InfluxDBClient instance.
        if self._local_client_spotswap is None:
            self._local_client_spotswap = InfluxDBClient('localhost',
                                                         8086,
                                                         os.getenv("DB_USERNAME"),
                                                         os.getenv("DB_PASSWORD"),
                                                         'spotswap')
        return self._local_client_spotswap

    @property
    # spotswap_dataframe is a dataframe that contains the data for all spotswaps that are local
    def local_client_spotswap_dataframe(self):
        """
         @brief A DataFrameClient for Spotswap. If you don't want to use this in your tests it's recommended to use : meth : ` ~test. setUp ` instead.
         @return A : class : ` pandas. DataFrameClient `
        """
        # A pandas DataFrameClient that will be used to store the spotswap data.
        if self._local_client_spotswap_dataframe is None:
            self._local_client_spotswap_dataframe = DataFrameClient('localhost',
                                                                    8086,
                                                                    os.getenv("DB_USERNAME"),
                                                                    os.getenv("DB_PASSWORD"),
                                                                    'spotswap', retries=1)
        return self._local_client_spotswap_dataframe

    @property
    # This function is used to read data from spotswap. The data is read in a dataframe and returned
    def prod_client_spotswap_dataframe(self):
        """
         @brief A pandas dataframe with data from spotswap. This is used to make requests to the spotswap server.
         @return A pandas dataframe with data from spotswap. Note that the headers are set to AUTHELIA_COOKIE
        """
        # This method is used to create a pandas DataFrameClient to hold the spotswap data.
        if self._prod_client_spotswap_dataframe is None:
            self._prod_client_spotswap_dataframe = DataFrameClient('influxdb.equinoxai.com',
                                                                   443,
                                                                   os.getenv("DB_USERNAME"),
                                                                   os.getenv("DB_PASSWORD"),
                                                                   'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._prod_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE")
        return self._prod_client_spotswap_dataframe

    @property
    # spotswap_client is a wrapper around production_client_spotswap that can be used to interact with the client
    def prod_client_spotswap(self):
        """
         @brief InfluxDB client for spotswap. This is used to connect to the production server.
         @return An instance of : class : ` InfluxDBClient
        """
        # Set up the InfluxDBClient for the SpotSwap server.
        if self._prod_client_spotswap is None:
            self._prod_client_spotswap = InfluxDBClient('influxdb.equinoxai.com',
                                                        443,
                                                        os.getenv("DB_USERNAME"),
                                                        os.getenv("DB_PASSWORD"),
                                                        'spotswap', ssl=True, verify_ssl=True, gzip=True)
            self._prod_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE")
        return self._prod_client_spotswap

    @property
    # This function is used to read data from prod client connection monitoring dataframe and return it as a pandas Dataframe
    def prod_client_connection_monitoring_dataframe(self):
        """
         @brief A DataFrameClient with connection monitoring information. It is cached so repeated calls are fast.
         @return A : class : ` DataFrameClient ` with connection monitoring information
        """
        # This method is used to create a dataframe client connection monitoring dataframe.
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
    # > archival_client_spotswap This function is called when the client wants to connect to SpotSwap
    def archival_client_spotswap(self):
        """
         @brief InfluxDB client for archival. This is used to get data from Spotswap.
         @return An : class : ` InfluxDBClient ` object
        """
        # This method is used to set up the InfluxDBClient. _archival_client_spotswap. _headers Cookie is set to the cookie name and the server s session name.
        if self._archival_client_spotswap is None:
            self._archival_client_spotswap = InfluxDBClient('simulations-influxdb.staging.equinoxai.com',
                                                            443,
                                                            os.getenv("DB_USERNAME"),
                                                            os.getenv("DB_PASSWORD"),
                                                            'spotswap', ssl=True, gzip=True)
            self._archival_client_spotswap._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_spotswap

    @property
    # Returns the secondary AI of the archival_client_secondary_ai
    def archival_client_secondary_ai(self):
        """
         @brief Get or set the secondary_ai InfluxDB client. This is used to download archives from equinoxai.
         @return an : class : ` InfluxDBClient `
        """
        # This method is used to create a new InfluxDBClient for the secondary ai server.
        if self._archival_client_secondary_ai is None:
            self._archival_client_secondary_ai = InfluxDBClient('simulations-influxdb.staging.equinoxai.com',
                                                                443,
                                                                os.getenv("DB_USERNAME"),
                                                                os.getenv("DB_PASSWORD"),
                                                                'secondary_ai', ssl=True, verify_ssl=True, gzip=True)
            self._archival_client_secondary_ai._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_secondary_ai

    @property
    # Staging Flux Client DF for Staging Flux Dataflow. This is a wrapper around the client_df that can be used to create a staging_flux_df
    def staging_flux_client_df(self):
        """
         @brief Flux that connects to Staging database. This is useful for testing and testing the InfluxDB API.
         @return A : class : ` pandas. DataFrame ` with the data to
        """
        # This is a convenience method to create a new InfluxDBClientFlux object that will be used to create a StagingFlux object for the staging.
        if self._staging_flux_client_df is None:
            self._staging_flux_client_df = InfluxDBClientFlux('https://influxdb.staging.equinoxai.com:443',
                                                              f'{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}',
                                                              verify_ssl=True,
                                                              org='-')

            self._staging_flux_client_df.api_client.cookie = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_flux_client_df

    @property
    # This is a wrapper for prod_flux_client_df which will be used for testing
    def prod_flux_client_df(self):
        """
         @brief Get InfluxDBClientFlux for production use. Caches and returns a reference to the object so it can be used multiple times without recalculating the object.
         @return A : class : ` InfluxDBClientFlux `
        """
        # InfluxDBClientFlux for InfluxDBClientFlux.
        if self._prod_flux_client_df is None:
            self._prod_flux_client_df = InfluxDBClientFlux('https://influxdb.equinoxai.com:443',
                                                           f'{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}',
                                                           verify_ssl=True,
                                                           org='-')

            self._prod_flux_client_df.api_client.cookie = os.getenv("AUTHELIA_COOKIE")
        return self._prod_flux_client_df

    @property
    # archival_client_secondary_ai_dataframe : secondary AI dataframe for client
    def archival_client_secondary_ai_dataframe(self):
        """
         @brief Archival client for secondary_ai. Dataframe is cached for performance reasons.
         @return pandas. DataFrame with data from equinoxai
        """
        # This method is used to create a pandas DataFrameClient for the secondary ai server.
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
    # Staging client AI ( staging_client_ai ) This function is called when the user presses the button to download a file
    def staging_client_ai(self):
        """
         @brief Get or create an InfluxDB client for staging. This is used to make requests to AI's staging API.
         @return A Staging client with HTTP cookies set and gzip
        """
        # This method will create a new InfluxDBClient and set the cookie headers to the data stored in the staging server.
        if self._staging_client_ai is None:
            self._staging_client_ai = InfluxDBClient('influxdb.staging.equinoxai.com',
                                                     443,
                                                     os.getenv("DB_USERNAME"),
                                                     os.getenv("DB_PASSWORD"),
                                                     'ai', ssl=True, verify_ssl=True, gzip=True)

            self._staging_client_ai._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_ai

    @property
    # Staging client AI dataframe. This dataframe is used to generate a list of client AI's for testing
    def staging_client_ai_dataframe(self):
        """
         @brief A DataframeClient for communicating with Asterisk's staging server.
         @return A : class : ` pandas. DataFrameClient ` for communicating with Asterisk's staging server
        """
        # This method will create a pandas DataFrameClient for the staging server.
        if self._staging_client_ai_dataframe is None:
            self._staging_client_ai_dataframe = DataFrameClient('influxdb.staging.equinoxai.com',
                                                                443,
                                                                os.getenv("DB_USERNAME"),
                                                                os.getenv("DB_PASSWORD"),
                                                                'ai', ssl=True, verify_ssl=True, gzip=True)

            self._staging_client_ai_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._staging_client_ai_dataframe

    @property
    # Archival client spotswap data frame. This is a dataframe with two columns : archival_client_spotswap_id : ID of the swap that was used to train the client. The first column is the name of the archival client and the second column is the time at which the swap was used
    def archival_client_spotswap_dataframe(self):
        """
         @brief Establish a connection to Spotswap and return a : class : ` DataFrameClient `.
         @return : class : ` DataFrameClient ` connected to Spot
        """
        # This method will create a pandas DataFrameClient for the archival client spotswap.
        if self._archival_client_spotswap_dataframe is None:
            self._archival_client_spotswap_dataframe = DataFrameClient('simulations-influxdb.staging.equinoxai.com',
                                                                       443,
                                                                       os.getenv("DB_USERNAME"),
                                                                       os.getenv("DB_PASSWORD"),
                                                                       'spotswap', ssl=True, gzip=True)
            self._archival_client_spotswap_dataframe._headers["Cookie"] = os.getenv("AUTHELIA_COOKIE_STAGING")
        return self._archival_client_spotswap_dataframe


# This function returns measurements in flux. The measurements are returned as InfluxMeasurements
class InfluxMeasurements:
    def __init__(self, single_measurement=None, many_measurements=None):
        """
         @brief Initialize the class. This is the constructor. You can pass parameters to this method
         @param single_measurement True if you want to receive single measurements
         @param many_measurements True if you want to receive many
        """
        self.single_measurement = single_measurement
        self.many_measurements = many_measurements
        self.influx_connection = InfluxConnection.getInstance()
        self.influxclient = {
            "prod": self.influx_connection.prod_client_spotswap
        }

    # Returns a list of active strategies. This is useful for debugging and to check if there are any strategies that are active
    def get_active_strategies(self):
        """
         @brief Get strategies that are active in the past 14 days. This is used to determine which strategies have been used for a measurement
         @return A dataframe with the names of the
        """
        query = f'SHOW TAG VALUES FROM {self.single_measurement} WITH KEY = "strategy" WHERE  time > now() - 14d'
        array = self.influx_connection.query_influx(self.influxclient[os.getenv("ENVIRONMENT")], query,
                                                    epoch='ns')
        strategies = pd.DataFrame(array.raw['series'][0]['values'], columns=["name", "Strategies"])
        strategies.drop("name", axis=1, inplace=True)
        strategies = strategies.squeeze()
        return strategies

    # Returns the field from the measurement. This is a wrapper around get_field_from_measurement
    def get_field_from_measurement(self, measurement, t0, t1):
        """
         @brief Get field values from a measurement. This is a helper function for get_field_from_measurement_data.
         @param measurement name of the measurement to query. Must be a string of comma separated values
         @param t0 start time of the query in milliseconds since 1970 - 01 - 01 00 : 00 : 00 UTC
         @param t1 end time of the query in milliseconds since 1970 - 01 - 01 00 : 00 : 00 UTC
         @return array of field values in chronological order. Each element is a 2 - tuple ( field_name value
        """
        query = f"SELECT * FROM {measurement} WHERE time >= {t0}ms and time <= {t1}ms GROUP BY *"
        array = self.influx_connection.query_influx(self.influxclient[os.getenv("ENVIRONMENT")], query,
                                                    epoch='ns')
        return array


# Creates a PostgresConnection for use with PostgreSQL. This is a no - op if there is no connection
class PostgresConnection:

    def __init__(self) -> None:
        """
         @brief Connect to Postgres and set logger. This is called by __init__ and should not be called directly
        """
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

    # Closes the connection to the MySQL server. This is a no - op if there is no connection
    def close_connection(self):
        """
         @brief Close connection to database and TA. @ In None @ Out close_connection bool True if
        """
        self.connection.close()
        self.connection_ta.close()

        self.logger.info("Connection closed")

    # Connect to the server. This is a no - op if the server is already connected
    def connect(self):
        """
         @brief Connect to Postgres and set self. connection to the psycopg2
        """
        self.connection = psycopg2.connect(user=os.getenv("POSTGRES_USERNAME"),
                                           password=os.getenv("POSTGRES_PASSWORD"),
                                           host="pgbouncer",
                                           port=6432)

    # Returns a list of query strategies that can be used to query the database. This is the list of strategies that are available for the current connection
    def query_strategies(self) -> List[Strategy]:
        """
         @brief Query strategies from database. This is a list of strategies that can be used to generate an OHLCV file.
         @return A list of : class : ` Strategy ` objects
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT spot_name, spot_symbol, swap_name, swap_symbol, description from strategy")
        records = cursor.fetchall()
        cursor.close()

        strategies = []
        # Add a strategy to the list of strategies.
        for strategy in records:
            prefix = "hybrid"
            description = strategy[4]
            # Add prefix to the description if cross is present.
            if "cross" in description:
                prefix = "cross"

            strategies.append(
                Strategy(strategy[0], f"{prefix}_{strategy[1]}", strategy[2], f"{prefix}_{strategy[3]}", description))
        return strategies

    # Query account per strategy. This is a low - level function that should be used in place of query_account_per_strategy
    def query_account_per_strategy(self, query):
        """
         @brief Query account per strategy. This is a helper method for query_account_per_strategy
         @param query SQL query to be executed
         @return first record of query or None if no records were
        """
        # @TODO maybe improve
        cursor = self.connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        self.logger.info(f"Queried account {records}")
        return records[0]

    # Query daily transfers. This is a low - level function and should not be called directly
    def query_daily_transfers(self, query):
        """
         @brief Queries the database for daily transfers. This is a low - level method to be used by subclasses
         @param query SQL query to be executed
         @return list of records returned by the query ( empty if none
        """
        # @TODO maybe improve
        cursor = self.connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        self.logger.info(f"Queried transfers. type {type(records)}")
        return records

    # Insert AI experiment result into database. This is a wrapper around the SQLAlchemy insert_ai_experiment_result
    def insert_ai_experiment_result(self, model_name, experiment_name, price_movement, max_accuracy_ask,
                                    max_accuracy_bid, total_predictions_ask, total_predictions_bid,
                                    number_high_accuracy_predictions_ask, number_high_accuracy_predictions_bid, from_,
                                    to_):
        """
        @brief Insert ai_experiment_results row into database. This method is called by AI_Experiment to insert data into ai_experiment_results table.
        @param model_name ( str ) name of model that is going to be used for evaluation
        @param experiment_name ( str ) name of experiment that is going to be used for evaluation
        @param price_movement ( float ) price of movement
        @param max_accuracy_ask ( int ) maximum accuracy that can be ask for predictions ( 0 - 100 ). In this case it is set to 0. This can be used to determine how much to bid
        @param max_accuracy_bid ( int ) maximum accuracy that can be bid for predictions ( 0 - 100 ). In this case it is set to 0.
        @param total_predictions_ask ( float ) total number of predictions that are going to be asked for
        @param total_predictions_bid ( float ) total number of predictions that are going to bid for predictions
        @param number_high_accuracy_predictions_ask ( int
        @param number_high_accuracy_predictions_bid
        @param from_
        @param to_
        """
        cursor = self.connection.cursor()
        cursor.execute('''INSERT INTO "ai_experiment_results" 
          ("model_name", "experiment_name", "price_movement", "max_accuracy_ask", "max_accuracy_bid", "total_predictions_ask", "total_predictions_bid", "number_high_accuracy_predictions_ask", "number_high_accuracy_predictions_bid", "from", "to") VALUES 
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                       (model_name, experiment_name, price_movement, max_accuracy_ask, max_accuracy_bid,
                        total_predictions_ask, total_predictions_bid, number_high_accuracy_predictions_ask,
                        number_high_accuracy_predictions_bid, from_, to_))
        self.connection.commit()
        cursor.close()

    # Insert AI experiment optimal regions into the data. This is a wrapper around : func : ` insert_ai_experiment_optimal_regions `
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
        """
        @brief Insert optimal AI regions into the database. This is a wrapper for INSERT_OPTIMAL_REGIONS
        @param model_name Name of the model to be used in the insert.
        @param experiment_name Name of the experiment to be used in the insert.
        @param price_movement Price movement of the model to be used in the insert.
        @param starting_threshold_ask Starting threshold at the beginning of the ask.
        @param ending_threshold_ask
        @param starting_price_movement_ask Ending price movement of the ask.
        @param ending_price_movement_ask
        @param max_average_ask
        @param number_of_predictions_in_region_ask
        @param starting_threshold_ask_at_least_n_thresholds
        @param ending_threshold_ask_at_least_n_thresholds
        @param starting_price_movement_ask_at_least_n_thresholds
        @param ending_price_movement_ask_at_least_n_thresholds
        @param max_average_ask_at_least_n_thresholds
        @param number_of_predictions_ask_at_least_n_thresholds
        @param starting_threshold_bid
        @param ending_threshold_bid
        @param starting_price_movement_bid
        @param ending_price_movement_bid
        @param max_average_bid
        @param number_of_predictions_in_region_bid
        @param starting_threshold_bid_at_least_n_thresholds
        @param ending_threshold_bid_at_least_n_thresholds
        @param starting_price_movement_bid_at_least_n_thresholds
        @param ending_price_movement_bid_at_least_n_thresholds
        @param max_average_bid_at_least_n_thresholds
        @param number_of_predictions_bid_at_least_n_thresholds
        @param from_
        @param to_
        """
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

    # Queries AI experiments and returns a list of model names and price movements
    def query_ai_experiment_results(self):
        """
         @brief Query AI experiment results to retrieve model names experiment names price movements from and to dates.
         @return ( list ) List of model names. ( list ) List of experiment names
        """
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
        # This function is used to generate the experiment results.
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

    # Return profit for Ethernet MTM. This is a wrapper for mtm_profit_for_ETH
    def get_mtm_profit_for_ETH(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z'):
        """
         @brief Get MtM profit for Eth time period. This method is based on : func : ` get_mtm_profit ` but with different time range
         @param t0 starting time in YYYY - MM - DD HH : MM : SS format
         @param t1 ending time in YYYY - MM - DD HH : MM : SS format
         @return Mark to Market profit in UTC time zone
        """
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
        # Add strategy and mtm_profits to the list of records
        for profit in records:
            strategy = profit[0]
            strategies.append(strategy)
            mtm_profit = profit[1]
            mtm_profits.append(mtm_profit)

        return strategies, mtm_profits

    # Return mtm profit for XBTUSD. This is a wrapper for get_mtm_profit_for_XBTUSD
    def get_mtm_profit_for_XBTUSD(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z'):
        """
         @brief Get Mtm Profit for XBTUSD. This method is based on : func : ` get_mtm_profit ` but with different time ranges
         @param t0 starting time in YYYY - MM - DD format
         @param t1 ending time in YYYY - MM - DD format
         @return Mark to Market profit as a : class : ` pandas. Series
        """
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
        # Add strategy and mtm_profits to the list of records
        for profit in records:
            strategy = profit[0]
            strategies.append(strategy)
            mtm_profit = profit[1]
            mtm_profits.append(mtm_profit)

        return strategies, mtm_profits

    # Returns profit value for MTM - TIM. This function is deprecated
    def get_perc_mtm_profit(self, t0='2022-02-28T11:13:02.384Z', t1='2022-03-30T10:13:02.384Z',
                            strategy_family='deribit_ETH'):

        """
            @brief Returns the profit of the mark - to - market data between two times
            @param t0 starting time in YYYYMMDDThh : mm : ss format
            @param t1 ending time in YYYYMMDDThh : mm : ss format
            @param strategy_family name of strategies : deribit_ETH
            @return tuple of ( profits mtm_profit
        """

        cursor = self.connection.cursor()
        # t0=datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
        # t1=datetime.utcfromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')

        # A strategy family is deribit_ETH or deribit_ETH.
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
        # Add the strategy mtm_profits and mtm_profits to the records.
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

    # Returns a list of transfer objects that can be used to transfer XBTUSD
    def get_transfers_XBTUSD(self, t0, t1):
        """
         @brief Get strategies that transfer XBTUSD. This is a list of strategies that have been transferred between t0 and t1
         @param t0 timestamp of first trade in seconds
         @param t1 timestamp of last trade in seconds ( exclusive )
         @return a tuple of two lists : 1 ) a list of strategies 2 ) a
        """
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
        # Add transfers and strategies to the list of records
        for ix in records:
            transfers.append(ix[0])
            strategies.append(ix[1])

        return strategies, transfers

    # Mapping strategy account for mapping_strategy_account This is a class property and can be used to access the mapping strategy
    def mapping_strategy_account(self, family):
        """
         @brief Get information about the perpetual account of a mapping strategy. This is a function to be used in order to get information about the perpetual account of a mapping strategy
         @param family name of the deribit family
         @return dictionary with key / value pairs : identity : identity of the mapping
        """
        # deribit_xbtusd deribit_eth deribit_xbtusd deribit_eth deribit_xbtusd deribit_eth deribit_eth_perpetual_bitmex_ETH_perpetual_bitmex_ETH_perpetual_bitmex_ETH_perpetual_bitmex_ETHUSD or deribit_eth_eth_bitmex_ETH_ETH_perpetual_bitmex_ETH_eth_bitmex_ETH_eth_bitmex_ETH_eth_bitmex_eth_bitmex_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH_ETH
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
        # appends the records to the identity and the information of the records
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

    # Returns the name of the exchange strategy to use for this exchange. This is a convenience function that can be used to get the name of the exchange strategy that is being used
    def get_exchange_from_strategy_name(self, strategy):
        """
         @brief Get information about exchange from strategy name. This method is used to get information about the exchange given a strategy name.
         @param strategy The name of the strategy. For example'BIRT '
         @return A tuple containing the identity the spot name the swap name the swap symbol the description
        """
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
        # appends the records to the identity and the information of the records
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

    # Get maker traded volume from TA DB. This is used to check if there is a taker traded volume
    def get_maker_taker_traded_volume_from_ta_db(self, t0, t1):
        """
         @brief Get maker traded volume from TA database. This is used to calculate the volume of taker traded swaps in order to get the total volumetric volume of swaps that have been traded to the account that is responsible for the trade
         @param t0 timestamp of the start of the trade
         @param t1 timestamp of the end of the trade ( inclusive )
         @return tuple of ( swap_account volume ratio_taker_maker_volume taker_total_vol
        """
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
        # Add all the records to the table.
        for ix in records:
            account_name.append(ix[0])
            swap_account.append(ix[1])
            ratio_taker_maker_total_vol.append(ix[2])
            taker_total_vol.append(ix[3])

        return account_name, swap_account, ratio_taker_maker_total_vol, taker_total_vol

    # Returns a list of transfer objects that can be used to transfer Ethernet traffic
    def get_transfers_ETH(self, t0, t1):
        """
         @brief Get ETH perpetual bitmex transfers between t0 and t1
         @param t0 timestamp of first transfer in seconds
         @param t1 timestamp of last transfer in seconds ( inclusive
        """
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
        # Add transfers and strategies to the list of records
        for ix in records:
            transfers.append(ix[0])
            strategies.append(ix[1])

        return strategies, transfers

    # This function returns a dictionary that maps band names to percentage values. The keys are band names and the values are lists of band
    def get_percentage_band_params(self):
        """
         @brief Get the parameters for the percentage band strategies. This is a list of parameters that are used to generate the percentages
         @return A tuple of the list of parameters the list of strategies and the list of
        """
        cursor = self.connection.cursor()
        base_query = '''SELECT parameters ,strategy,enabled FROM bands WHERE  (type='percentage_bogdan_band') '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        params = []
        strategies = []
        enabled = []
        # Add all parameters strategies and enabled to the records
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])

        return params, strategies, enabled

    # Returns dictionary of band parameters. Keys are band names values are lists of ( band_name band_value )
    def get_bollinger_band_params(self):
        """
         @brief Get bollinger band parameters. Note: For some reason some guy named them Bogdan, probably after his name.
         @return A tuple of three lists : 1. a list of parameters in json format 2. a list of strategies that have this set of
        """
        cursor = self.connection.cursor()
        base_query = '''SELECT parameters ,strategy,enabled FROM bands WHERE  (type='bogdan_bands') '''
        cursor.execute(base_query)
        records = cursor.fetchall()
        cursor.close()
        params = []
        strategies = []
        enabled = []
        # Add all parameters strategies and enabled to the records
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])

        return params, strategies, enabled

    # Returns a dictionary of band parameters that have changed since the last call to get_band_changed_params
    def get_band_changed_params(self, t1, type):
        """
         @brief Get the parameters that have changed since t1. This is used to determine if a band is in the process of being changed
         @param t1 the timestamp before which the change is made
         @param type the type of change ( percentage_band or not_band )
         @return a tuple of ( params strategies enabled timestamp ) where params is a list of parameters to be passed to the
        """
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
        # Add all the records to the records list.
        for idx in records:
            parameters = idx[0]
            params.append(parameters)
            strategies.append(idx[1])
            enabled.append(idx[2])
            timestamp.append(idx[3])

        return params, strategies, enabled, timestamp

    # Returns a function that takes a snapshot id and returns an array of snapshots. The array is sorted by snapshot id
    def get_snapshots(self, t0, t1, family='deribit_eth'):
        """
         @brief Get snapshots between two times. This is a generator that yields tuples of ( strategy name value )
         @param t0 the starting time of the snapshots
         @param t1 the ending time of the snapshots ( inclusive )
         @param family the strategy family ( deribit_eth deribit_xbtusd )
         @return a generator of tuples of ( strategy name value )
        """
        cursor = self.connection.cursor()
        # Returns the total amount of time snapshots for the given family.
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
        # Add strategy timestamp and value to the records.
        for profit in records:
            strategy = profit[2]
            strategies.append(strategy)
            timestamp = profit[1]
            Time.append(timestamp)
            value = profit[0]
            values.append(value)

        return Time, strategies, values

    # Returns a list of Transfer objects that can be used to transfer data from an open Farm. This is the interface to the transfer manager
    def get_transfers(self, t0, t1, family='deribit_eth'):
        """
         @brief Get transfers made over this period. This is a generator that yields tuples of ( transfer timestamp strategy )
         @param t0 the starting timestamp of the snapshot
         @param t1 the ending timestamp of the snapshot ( inclusive )
         @param family the strategy family to get the data for ( deribit_eth deribit_xbtusd
        """
        cursor = self.connection.cursor()
        # deribit_eth or deribit_eth_perpetual_bitmex_ETHUSD_maker_perpetual. If family is deribit_eth then the query will be used to determine the price of the transfer.
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
        # Add strategy timestamp and transfer to the list of records
        for profit in records:
            strategy = profit[2]
            strategies.append(strategy)
            timestamp = profit[1]
            Time.append(timestamp)
            transfer = profit[0]
            transfers.append(transfer)

        return Time, strategies, transfers

    # Returns a list of tuples ( microparams_value time_elapsed ). This is useful for testing
    def get_past_microparams_values_postgres(self, t0, t1, family='deribit_eth'):
        """
         @brief Get past microparams values from postgres. This is a wrapper around self. connection. query_with_cursor
         @param t0 starting timestamp for snapshots to be retrieved
         @param t1 ending timestamp for snapshots to be retrieved ( inclusive )
         @param family strategy family ( deribit_eth xbtusd )
         @return dict with timestamps as keys and values as values or None if no snapshots
        """
        cursor = self.connection.cursor()
        # deribit_eth or deribit_eth_perpetual_bitmex_ETH_perpetual_bitmex_ETH or deribit_eth_perpetual_bitmex_ETHUSD.
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
        # Add all records to the list of records.
        for idx in records:
            types.append(idx[0])
            timestamps.append(idx[1])
            strategies.append(idx[2])
            from_params.append(idx[3])
            to_params.append(idx[4])
            status.append(idx[5])

        return types, timestamps, strategies, from_params, to_params, status

    # Returns microparams used to generate queries. This is a tuple of ( query_strategy microparams )
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

    # Query AI to find optimal regions. This is a wrapper around query_ai_experiment_optimal_regions
    def query_ai_experiment_optimal_regions(self):
        """
         @brief Query AI experiment optimal regions. This is a list of model names experiment names price movements from and to positions.
         @return A tuple containing 4 lists of strings : ( model_name experiment_name price_movements froms tos
        """
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
        # This function is used to generate the experiment results.
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

    # Updates the database. This is called after a change in the database or when an object is added
    def update(self, string):
        """
         @brief Update the database with data. This is a convenience method for performing updates on the database
         @param string Data to insert into
        """
        cursor = self.connection.cursor()
        cursor.execute(string)
        self.connection.commit()
        cursor.close()

    # Returns true if there is a host to connect to. This is a wrapper around query_simulation_host
    def query_simulation_host(self, host):
        """
         @brief Query simulation_host table to get information about a host. This is used for debugging and to check if there is an attached / availablity relationship between host and simulation host.
         @param host Hostname or IP address of simulation host. Example :'192. 168. 1. 1 '
         @return A tuple of ( hostname attached available ). Hostname is the host name attached is the host is attached to the simulation
        """
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT hostname, attached, available from simulation_host WHERE hostname = '{host}'")
        records = cursor.fetchall()
        cursor.close()
        return records[0]

    # Update Simulation Host This function is called every time we need to re - run the
    def update_simulation_host(self, host, attached, available):
        """
         @brief Update information about a simulation host. This is used to determine whether or not an agent is attached to a host or not.
         @param host The hostname of the host to update. It can be a host ID or an IP address.
         @param attached True if the agent is attached to the host False otherwise.
         @param available True if the agent is available to the host False otherwise
        """
        cursor = self.connection.cursor()
        cursor.execute('UPDATE simulation_host SET attached = %s, available = %s  WHERE hostname = %s',
                       (attached, available, host))
        self.connection.commit()
        cursor.close()

    # Insert simulation into database This function is called by simulation_insert. py to insert simulations
    def insert_simulation(self, creation_date, status, training_sweep_id, conf_sweep_ids):
        """
         @brief Insert a simulation into the database. This is a convenience method for inserting a simulation into the database
         @param creation_date The creation date of the simulation
         @param status The status of the simulation ( active failed etc. )
         @param training_sweep_id The ID of the training sweep
         @param conf_sweep_ids The IDs of the conf
        """
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT INTO simulation ("creation_date", "status", "training_sweep_id", "conf_sweep_ids") VALUES (%s, %s, %s, %s)',
            (creation_date, status, training_sweep_id, conf_sweep_ids))
        self.connection.commit()
        cursor.close()

    # Updates the simulation. This is called every time something changes in the simulation such as a change in time
    def update_simulation(self, sweep_id, status):
        """
         @brief Update status of simulation. This method is called by : meth : ` run_sweep ` to update the status of a simulation.
         @param sweep_id ID of the sweep to update.
         @param status Status of the simulation to update. Can be one of the following
        """
        cursor = self.connection.cursor()
        cursor.execute('UPDATE simulation SET status = %s WHERE training_sweep_id = %s', (status, sweep_id))
        self.connection.commit()
        cursor.close()

    # Inserts the data into sweeps and hosts. This is called after the simulation
    def insert_into_sweeps_and_hosts(self, sweep_id, host):
        """
         @brief Insert a sweep into the database. This is used to create sweeps_and_hosts tables
         @param sweep_id ID of the sweep to add
         @param host Host name or IP address of the sweep to
        """
        cursor = self.connection.cursor()
        cursor.execute('INSERT INTO sweeps_and_hosts ("sweep_id", "host") VALUES (%s, %s)', (sweep_id, host))
        self.connection.commit()
        cursor.close()


# This is a method to be called from clickhouse. cpp and it's not part of the public
class ClickhouseConnection:
    def __init__(self) -> None:
        """
         @brief Initializes the ClickhouseConnection. Connects to the Clickhouse server and returns the client
         @return The client to use
        """
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
        """
         @brief Close connection to RabbitMQ and clean up resources. This is called when the connection is
        """
        self.client.close()

        self.logger.info("Connection closed")

    def connect(self):
        """
         @brief Connect to Clickhouse and set self. client to the connection object. This is called by __init__
        """
        self.client = clickhouse_connect.get_client(host=os.getenv("CLICKHOUSE_HOST"),
                                                    port=8123,
                                                    database="v2data",
                                                    username=os.getenv("CLICKHOUSE_USERNAME"),
                                                    password=os.getenv("CLICKHOUSE_PASSWORD"))

    def get_funding_rate(self):
        """
         @brief Get 10 funds from funding_rate table and print to screen Args :
        """
        result = self.client.query('SELECT * FROM funding_rate LIMIT 10')
        rows = result.result_rows
        print(result.result_rows)
