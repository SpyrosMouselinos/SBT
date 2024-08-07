import itertools
import functools
import math
import logging
import logstash
import pickle
import shutil
import json
import wandb
import datetime
from fredapi import Fred
from numba import jit
from datetime import datetime
import numba
import os
import pandas as pd
import numpy as np


def itertools_flatten(aList):
    """
    Flatten a list of lists into a single list.

    This function takes a list of lists and concatenates them into a single list.

    @param aList A list of lists to be flattened.

    @return A single flattened list containing all elements of the sublists.
    """
    return list(itertools.chain(*aList))


def values_to_list(params: dict = None, inputs: list = None):
    """
    Transform specified keys in a dictionary into lists of their values, and update the dictionary in-place.

    This function iterates over each key specified in the 'inputs' list, collects values from the 'params' dictionary
    whose keys start with the specified input, and aggregates these values into a list. Each list of aggregated values
    is then stored in the 'params' dictionary under a new key formed by appending 'List' to the original input string.
    The original keys used to aggregate values are removed from the dictionary.

    @param params The dictionary to process. The default is None, which means an empty dictionary is used.
    @param inputs A list of strings representing the key prefixes to look for in the 'params' dictionary.
           The function aggregates values from keys that start with any of these strings. The default is None.

    @return The modified dictionary with aggregated lists of values for specified keys. Original keys that were used
            to aggregate values are removed, and new keys (each original key prefix with 'List' appended) are added to include
            the aggregated lists.

    @note The function modifies the 'params' dictionary in-place.
    @note If 'inputs' is None or an empty list, 'params' will be returned unchanged.
    @note It is assumed that the 'params' dictionary initially does not have keys that end with 'List' which are intended
          for use by this function. If such keys exist, their values might be overwritten or merged unexpectedly.

    @example
    >>> params = {'temp1': 20, 'temp2': 22, 'humidity': 80}
    >>> inputs = ['temp']
    >>> values_to_list(params, inputs)
    {'humidity': 80, 'tempList': [20, 22]}
    """
    for ix in inputs:
        empty_list = []
        for x, v in params.items():
            if x.startswith(ix):
                empty_list.append(params[x])
                del params[x]
            params = params | {ix + "List": empty_list}
    return params


def handle_different_size_lists(list1, list2, list3):
    """
    Adjust the lengths of three lists to ensure they are of equal length.

    This function checks if the input lists are of equal length. If they are not, it truncates
    the longer lists to match the length of the shortest one.

    @param list1 The first list.
    @param list2 The second list.
    @param list3 The third list.

    @return Three lists truncated to the same length.
    """
    if len(list1) != len(list2) or len(list1) != len(list3) or len(list2) != len(list3):
        if len(list2) != 0:
            min_length = min(len(list1), len(list2), len(list3))
            list1 = list1[:min_length]
            list2 = list2[:min_length]
            list3 = list3[:min_length]
        else:
            min_length = min(len(list1), len(list3))
            list1 = list1[:min_length]
            list2 = [0] * min_length
            list3 = list3[:min_length]
    return list1, list2, list3


def aggregated_volume(df):
    """
    Compute the aggregated volume of trades with the same price in a DataFrame.

    This function iterates over a DataFrame of trades, summing up the volume of consecutive trades
    that have the same price, and storing the result in a list.

    @param df A pandas DataFrame containing trade data with columns 'price' and 'volume'.

    @return A list of aggregated volumes for trades with the same price.
    """
    agr_volume = [df.loc[0, 'volume']]
    for ix in df.index:
        if ix != 0:
            if df.loc[ix - 1, 'price'] == df.loc[ix, 'price']:
                agr_volume.append(agr_volume[ix - 1] + df.loc[ix, 'volume'])
            else:
                agr_volume.append(df.loc[ix, 'volume'])
    return agr_volume


def parse_args(parser):
    """
    Parse command-line arguments and remove any with a value of 'delete'.

    This function uses an argument parser to parse command-line arguments, then removes any
    arguments that have a value of 'delete'.

    @param parser An argparse.ArgumentParser object used to parse command-line arguments.

    @return An argparse.Namespace object containing the parsed arguments.
    """
    args = parser.parse_args()
    temp_args = parser.parse_args()
    for arg in vars(temp_args):
        if hasattr(args, arg) and getattr(args, arg) == 'delete':
            args.__delattr__(arg)
    return args


def sharpe_sortino_ratio_fun(df, aum, t_start, t_end):
    """
    Calculate Sharpe and Sortino ratios from daily returns.

    This function calculates the Sharpe and Sortino ratios based on daily returns from a given DataFrame. It first
    fetches the risk-free rate from the Fred API and then computes the ratios using the provided data.

    @param df A pandas DataFrame containing daily return data with the column 'diff last-first'.
    @param aum The assets under management used for normalizing returns.
    @param t_start The start time in milliseconds since epoch.
    @param t_end The end time in milliseconds since epoch.

    @return A tuple containing mean daily rate of return, standard deviation of daily rate of return,
            Sharpe ratio, and Sortino ratio.
    """
    try:
        fred = Fred(api_key='2c145f752863180eb9303fa808629d43')
        t0 = datetime.datetime.fromtimestamp(t_start / 1000.0, tz=datetime.timezone.utc).strftime(
            "%m-%d-%Y %H:%M:%S")
        t1 = datetime.datetime.fromtimestamp(t_end / 1000.0, tz=datetime.timezone.utc).strftime(
            "%m-%d-%Y %H:%M:%S")
        data = fred.get_series('DGS1', observation_start=t0, observation_end=t1)
        data.dropna(inplace=True)
        risk_free_rate = data.iloc[-1] / 100
    except:
        risk_free_rate = 0.028

    df['daily_return'] = df['diff last-first'] / aum

    mean_daily_ror = df['daily_return'].mean()
    std_daily_ror = df['daily_return'].std()
    sharpe_ratio = np.sqrt(365) * (df['daily_return'].mean() - risk_free_rate / 365) / \
                   (df['daily_return'].std())

    sortino_ratio = np.sqrt(365) * (df['daily_return'].mean() - risk_free_rate / 365) / \
                    (df.loc[df['daily_return'] < 0, 'daily_return'].std())

    return mean_daily_ror, std_daily_ror, sharpe_ratio, sortino_ratio


def to_decimal(bp):
    """
    Convert basis points to a decimal representation.

    @param bp The value in basis points.

    @return The value converted to decimal representation.
    """
    return bp / 10000


def bp_to_dollars(bp, dollar_value):
    """
    Convert basis points to a dollar amount based on a given dollar value.

    @param bp The value in basis points.
    @param dollar_value The dollar value to which the basis points are applied.

    @return The dollar amount corresponding to the given basis points and dollar value.
    """
    return to_decimal(bp) * dollar_value


class Util:
    """
    Utility class providing various static methods for financial computations and data processing.
    """

    @staticmethod
    def add_data_orderbook(book, data, price_change_step):
        """
        Parse order book data and add orders to a given order book.

        This function parses price, quantity, and side information from the given data array,
        then adds the parsed orders to the specified order book.

        @param book The order book object to which orders will be added.
        @param data A NumPy array containing order data with columns for price, quantity, and side.
        @param price_change_step The step size for price changes.

        @note The format of the input data is expected to be (price, quantity, side), where side is 'Bid' or 'Ask'.
        """
        # parsing the prices
        first_column = np.around(((data[:, 1].astype(np.float)) * 100)).astype(np.int)
        first_column = getattr(first_column, "tolist", lambda x=first_column: x)()

        # parsing the quantity
        try:
            second_column = np.array(data[:, 2], dtype=np.int64)
        except ValueError:
            second_column = np.array(data[:, 2]).astype(float).astype(int)  # TODO this seems redundant
        # second_column = np.array(second_column*(10**8), dtype=np.int64) # TODO still holds if it's a spot market
        second_column = getattr(second_column, "tolist", lambda x=second_column: x)()

        # parsing the side (True == Bid, False == Ask)
        third_column = data[:, 3] == 'Bid'
        third_column = getattr(third_column, "tolist", lambda x=third_column: x)()

        # format is (price, quantity, side)
        book.add_orders(first_column, second_column, third_column, price_change_step)

    @staticmethod
    @jit(nopython=True)
    def discretizing_orderbook(bidprices, bidvolumes, askprices, askvolumes, num_levels=10):
        """
        Discretize an order book by aggregating volumes at specific price levels.

        This function discretizes the order book by aggregating volumes for bid and ask prices
        into a specified number of levels.

        @param bidprices A NumPy array of bid prices.
        @param bidvolumes A NumPy array of bid volumes.
        @param askprices A NumPy array of ask prices.
        @param askvolumes A NumPy array of ask volumes.
        @param num_levels The number of price levels for discretization (default is 10).

        @return A tuple containing arrays of discretized bid volumes and ask volumes.
        """
        bins = np.arange(bidprices[0] - bidprices[-1], 0, -50)
        bins = (bins + bidprices[-1])
        binned = np.digitize(bidprices, bins)
        binned = np.bincount(binned)

        start = 0
        maxn = np.max(binned)
        bidvolumes_subset = np.zeros(shape=(len(binned), maxn))
        for j in range(len(binned)):
            end = start + binned[j]
            current = bidvolumes[start:end]
            bidvolumes_subset[j, 0:len(current)] = current
            start = end
        bidvolumes_subset_result = np.sum(bidvolumes_subset, axis=1)
        bidvolumes_subset_result = bidvolumes_subset_result[0:num_levels]

        bins = np.arange(0, askprices[-1] - askprices[0], 50)
        bins = (bins + askprices[0] + 50)  # TODO: CHANGED RECENTLY, CHECK
        binned = np.digitize(askprices, bins)
        binned = np.bincount(binned)

        start = 0
        maxn = np.max(binned)
        askvolumes_subset = np.zeros(shape=(len(binned), maxn))
        for j in range(len(binned)):
            end = start + binned[j]
            current = askvolumes[start:end]
            askvolumes_subset[j, 0:len(current)] = current
            start = end
        askvolumes_subset_result = np.sum(askvolumes_subset, axis=1)
        askvolumes_subset_result = askvolumes_subset_result[0:num_levels]

        return bidvolumes_subset_result, askvolumes_subset_result

    def create_single_orderbook(self, bincount, data, book, start, end, counter):
        """
        Create a single order book snapshot from a given data range.

        This function creates a snapshot of an order book by processing data within a specified range,
        adding the parsed orders to the order book object, and updating the start and end indices.

        @param bincount An array representing the number of elements in each data bin.
        @param data A NumPy array containing order book data.
        @param book The order book object to which orders will be added.
        @param start The starting index for the current data range.
        @param end The ending index for the current data range.
        @param counter A counter used for tracking the current data bin.

        @return A tuple containing the updated order book's bid prices, bid volumes, ask prices, ask volumes,
                the updated start and end indices, and the incremented counter.
        """
        end = start + bincount[counter]
        # movements in a 50 ms window
        snap = data[start:end, :]

        self.add_data_orderbook(book, snap, 50)
        start = end  # len(np.where(np.array(book.askVolumes()) == 0)[0]) > 0 and len(np.where(np.array(book.askVolumes()) == 0)[0]) < 4 and counter > 1000
        counter += 1
        return book.bidPrices(), book.bidVolumes(), book.askPrices(), book.askVolumes(), start, end, counter

    @staticmethod
    def create_array_from_generator(gen):
        """
        Convert a generator into a NumPy array.

        This function takes a generator and extracts its elements, converting them into a NumPy array.

        @param gen A generator object.

        @return A NumPy array containing the elements extracted from the generator.
        """
        r = np.array([x for x in gen])
        data = []
        for dic in r:
            data.append(list(dic.values()))
        data = np.array(data)
        return data

    @staticmethod
    def step_decay(epoch):
        """
        Calculate the learning rate decay based on the current epoch.

        This function computes the learning rate decay using a step decay schedule.

        @param epoch The current epoch number.

        @return The adjusted learning rate.
        """
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 100.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    @staticmethod
    def remove_digits(string):
        """
        Remove all digits from a string.

        This function removes all numeric characters from the input string.

        @param string The input string.

        @return A string with all digits removed.
        """
        return ''.join(i for i in string if not i.isdigit())

    @staticmethod
    def drop_nans(array):
        """
        Remove NaN values from a NumPy array.

        This function removes NaN values from a NumPy array and returns a new array
        containing only non-NaN values.

        @param array The input NumPy array.

        @return A new NumPy array with NaN values removed.
        """
        array_float = array.astype("float64")
        if len(array_float.shape) == 2:
            mask = ~np.isnan(array_float[:, 1])
        else:
            mask = ~np.isnan(array_float)
        return array_float[mask]

    @staticmethod
    def remove_null_values_in_array(array):
        """
        Remove null values from a NumPy array.

        This function removes null values from a NumPy array by replacing them with
        the last non-null value in the array.

        @param array The input NumPy array.

        @return A new NumPy array with null values removed.
        """
        try:
            index_non_null_value = np.where(array[:, 1] == None)[0][-1] + 1
            array[:index_non_null_value, 1:] = array[index_non_null_value, 1:]
            return array
        except IndexError:
            return array

    @staticmethod
    def get_digits(string):
        """
        Extract digits from a string and convert them to an integer.

        This function extracts all numeric characters from the input string and converts
        them to an integer.

        @param string The input string.

        @return An integer representing the extracted digits from the string.
        """
        return int(''.join(i for i in string if i.isdigit()))

    @staticmethod
    def floor_to_hour(timestamp):
        """
        Round a timestamp down to the nearest hour.

        This function rounds the given timestamp down to the nearest hour.

        @param timestamp The input timestamp in milliseconds since epoch.

        @return The timestamp rounded down to the nearest hour.
        """
        return int(np.floor(timestamp / (1000 * 60 * 60)) * 1000 * 60 * 60)

    @staticmethod
    def round_to_day(timestamp):
        """
        Round a timestamp to the nearest day.

        This function rounds the given timestamp to the nearest day.

        @param timestamp The input timestamp in milliseconds since epoch.

        @return The timestamp rounded to the nearest day.
        """
        return int(np.round(timestamp / (1000 * 60 * 60 * 24)) * 1000 * 60 * 60 * 24)

    @staticmethod
    def get_filters_dictionary(filters_string):
        """
        Parse a filter string and convert it into a dictionary.

        This function parses a filter string and creates a dictionary mapping filter names
        to their corresponding numeric values.

        @param filters_string The input filter string.

        @return A dictionary with filter names as keys and their numeric values as values.
        """
        filters_list = filters_string.replace("self.", "").replace("filter_", "").split(" * ")
        return {Util.remove_digits(x).replace("()", ""): Util.get_digits(x) for x in filters_list}

    @staticmethod
    def find_local_wandb_run_folder(wandb_directory, run_id):
        """
        Find the local directory for a specific Weights & Biases run.

        This function searches a local directory for the folder corresponding to a specific
        Weights & Biases run ID.

        @param wandb_directory The local directory containing Weights & Biases run folders.
        @param run_id The ID of the Weights & Biases run to find.

        @return The path to the local folder for the specified run, or None if not found.
        """
        run_names = os.listdir(wandb_directory)
        names_with_run_id = [run_name for run_name in run_names if run_id in run_name]
        if len(names_with_run_id) == 1:
            return os.path.join(wandb_directory, names_with_run_id[0])
        # @ TODO deal with multiple runs with the same id

    @staticmethod
    def load_processor_from_artifact(current_run_from_api):
        """
        Load a processor object from a Weights & Biases artifact.

        This function downloads the processor artifact from a specific Weights & Biases run
        and loads the processor object from it.

        @param current_run_from_api The Weights & Biases run object from which to load the processor.

        @return The loaded processor object.

        @throws AssertionError If the processor artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        processor_artifact = None
        for artifact in artifacts:
            if artifact.type == 'processor':
                processor_artifact = artifact
                break
        assert processor_artifact is not None, f"Run {current_run_from_api.id}: Processor not found"
        artifact_dir = processor_artifact.download()
        with open(os.path.join(artifact_dir, f'processor_{current_run_from_api.id}.pkl'), 'rb') as f:
            processor = pickle.load(f)
        return processor

    @staticmethod
    def load_processor_to_artifact(initialised_wandb, processor, dir):
        """
        Save a processor object to a Weights & Biases artifact.

        This function saves a processor object to a Weights & Biases artifact and logs it to
        the initialized run.

        @param initialised_wandb The initialized Weights & Biases run object.
        @param processor The processor object to save.
        @param dir The directory in which to temporarily store the processor file.

        @note The processor file is removed from the local directory after being logged as an artifact.
        """
        with open(os.path.join(dir, f'processor_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(processor, fid)
        artifact = initialised_wandb.Artifact('processor', type='processor')
        artifact.add_file(os.path.join(dir, f'processor_{initialised_wandb.run.id}.pkl'))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"processor_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def load_model_from_artifact(current_run_from_api):
        """
        Load a trained model object from a Weights & Biases artifact.

        This function downloads the trained model artifact from a specific Weights & Biases run
        and loads the model object from it.

        @param current_run_from_api The Weights & Biases run object from which to load the model.

        @return The loaded model object.

        @throws AssertionError If the model artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        model_artifact = None
        for artifact in artifacts:
            if artifact.type == 'model':
                model_artifact = artifact
                break
        assert model_artifact is not None, f"Run {current_run_from_api.id}: Trained model not found"
        artifact_dir = model_artifact.download()
        with open(os.path.join(artifact_dir, f'trained_model_{current_run_from_api.id}.pkl'), 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def load_writer_from_artifact(current_run_from_api):
        """
        Load a writer object from a Weights & Biases artifact.

        This function downloads the writer artifact from a specific Weights & Biases run
        and loads the writer object from it.

        @param current_run_from_api The Weights & Biases run object from which to load the writer.

        @return The loaded writer object.

        @throws AssertionError If the writer artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        writer_artifact = None
        for artifact in artifacts:
            if artifact.type == 'writer':
                writer_artifact = artifact
                break
        assert writer_artifact is not None, f"Run {current_run_from_api.id}: Writer not found"
        artifact_dir = writer_artifact.download()
        with open(os.path.join(artifact_dir, f'writer_{current_run_from_api.id}.pkl'), 'rb') as f:
            writer = pickle.load(f)
        return writer

    @staticmethod
    def load_writer_to_artifact(initialised_wandb, writer, dir):
        """
        Save a writer object to a Weights & Biases artifact.

        This function saves a writer object to a Weights & Biases artifact and logs it to
        the initialized run.

        @param initialised_wandb The initialized Weights & Biases run object.
        @param writer The writer object to save.
        @param dir The directory in which to temporarily store the writer file.

        @note The writer file is removed from the local directory after being logged as an artifact.
        """
        with open(os.path.join(dir, f'writer_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(writer, fid)
        artifact = initialised_wandb.Artifact('writer', type='writer')
        artifact.add_file(os.path.join(dir, f"writer_{initialised_wandb.run.id}.pkl"))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"writer_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def load_data_evaluation_loader_from_artifact(current_run_from_api):
        """
        Load a data evaluation loader object from a Weights & Biases artifact.

        This function downloads the data evaluation loader artifact from a specific Weights & Biases run
        and loads the object from it.

        @param current_run_from_api The Weights & Biases run object from which to load the data evaluation loader.

        @return The loaded data evaluation loader object.

        @throws AssertionError If the data evaluation loader artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        data_evaluation_loader_artifact = None
        for artifact in artifacts:
            if artifact.type == 'data_evaluation_loader':
                data_evaluation_loader_artifact = artifact
                break
        assert data_evaluation_loader_artifact is not None, f"Run {current_run_from_api.id}: Data evaluation loader not found"
        artifact_dir = data_evaluation_loader_artifact.download()
        with open(os.path.join(artifact_dir, f'data_evaluation_loader_{current_run_from_api.id}.pkl'), 'rb') as f:
            data_evaluation_loader = pickle.load(f)
        return data_evaluation_loader

    @staticmethod
    def load_data_aggregator_from_artifact(current_run_from_api):
        """
        Load a data aggregator object from a Weights & Biases artifact.

        This function downloads the data aggregator artifact from a specific Weights & Biases run
        and loads the object from it.

        @param current_run_from_api The Weights & Biases run object from which to load the data aggregator.

        @return The loaded data aggregator object.

        @throws AssertionError If the data aggregator artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        aggregator_artifact = None
        for artifact in artifacts:
            if artifact.type == 'aggregator':
                aggregator_artifact = artifact
                break
        assert aggregator_artifact is not None, f"Run {current_run_from_api.id}: Aggregator not found"
        artifact_dir = aggregator_artifact.download()
        with open(os.path.join(artifact_dir, f'aggregator_{current_run_from_api.id}.pkl'), 'rb') as f:
            aggregator = pickle.load(f)
        return aggregator

    @staticmethod
    def load_data_evaluation_loader_to_artifact(initialised_wandb, data_evaluation_loader, dir):
        """
        Save a data evaluation loader object to a Weights & Biases artifact.

        This function saves a data evaluation loader object to a Weights & Biases artifact and logs it to
        the initialized run.

        @param initialised_wandb The initialized Weights & Biases run object.
        @param data_evaluation_loader The data evaluation loader object to save.
        @param dir The directory in which to temporarily store the data evaluation loader file.

        @note The data evaluation loader file is removed from the local directory after being logged as an artifact.
        """
        with open(os.path.join(dir, f'data_evaluation_loader_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(data_evaluation_loader, fid)
        artifact = initialised_wandb.Artifact('data_evaluation_loader', type='data_evaluation_loader')
        artifact.add_file(os.path.join(dir, f"data_evaluation_loader_{initialised_wandb.run.id}.pkl"))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"data_evaluation_loader_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def update_table_artifact(current_run_from_api, current_run_resumed, data, remove_old_versions=True):
        """
        Update a score table artifact in a Weights & Biases run with new data.

        This function updates an existing score table artifact in a Weights & Biases run with new data,
        and optionally removes old versions of the artifact.

        @param current_run_from_api The Weights & Biases run object from which to update the score table.
        @param current_run_resumed The Weights & Biases run object to which the updated score table will be logged.
        @param data The new data to be added to the score table.
        @param remove_old_versions A boolean indicating whether to remove old versions of the score table artifact (default is True).

        @throws AssertionError If the score table artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        score_table_artifact = None
        for artifact in artifacts:
            if artifact.type == 'run_table' and 'score_table' in artifact.name and 'latest' in artifact.aliases:
                score_table_artifact = artifact
                break
        assert score_table_artifact is not None, f"Run {current_run_from_api.id}: Trained accuracy_table not found"
        _ = score_table_artifact.download()
        score_table = current_run_from_api.use_artifact(score_table_artifact).get('score_table.table.json')
        current_run_resumed.log(
            {'score_table': wandb.Table(data=score_table.data + [data], columns=score_table.columns)})
        if remove_old_versions:
            for artifact in artifacts:
                if artifact.type == 'run_table' and 'score_table' in artifact.name and 'latest' not in artifact.aliases:
                    artifact.delete()

    @staticmethod
    def load_accuracy_table_from_artifact(current_run_from_api, run_a, api):
        """
        Load an accuracy table from a Weights & Biases artifact and update it with new data.

        This function downloads an accuracy table artifact from a specific Weights & Biases run,
        updates the table with new data, and logs it to another run.

        @param current_run_from_api The Weights & Biases run object from which to load the accuracy table.
        @param run_a The Weights & Biases run object to which the updated accuracy table will be logged.
        @param api The Weights & Biases API object for interacting with artifacts.

        @return The loaded accuracy table as a JSON object.

        @throws AssertionError If the accuracy table artifact is not found in the run.
        """
        artifacts = current_run_from_api.logged_artifacts()
        accuracy_table_artifact = None
        for artifact in artifacts:
            if artifact.type == 'run_table' and 'test_table' in artifact.name and 'latest' in artifact.aliases:
                accuracy_table_artifact = artifact
                break
        assert accuracy_table_artifact is not None, f"Run {current_run_from_api.id}: Trained accuracy_table not found"
        artifact_dir = accuracy_table_artifact.download()
        a = current_run_from_api.use_artifact(accuracy_table_artifact).get('test_table.table.json')
        # a.add_data(1, 1, 'a')
        it = current_run_from_api.summary['test_table'].items()
        mm = []
        for i in it:
            mm.append(i)
        run_a.log({'test_table': wandb.Table(data=[[1, 2, 'n']] + a.data, columns=["Threshold", "y", 'Type'])})
        b = wandb.Artifact(name='fun_with_WANDB', type='run_table')
        b.add(a, 'custom_table')
        str(current_run_from_api.files()[6].download(replace=True))
        # accuracy_table_artifact.add(a, 'my_custom_id_table')
        api.artifact(a)
        run_a.summary['my_custom_id'] = a
        run_a.summary['my_custom_id'].update({"Threshold": 1, "y": 2, "Type": "PNL"})
        run_a.summary.x = 1
        run_a.update()
        with open(os.path.join(artifact_dir, f'my_custom_id_table.table.json'), 'rb') as f:
            accuracy_table = json.load(f)
        return accuracy_table

    @staticmethod
    def remove_artifacts_from_disk(dir):
        """
        Remove all artifacts from the specified directory.

        This function deletes the 'artifacts' subdirectory from the given directory if it exists.

        @param dir The directory from which to remove artifacts.

        @note This function removes the entire 'artifacts' subdirectory and its contents.
        """
        if 'artifacts' in os.listdir(dir):
            shutil.rmtree('artifacts')

    @staticmethod
    def get_logger(name):
        """
        Create and configure a logger with Logstash support.

        This function creates a logger with the specified name and configures it to send logs
        to a Logstash server if the environment is set to 'staging' or 'prod'.

        @param name The name of the logger.

        @return A configured logger object.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if os.getenv("ENV") in ['staging', 'prod']:
            logger.addHandler(logstash.TCPLogstashHandler('logstash', 5000, version=1))
        return logger

    @staticmethod
    def get_time_string_from_timestamp(timestamp):
        """
        Convert a timestamp in milliseconds to a formatted time string.

        @param timestamp The input timestamp in milliseconds since epoch.

        @return A formatted time string in the format "YYYY-MM-DD HH:MM:SS.ffffff".
        """
        # Timestamp expected to be in milliseconds
        return datetime.datetime.fromtimestamp(timestamp / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def indices_to_one_hot(data, nb_classes=None):
        """
        Convert an iterable of indices to one-hot encoded labels.

        @param data An iterable of indices.
        @param nb_classes The number of classes for one-hot encoding. If None, it defaults to the maximum index value plus one.

        @return A NumPy array containing one-hot encoded labels.
        """
        data_int = data.astype(np.int64)
        nb_classes = nb_classes if nb_classes else data_int.max() + 1
        one_hot = np.zeros((data_int.size, nb_classes))
        one_hot[np.arange(data_int.size), data_int] = 1
        return one_hot[:]

    @staticmethod
    def unison_shuffled_copies(a, b, c, d):
        """
        Shuffle the elements of four arrays in unison.

        This function shuffles the elements of four input arrays such that corresponding elements maintain their relative positions.

        @param a A NumPy array to be shuffled.
        @param b A NumPy array to be shuffled.
        @param c A NumPy array to be shuffled.
        @param d A NumPy array to be shuffled.

        @return A tuple of four shuffled NumPy arrays.

        @throws AssertionError If the input arrays are not of equal length.
        """
        assert len(a) == len(b)
        assert len(a) == len(c)
        assert len(a) == len(d)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p]

    @staticmethod
    def unison_shuffled_copies_2(a, b):
        """
        Shuffle the elements of two arrays in unison.

        This function shuffles the elements of two input arrays such that corresponding elements maintain their relative positions.

        @param a A NumPy array to be shuffled.
        @param b A NumPy array to be shuffled.

        @return A tuple of two shuffled NumPy arrays.

        @throws AssertionError If the input arrays are not of equal length.
        """
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def organize_into_bins(data, t0, t1, step):
        """
        Organize data into bins based on a specified step size.

        This function creates bins of equal width based on the given step size and assigns each element
        in the input data to a bin.

        @param data A NumPy array containing data to be organized into bins.
        @param t0 The start time for binning.
        @param t1 The end time for binning.
        @param step The width of each bin.

        @return An array of indices representing the bin to which each element in the input data belongs.
        """
        bins = np.linspace(step, t1 - (t0 + step), num=int((t1 - (t0 + step)) / step))
        data = np.array(data[:, 0], dtype=np.int64)
        bins = (bins + t0).astype(np.int64)

        # returns the indices of the bins to which each value in input array belongs
        inds = np.digitize(data, bins)

        return inds

    @staticmethod
    def organize_into_bins_vector(data, bins, t0):
        """
        Organize data into bins using vectorized search.

        This function uses a vectorized search to assign each element in the input data to a bin based on the specified bins and starting time.

        @param data A NumPy array containing data to be organized into bins.
        @param bins An array of bin edges.
        @param t0 The starting time for binning.

        @return An array of indices representing the bin to which each element in the input data belongs.
        """
        bins = (bins + t0)
        inds = np.searchsorted(bins, data, side='right')

        return inds

    @staticmethod
    def create_array_from_generator(gen):
        """
        Convert a generator into a NumPy array.

        This function takes a generator and extracts its elements, converting them into a NumPy array.

        @param gen A generator object.

        @return A NumPy array containing the elements extracted from the generator.
        """
        # parse all the elements in the generator
        r = np.array([x for x in gen])

        data = []

        for dic in r:
            data.append(list(dic.values()))
        data = np.array(data)

        return data

    @staticmethod
    def create_array_from_query(array):
        """
        Create a NumPy array from a query result.

        This function extracts values from a query result and converts them into a NumPy array.

        @param array A query result object containing the data.

        @return A NumPy array containing the extracted values, or an empty array if an error occurs.
        """
        try:
            result = np.array(array.raw['series'][0]['values'])
        except (ValueError, IndexError, KeyError):
            result = np.array([])
        return result

    @staticmethod
    def time_plus_one_hour(time):
        """
        Add one hour to a given time.

        @param time The input time in milliseconds since epoch.

        @return The input time plus one hour.
        """
        hour = 1000 * 60 * 60
        return time + hour

    @staticmethod
    def time_plus_one_day(time):
        """
        Add one day to a given time.

        @param time The input time in milliseconds since epoch.

        @return The input time plus one day.
        """
        day = 1000 * 60 * 60 * 24
        return time + day

    @staticmethod
    def normalize(values, min, max):
        """
        Normalize values to a range of 0 to 1.

        This function normalizes the input values to a range between 0 and 1 based on the specified minimum and maximum values.

        @param values A NumPy array containing the values to be normalized.
        @param min The minimum value of the range.
        @param max The maximum value of the range.

        @return A NumPy array containing the normalized values.
        """
        return (values - min) / (max - min)

    @staticmethod
    def normalize_inverse(values, min, max):
        """
        Inverse normalization of values from a range of 0 to 1.

        This function applies inverse normalization to convert the input values from a range between 0 and 1 to their original scale.

        @param values A NumPy array containing the values to be inverse normalized.
        @param min The minimum value of the original range.
        @param max The maximum value of the original range.

        @return A NumPy array containing the inverse normalized values.
        """
        return values * (max - min) + min

    @staticmethod
    def standardize(values, mean, stddev):
        """
        Standardize values by subtracting the mean and dividing by the standard deviation.

        This function standardizes the input values to have a mean of 0 and a standard deviation of 1.

        @param values A NumPy array containing the values to be standardized.
        @param mean The mean value used for standardization.
        @param stddev The standard deviation used for standardization.

        @return A NumPy array containing the standardized values.
        """
        return (values - mean) / stddev

    @staticmethod
    def standardize_inverse(values, mean, stddev):
        """
        Inverse standardize values by multiplying by the standard deviation and adding the mean.

        This function applies inverse standardization to convert standardized values back to their original scale.

        @param values A NumPy array containing the standardized values to be inverse standardized.
        @param mean The mean value used for inverse standardization.
        @param stddev The standard deviation used for inverse standardization.

        @return A NumPy array containing the inverse standardized values.
        """
        return values * stddev + mean

    @staticmethod
    def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
        """
        Create a view of `array` which for every point gives the n-dimensional
        neighbourhood of size window. New dimensions are added at the end of
        `array` or after the corresponding original dimension.

        @param array Array to which the rolling window is applied.
        @param window Either a single integer to create a window of only the last axis or a
                      tuple to create it for the last len(window) axes. 0 can be used as a
                      to ignore a dimension in the window.
        @param asteps Aligned at the last axis, new steps for the original array, i.e., for
                      creation of non-overlapping windows. (Equivalent to slicing result)
        @param wsteps Steps for the added window dimensions. These can be 0 to repeat values
                      along the axis.
        @param axes If given, must have the same size as window. In this case window is
                    interpreted as the size in the dimension given by axes. IE. a window
                    of (2, 1) is equivalent to window=2 and axis=-2.
        @param toend If False, the new dimensions are right after the corresponding original
                     dimension, instead of at the end of the array. Adding the new axes at the
                     end makes it easier to get the neighborhood, however toend=False will give
                     a more intuitive result if you view the whole array.

        @return A view on `array` which is smaller to fit the windows and has windows added
                dimensions (0s not counting), i.e., every point of `array` is an array of size
                window.

        @example
        >>> a = np.arange(9).reshape(3,3)
        >>> rolling_window(a, (2,2))
        array([[[[0, 1],
                 [3, 4]],

                [[1, 2],
                 [4, 5]]],


               [[[3, 4],
                 [6, 7]],

                [[4, 5],
                 [7, 8]]]])

        Or to create non-overlapping windows, but only along the first dimension:
        >>> rolling_window(a, (2,0), asteps=(2,1))
        array([[[0, 3],
                [1, 4],
                [2, 5]]])

        Note that the 0 is discarded, so that the output dimension is 3:
        >>> rolling_window(a, (2,0), asteps=(2,1)).shape
        (1, 3, 2)

        This is useful for example to calculate the maximum in all (overlapping)
        2x2 submatrices:
        >>> rolling_window(a, (2,2)).max((2,3))
        array([[4, 5],
               [7, 8]])

        Or delay embedding (3D embedding with delay 2):
        >>> x = np.arange(10)
        >>> rolling_window(x, 3, wsteps=2)
        array([[0, 2, 4],
               [1, 3, 5],
               [2, 4, 6],
               [3, 5, 7],
               [4, 6, 8],
               [5, 7, 9]])
        """
        array = np.asarray(array)
        orig_shape = np.asarray(array.shape)
        window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

        if axes is not None:
            axes = np.atleast_1d(axes)
            w = np.zeros(array.ndim, dtype=int)
            for axis, size in zip(axes, window):
                w[axis] = size
            window = w

        # Check if window is legal:
        if window.ndim > 1:
            raise ValueError("`window` must be one-dimensional.")
        if np.any(window < 0):
            raise ValueError("All elements of `window` must be larger than 1.")
        if len(array.shape) < len(window):
            raise ValueError("`window` length must be less or equal `array` dimension.")

        _asteps = np.ones_like(orig_shape)
        if asteps is not None:
            asteps = np.atleast_1d(asteps)
            if asteps.ndim != 1:
                raise ValueError("`asteps` must be either a scalar or one dimensional.")
            if len(asteps) > array.ndim:
                raise ValueError("`asteps` cannot be longer than the `array` dimension.")
            # does not enforce alignment, so that steps can be the same as window too.
            _asteps[-len(asteps):] = asteps

            if np.any(asteps < 1):
                raise ValueError("All elements of `asteps` must be larger than 1.")
        asteps = _asteps

        _wsteps = np.ones_like(window)
        if wsteps is not None:
            wsteps = np.atleast_1d(wsteps)
            if wsteps.shape != window.shape:
                raise ValueError("`wsteps` must have the same shape as `window`.")
            if np.any(wsteps < 0):
                raise ValueError("All elements of `wsteps` must be larger than 0.")

            _wsteps[:] = wsteps
            _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
        wsteps = _wsteps

        # Check that the window would not be larger than the original:
        if np.any(orig_shape[-len(window):] < window * wsteps):
            raise ValueError("`window` * `wsteps` larger than `array` in at least one dimension.")

        new_shape = orig_shape  # just renaming...

        # For calculating the new shape 0s must act like 1s:
        _window = window.copy()
        _window[_window == 0] = 1

        new_shape[-len(window):] += wsteps - _window * wsteps
        new_shape = (new_shape + asteps - 1) // asteps
        # make sure the new_shape is at least 1 in any "old" dimension (i.e., steps
        # is (too) large, but we do not care.
        new_shape[new_shape < 1] = 1
        shape = new_shape

        strides = np.asarray(array.strides)
        strides *= asteps
        new_strides = array.strides[-len(window):] * wsteps

        # The full new shape and strides:
        if toend:
            new_shape = np.concatenate((shape, window))
            new_strides = np.concatenate((strides, new_strides))
        else:
            _ = np.zeros_like(shape)
            _[-len(window):] = window
            _window = _.copy()
            _[-len(window):] = new_strides
            _new_strides = _

            new_shape = np.zeros(len(shape) * 2, dtype=int)
            new_strides = np.zeros(len(shape) * 2, dtype=int)

            new_shape[::2] = shape
            new_strides[::2] = strides
            new_shape[1::2] = _window
            new_strides[1::2] = _new_strides

        new_strides = new_strides[new_shape != 0]
        new_shape = new_shape[new_shape != 0]

        return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

    @staticmethod
    def influx_points_from_dataframe(dataframe, measurement,
                                     field_columns=None,
                                     tag_columns=None,
                                     global_tags=None,
                                     time_precision=None,
                                     numeric_precision=None):
        """
        Convert a pandas DataFrame into a list of InfluxDB points.

        This function takes a DataFrame with a DatetimeIndex or PeriodIndex and converts it into a list
        of dictionaries formatted for InfluxDB, using the specified measurement name, field columns, and tag columns.

        @param dataframe A pandas DataFrame with time-based index.
        @param measurement A string representing the measurement name for InfluxDB points.
        @param field_columns A list of columns in the DataFrame to be used as fields in the InfluxDB points.
        @param tag_columns A list of columns in the DataFrame to be used as tags in the InfluxDB points.
        @param global_tags A dictionary of global tags to add to every point.
        @param time_precision A string representing the time precision for InfluxDB (e.g., 's', 'ms', 'us', 'ns').
        @param numeric_precision A string for numeric precision settings.

        @return A list of dictionaries representing points for InfluxDB.

        @throws TypeError If the input DataFrame does not have a valid index type.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError('Must be DataFrame, but type was: {0}.'.format(type(dataframe)))
        if not (isinstance(dataframe.index, pd.PeriodIndex) or isinstance(dataframe.index, pd.DatetimeIndex)):
            raise TypeError('Must be DataFrame with DatetimeIndex or PeriodIndex.')

        precision_factor = int({
                                   "n": 1,
                                   "u": 1e3,
                                   "ms": 1e6,
                                   "s": 1e9,
                                   "m": 1e9 * 60,
                                   "h": 1e9 * 3600,
                               }.get(time_precision, 1))

        if time_precision == 'n':
            timestamps = np.array((dataframe.index.view(np.int64))).astype(np.int64)
        else:
            timestamps = np.array((dataframe.index.view(np.int64) + 1) / precision_factor).astype(np.int64)

        points = []
        for j in range(len(dataframe)):
            x = dataframe.iloc[j]
            fields_for_point = {}
            tags_for_point = {}
            for t in tag_columns:
                tags_for_point[t] = x[t]
            for f in field_columns:
                fields_for_point[f] = x[f]
            point = {'time': timestamps[j], 'measurement': measurement,
                     'tags': tags_for_point,
                     'fields': fields_for_point}
            points.append(point)
        return points


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    """
    Find the leftmost index in the timestamps array that satisfies the latency condition.

    This function searches backward from the current index to find the first timestamp
    that is less than the timestamp at the current index minus the latency.

    @param timestamps A NumPy array of timestamps.
    @param current_index The current index in the timestamps array.
    @param latency The latency threshold in the same units as the timestamps.

    @return The index of the leftmost timestamp satisfying the latency condition, or None if not found.
    """
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    """
    Find the rightmost index in the timestamps array that satisfies the latency condition.

    This function searches forward from the current index to find the first timestamp
    that is greater than the timestamp at the current index plus the latency.

    @param timestamps A NumPy array of timestamps.
    @param current_index The current index in the timestamps array.
    @param latency The latency threshold in the same units as the timestamps.

    @return The index of the rightmost timestamp satisfying the latency condition, or -1 if not found.
    """
    done = True
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            done = False
            return j
    if done:
        return -1


@functools.lru_cache(maxsize=3)
def cached_get_loc(a, v, side):
    """
    Cache the result of getting the location of a value in a pandas Index or Series.

    This function caches the result of the get_loc method for a given index or series
    to improve performance for repeated lookups of the same value.

    @param a The pandas Index or Series to search.
    @param v The value to locate within the Index or Series.
    @param side A string indicating the side to search ('left' or 'right').

    @return The location of the value within the Index or Series.
    """
    return a.get_loc(v)


@numba.jit(nopython=True)
def pop_old(l, timestamp, interval):
    """
    Remove elements from the list that are older than a given interval.

    This function iterates through the list and removes elements that are
    older than the specified interval from the given timestamp.

    @param l A list of elements with timestamps.
    @param timestamp The current timestamp.
    @param interval The interval threshold for removing old elements.

    @return The list with old elements removed.
    """
    length = len(l)
    for j in range(length):
        if len(l) == 0:
            return l
        if l[0] < timestamp - interval:
            l.pop(0)
        else:
            return l
    return l


@numba.jit(nopython=True)
def count_from_end(l, timestamp, interval):
    """
    Count the number of elements from the end of the list that are within a given interval.

    This function counts the number of elements from the end of the list that
    are within the specified interval from the given timestamp.

    @param l A list of elements with timestamps.
    @param timestamp The current timestamp.
    @param interval The interval threshold for counting elements.

    @return The count of elements within the interval.
    """
    count = 0
    for j in range(len(l) - 1, -1, -1):
        if l[j] > timestamp - interval:
            count += 1
        else:
            return count
    return count


@numba.jit()
def spread_entry_func_numba(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the entry spread between swap and spot prices using Numba for optimization.

    This function computes the entry spread between swap and spot prices, taking into
    account the fees associated with each transaction.

    @param entry_swap The entry price for the swap.
    @param entry_spot The entry price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The entry spread between the swap and spot prices.
    """
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


@numba.jit()
def spread_exit_func_numba(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the exit spread between swap and spot prices using Numba for optimization.

    This function computes the exit spread between swap and spot prices, taking into
    account the fees associated with each transaction.

    @param exit_swap The exit price for the swap.
    @param exit_spot The exit price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The exit spread between the swap and spot prices.
    """
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


@numba.jit()
def spread_entry_func_bp_numba(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the entry spread in basis points between swap and spot prices using Numba for optimization.

    This function computes the entry spread between swap and spot prices in basis points,
    taking into account the fees associated with each transaction.

    @param entry_swap The entry price for the swap.
    @param entry_spot The entry price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The entry spread in basis points between the swap and spot prices.
    """
    return (entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)) / entry_swap * 10000


@numba.jit()
def spread_exit_func_bp_numba(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the exit spread in basis points between swap and spot prices using Numba for optimization.

    This function computes the exit spread between swap and spot prices in basis points,
    taking into account the fees associated with each transaction.

    @param exit_swap The exit price for the swap.
    @param exit_spot The exit price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The exit spread in basis points between the swap and spot prices.
    """
    return (exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)) / exit_swap * 10000


def spread_entry_func(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the entry spread between swap and spot prices.

    This function computes the entry spread between swap and spot prices, taking into
    account the fees associated with each transaction.

    @param entry_swap The entry price for the swap.
    @param entry_spot The entry price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The entry spread between the swap and spot prices.
    """
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


def spread_exit_func(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the exit spread between swap and spot prices.

    This function computes the exit spread between swap and spot prices, taking into
    account the fees associated with each transaction.

    @param exit_swap The exit price for the swap.
    @param exit_spot The exit price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The exit spread between the swap and spot prices.
    """
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


def spread_entry_func_bp(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the entry spread in basis points between swap and spot prices.

    This function computes the entry spread between swap and spot prices in basis points,
    taking into account the fees associated with each transaction.

    @param entry_swap The entry price for the swap.
    @param entry_spot The entry price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The entry spread in basis points between the swap and spot prices.
    """
    return (entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)) / entry_swap * 10000


def spread_exit_func_bp(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the exit spread in basis points between swap and spot prices.

    This function computes the exit spread between swap and spot prices in basis points,
    taking into account the fees associated with each transaction.

    @param exit_swap The exit price for the swap.
    @param exit_spot The exit price for the spot.
    @param swap_fee The fee associated with the swap transaction.
    @param spot_fee The fee associated with the spot transaction.

    @return The exit spread in basis points between the swap and spot prices.
    """
    return (exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)) / exit_swap * 10000
