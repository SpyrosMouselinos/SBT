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
    A function to flatten a list (remove singleton for example)
    """
    return list(itertools.chain(*aList))



def values_to_list(params: dict = None, inputs: list = None):
    """
    Transforms specified keys in a dictionary into lists of their values, and updates the dictionary in-place.

    This function iterates over each key specified in the 'inputs' list, collects values from the 'params' dictionary
    whose keys start with the specified input, and aggregates these values into a list. Each list of aggregated values
    is then stored in the 'params' dictionary under a new key formed by appending 'List' to the original input string.
    The original keys used to aggregate values are removed from the dictionary.

    Parameters:
    - params (dict, optional): The dictionary to process. The default is None, which means an empty dictionary is used.
    - inputs (list, optional): A list of strings representing the key prefixes to look for in the 'params' dictionary.
      The function aggregates values from keys that start with any of these strings. The default is None.

    Returns:
    - dict: The modified dictionary with aggregated lists of values for specified keys. Original keys that were used
      to aggregate values are removed, and new keys (each original key prefix with 'List' appended) are added to include
      the aggregated lists.

    Example:
    >>> params = {'temp1': 20, 'temp2': 22, 'humidity': 80}
    >>> inputs = ['temp']
    >>> values_to_list(params, inputs)
    {'humidity': 80, 'tempList': [20, 22]}

    Note:
    - The function modifies the 'params' dictionary in-place.
    - If 'inputs' is None or an empty list, 'params' will be returned unchanged.
    - It is assumed that the 'params' dictionary initially does not have keys that end with 'List' which are intended
      for use by this function. If such keys exist, their values might be overwritten or merged unexpectedly.
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
    agr_volume = [df.loc[0, 'volume']]
    for ix in df.index:
        if ix != 0:
            if df.loc[ix - 1, 'price'] == df.loc[ix, 'price']:
                agr_volume.append(agr_volume[ix - 1] + df.loc[ix, 'volume'])
            else:
                agr_volume.append(df.loc[ix, 'volume'])
    return agr_volume


def parse_args(parser):
    args = parser.parse_args()
    temp_args = parser.parse_args()
    for arg in vars(temp_args):
        if hasattr(args, arg) and getattr(args, arg) == 'delete':
            args.__delattr__(arg)
    return args


def sharpe_sortino_ratio_fun(df, aum, t_start, t_end):
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
    return bp / 10000


def bp_to_dollars(bp, dollar_value):
    return to_decimal(bp) * dollar_value


class Util:

    @staticmethod
    def add_data_orderbook(book, data, price_change_step):
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
        end = start + bincount[counter]
        # movements in a 50 ms window
        snap = data[start:end, :]

        self.add_data_orderbook(book, snap, 50)
        start = end  # len(np.where(np.array(book.askVolumes()) == 0)[0]) > 0 and len(np.where(np.array(book.askVolumes()) == 0)[0]) < 4 and counter > 1000
        counter += 1
        return book.bidPrices(), book.bidVolumes(), book.askPrices(), book.askVolumes(), start, end, counter

    @staticmethod
    def create_array_from_generator(gen):
        r = np.array([x for x in gen])
        data = []
        for dic in r:
            data.append(list(dic.values()))
        data = np.array(data)
        return data

    @staticmethod
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 100.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    @staticmethod
    def remove_digits(string):
        return ''.join(i for i in string if not i.isdigit())

    @staticmethod
    def drop_nans(array):
        array_float = array.astype("float64")
        if len(array_float.shape) == 2:
            mask = ~np.isnan(array_float[:, 1])
        else:
            mask = ~np.isnan(array_float)
        return array_float[mask]

    @staticmethod
    def remove_null_values_in_array(array):
        try:
            index_non_null_value = np.where(array[:, 1] == None)[0][-1] + 1
            array[:index_non_null_value, 1:] = array[index_non_null_value, 1:]
            return array
        except IndexError:
            return array

    @staticmethod
    def get_digits(string):
        return int(''.join(i for i in string if i.isdigit()))

    @staticmethod
    def floor_to_hour(timestamp):
        return int(np.floor(timestamp / (1000 * 60 * 60)) * 1000 * 60 * 60)

    @staticmethod
    def round_to_day(timestamp):
        return int(np.round(timestamp / (1000 * 60 * 60 * 24)) * 1000 * 60 * 60 * 24)

    @staticmethod
    def get_filters_dictionary(filters_string):
        filters_list = filters_string.replace("self.", "").replace("filter_", "").split(" * ")
        return {Util.remove_digits(x).replace("()", ""): Util.get_digits(x) for x in filters_list}

    @staticmethod
    def find_local_wandb_run_folder(wandb_directory, run_id):
        run_names = os.listdir(wandb_directory)
        names_with_run_id = [run_name for run_name in run_names if run_id in run_name]
        if len(names_with_run_id) == 1:
            return os.path.join(wandb_directory, names_with_run_id[0])
        # @ TODO deal with multiple runs with the same id

    @staticmethod
    def load_processor_from_artifact(current_run_from_api):
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
        with open(os.path.join(dir, f'processor_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(processor, fid)
        artifact = initialised_wandb.Artifact('processor', type='processor')
        artifact.add_file(os.path.join(dir, f'processor_{initialised_wandb.run.id}.pkl'))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"processor_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def load_model_from_artifact(current_run_from_api):
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
        with open(os.path.join(dir, f'writer_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(writer, fid)
        artifact = initialised_wandb.Artifact('writer', type='writer')
        artifact.add_file(os.path.join(dir, f"writer_{initialised_wandb.run.id}.pkl"))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"writer_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def load_data_evaluation_loader_from_artifact(current_run_from_api):
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
        with open(os.path.join(dir, f'data_evaluation_loader_{initialised_wandb.run.id}.pkl'), 'wb') as fid:
            pickle.dump(data_evaluation_loader, fid)
        artifact = initialised_wandb.Artifact('data_evaluation_loader', type='data_evaluation_loader')
        artifact.add_file(os.path.join(dir, f"data_evaluation_loader_{initialised_wandb.run.id}.pkl"))
        initialised_wandb.log_artifact(artifact)
        os.remove(os.path.join(dir, f"data_evaluation_loader_{initialised_wandb.run.id}.pkl"))

    @staticmethod
    def update_table_artifact(current_run_from_api, current_run_resumed, data, remove_old_versions=True):
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
        # Artifacts can be deleted, but not the latest version
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
        if 'artifacts' in os.listdir(dir):
            shutil.rmtree('artifacts')

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if os.getenv("ENV") in ['staging', 'prod']:
            logger.addHandler(logstash.TCPLogstashHandler('logstash', 5000, version=1))
        return logger

    @staticmethod
    def get_time_string_from_timestamp(timestamp):
        # Timestamp expected to be in milliseconds
        return datetime.datetime.fromtimestamp(timestamp / 1000, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def indices_to_one_hot(data, nb_classes=None):
        """Convert an iterable of indices to one-hot encoded labels."""
        data_int = data.astype(np.int64)
        nb_classes = nb_classes if nb_classes else data_int.max() + 1
        one_hot = np.zeros((data_int.size, nb_classes))
        one_hot[np.arange(data_int.size), data_int] = 1
        return one_hot[:]

    @staticmethod
    def unison_shuffled_copies(a, b, c, d):
        """
        Shuffles the elements of the arrays in unison

        :param a: numpy array
        :param b: numpy array
        :param c: numpy array
        :param d: numpy array
        :return: shuffled arrays "a", "b", "c", "d"
        """
        assert len(a) == len(b)
        assert len(a) == len(c)
        assert len(a) == len(d)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p]

    @staticmethod
    def unison_shuffled_copies_2(a, b):
        """
        Shuffles the elements of the arrays in unison

        :param a: numpy array
        :param b: numpy array
        :return: shuffled arrays "a", "b"
        """
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def organize_into_bins(data, t0, t1, step):

        bins = np.linspace(step, t1 - (t0 + step), num=int((t1 - (t0 + step)) / step))
        data = np.array(data[:, 0], dtype=np.int64)
        bins = (bins + t0).astype(np.int64)

        # returns the indices of the bins to which each value in input array belongs
        inds = np.digitize(data, bins)

        return inds

    @staticmethod
    def organize_into_bins_vector(data, bins, t0):
        bins = (bins + t0)
        inds = np.searchsorted(bins, data, side='right')

        return inds

    @staticmethod
    def create_array_from_generator(gen):
        # parse all the elements in the generator
        r = np.array([x for x in gen])

        data = []

        for dic in r:
            data.append(list(dic.values()))
        data = np.array(data)

        return data

    @staticmethod
    def create_array_from_query(array):

        try:
            result = np.array(array.raw['series'][0]['values'])
        except (ValueError, IndexError, KeyError):
            result = np.array([])
        return result

    @staticmethod
    def time_plus_one_hour(time):
        hour = 1000 * 60 * 60
        return time + hour

    @staticmethod
    def time_plus_one_day(time):
        day = 1000 * 60 * 60 * 24
        return time + day

    @staticmethod
    def normalize(values, min, max):
        return (values - min) / (max - min)

    @staticmethod
    def normalize_inverse(values, min, max):
        return values * (max - min) + min

    @staticmethod
    def standardize(values, mean, stddev):
        return (values - mean) / stddev

    @staticmethod
    def standardize_inverse(values, mean, stddev):
        return values * stddev + mean

    @staticmethod
    def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
        """Create a view of `array` which for every point gives the n-dimensional
        neighbourhood of size window. New dimensions are added at the end of
        `array` or after the corresponding original dimension.

        Parameters
        ----------
        array : array_like
            Array to which the rolling window is applied.
        window : int or tuple
            Either a single integer to create a window of only the last axis or a
            tuple to create it for the last len(window) axes. 0 can be used as a
            to ignore a dimension in the window.
        asteps : tuple
            Aligned at the last axis, new steps for the original array, ie. for
            creation of non-overlapping windows. (Equivalent to slicing result)
        wsteps : int or tuple (same size as window)
            steps for the added window dimensions. These can be 0 to repeat values
            along the axis.
        axes: int or tuple
            If given, must have the same size as window. In this case window is
            interpreted as the size in the dimension given by axes. IE. a window
            of (2, 1) is equivalent to window=2 and axis=-2.
        toend : bool
            If False, the new dimensions are right after the corresponding original
            dimension, instead of at the end of the array. Adding the new axes at the
            end makes it easier to get the neighborhood, however toend=False will give
            a more intuitive result if you view the whole array.

        Returns
        -------
        A view on `array` which is smaller to fit the windows and has windows added
        dimensions (0s not counting), ie. every point of `array` is an array of size
        window.

        Examples
        --------
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

        Note that the 0 is discared, so that the output dimension is 3:
        >>> rolling_window(a, (2,0), asteps=(2,1)).shape
        (1, 3, 2)

        This is useful for example to calculate the maximum in all (overlapping)
        2x2 submatrixes:
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
            raise ValueError("All elements of `window` must be larger then 1.")
        if len(array.shape) < len(window):
            raise ValueError("`window` length must be less or equal `array` dimension.")

        _asteps = np.ones_like(orig_shape)
        if asteps is not None:
            asteps = np.atleast_1d(asteps)
            if asteps.ndim != 1:
                raise ValueError("`asteps` must be either a scalar or one dimensional.")
            if len(asteps) > array.ndim:
                raise ValueError("`asteps` cannot be longer then the `array` dimension.")
            # does not enforce alignment, so that steps can be same as window too.
            _asteps[-len(asteps):] = asteps

            if np.any(asteps < 1):
                raise ValueError("All elements of `asteps` must be larger then 1.")
        asteps = _asteps

        _wsteps = np.ones_like(window)
        if wsteps is not None:
            wsteps = np.atleast_1d(wsteps)
            if wsteps.shape != window.shape:
                raise ValueError("`wsteps` must have the same shape as `window`.")
            if np.any(wsteps < 0):
                raise ValueError("All elements of `wsteps` must be larger then 0.")

            _wsteps[:] = wsteps
            _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
        wsteps = _wsteps

        # Check that the window would not be larger then the original:
        if np.any(orig_shape[-len(window):] < window * wsteps):
            raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

        new_shape = orig_shape  # just renaming...

        # For calculating the new shape 0s must act like 1s:
        _window = window.copy()
        _window[_window == 0] = 1

        new_shape[-len(window):] += wsteps - _window * wsteps
        new_shape = (new_shape + asteps - 1) // asteps
        # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
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
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    done = True
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            done = False
            return j
    if done:
        return -1


@functools.lru_cache(maxsize=3)
def cached_get_loc(a, v, side):
    return a.get_loc(v)


@numba.jit(nopython=True)
def pop_old(l, timestamp, interval):
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
    count = 0
    for j in range(len(l) - 1, -1, -1):
        if l[j] > timestamp - interval:
            count += 1
        else:
            return count
    return count


import numba


@numba.jit()
def spread_entry_func_numba(entry_swap, entry_spot, swap_fee, spot_fee):
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


@numba.jit()
def spread_exit_func_numba(exit_swap, exit_spot, swap_fee, spot_fee):
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


@numba.jit()
def spread_entry_func_bp_numba(entry_swap, entry_spot, swap_fee, spot_fee):
    return (entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)) / entry_swap * 10000


@numba.jit()
def spread_exit_func_bp_numba(exit_swap, exit_spot, swap_fee, spot_fee):
    return (exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)) / exit_swap * 10000


def spread_entry_func(entry_swap, entry_spot, swap_fee, spot_fee):
    return entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)


def spread_exit_func(exit_swap, exit_spot, swap_fee, spot_fee):
    return exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)


def spread_entry_func_bp(entry_swap, entry_spot, swap_fee, spot_fee):
    return (entry_swap * (1 - swap_fee) - entry_spot * (1 + spot_fee)) / entry_swap * 10000


def spread_exit_func_bp(exit_swap, exit_spot, swap_fee, spot_fee):
    return (exit_swap * (1 + swap_fee) - exit_spot * (1 - spot_fee)) / exit_swap * 10000

