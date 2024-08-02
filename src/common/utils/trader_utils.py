import numba

def get_spread_entry(swap_price, spot_price, swap_fee, spot_fee):
    # This function can be used to compute both maker-maker spread, balancing spread and taker-maker spread, depending
    # on the side and fee signs
    # The convention is that fees are always paid, thus maker rebates need to be negative
    return swap_price - spot_price - swap_price * swap_fee - spot_price * spot_fee


def get_spread_exit(swap_price, spot_price, swap_fee, spot_fee):
    # This function can be used to compute both maker-maker spread, balancing spread and taker-maker spread, depending
    # on the side and fee signs
    # The convention is that fees are always paid, thus maker rebates need to be negative
    return swap_price - spot_price + swap_price * swap_fee + spot_price * spot_fee


def get_maker_maker_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    return entry_swap - entry_spot - entry_swap * swap_fee - entry_spot * spot_fee


def get_maker_maker_balancing_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    return entry_swap - entry_spot - entry_swap * swap_fee - entry_spot * spot_fee


def get_maker_maker_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    return exit_swap - exit_spot + exit_swap * swap_fee + exit_spot * spot_fee


def get_maker_maker_balancing_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    return exit_swap - exit_spot + exit_swap * swap_fee + exit_spot * spot_fee



@numba.jit(nopython=True)
def shift_array_possible_extension(array, shifted_array, amount):
    for j in range(len(array)):
        found = False
        for k in range(j, -1, -1):
            if array[k, 0] <= array[j, 0] - amount:
                for i in range(len(shifted_array[0])):
                    if i == 0:
                        shifted_array[j, i] = float(array[j, 0])
                    else:
                        shifted_array[j, i] = float(array[k, i])
                found = True
                break
        if not found:
            for i in range(len(shifted_array[0])):
                shifted_array[j, i] = float(array[j, i])
    return shifted_array


@numba.jit(nopython=True)
def df_numba(df_mat):
    for idx in range(1, len(df_mat) - 1):
        if df_mat[idx, 0] >= df_mat[idx, 2]:
            df_mat[idx, 5] = df_mat[idx - 1, 5] + abs(df_mat[idx, 0] - df_mat[idx, 2]) * df_mat[idx, 4]

        if df_mat[idx, 1] <= df_mat[idx, 3]:
            df_mat[idx, 6] = df_mat[idx - 1, 6] + abs(df_mat[idx, 1] - df_mat[idx, 3]) * df_mat[idx, 4]
    return df_mat


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def shift_array(array, shifted_array, amount):
    for j in range(len(array)):
        found = False
        for k in range(j, -1, -1):
            if array[k, 0] < array[j, 0] - amount:
                shifted_array[j, 0] = float(array[j, 0])
                shifted_array[j, 1] = float(array[k, 1])
                shifted_array[j, 2] = float(array[k, 2])
                found = True
                break
        if not found:
            shifted_array[j, 0] = float(array[j, 0])
            shifted_array[j, 1] = float(array[j, 1])
            shifted_array[j, 2] = float(array[j, 2])
    return shifted_array


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            return j
    return len(timestamps) - 1


@numba.jit(nopython=True)
def shift_array(array, shifted_array, amount):
    for j in range(len(array)):
        found = False
        for k in range(j, -1, -1):
            if array[k, 0] < array[j, 0] - amount:
                shifted_array[j, 0] = float(array[j, 0])
                shifted_array[j, 1] = float(array[k, 1])
                shifted_array[j, 2] = float(array[k, 2])
                found = True
                break
        if not found:
            shifted_array[j, 0] = float(array[j, 0])
            shifted_array[j, 1] = float(array[j, 1])
            shifted_array[j, 2] = float(array[j, 2])
    return shifted_array
