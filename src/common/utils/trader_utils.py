import numba


def get_spread_entry(swap_price, spot_price, swap_fee, spot_fee):
    """
    Calculate the spread entry value for maker-maker, balancing, or taker-maker spreads.

    This function can be used to compute the maker-maker spread, balancing spread, or taker-maker spread, 
    depending on the side and fee signs. The convention is that fees are always paid, thus maker rebates need to be negative.

    @param swap_price The price of the swap.
    @param spot_price The price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed spread entry value.
    """
    return swap_price - spot_price - swap_price * swap_fee - spot_price * spot_fee


def get_spread_exit(swap_price, spot_price, swap_fee, spot_fee):
    """
    Calculate the spread exit value for maker-maker, balancing, or taker-maker spreads.

    This function can be used to compute the maker-maker spread, balancing spread, or taker-maker spread, 
    depending on the side and fee signs. The convention is that fees are always paid, thus maker rebates need to be negative.

    @param swap_price The price of the swap.
    @param spot_price The price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed spread exit value.
    """
    return swap_price - spot_price + swap_price * swap_fee + spot_price * spot_fee


def get_maker_maker_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the maker-maker spread entry value.

    @param entry_swap The entry price of the swap.
    @param entry_spot The entry price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed maker-maker spread entry value.
    """
    return entry_swap - entry_spot - entry_swap * swap_fee - entry_spot * spot_fee


def get_maker_maker_balancing_spread_entry(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Calculate the maker-maker balancing spread entry value.

    @param entry_swap The entry price of the swap.
    @param entry_spot The entry price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed maker-maker balancing spread entry value.
    """
    return entry_swap - entry_spot - entry_swap * swap_fee - entry_spot * spot_fee


def get_maker_maker_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the maker-maker spread exit value.

    @param exit_swap The exit price of the swap.
    @param exit_spot The exit price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed maker-maker spread exit value.
    """
    return exit_swap - exit_spot + exit_swap * swap_fee + exit_spot * spot_fee


def get_maker_maker_balancing_spread_exit(exit_swap, exit_spot, swap_fee, spot_fee):
    """
    Calculate the maker-maker balancing spread exit value.

    @param exit_swap The exit price of the swap.
    @param exit_spot The exit price of the spot.
    @param swap_fee The fee for the swap transaction. (Positive for fees, negative for rebates)
    @param spot_fee The fee for the spot transaction. (Positive for fees, negative for rebates)

    @return The computed maker-maker balancing spread exit value.
    """
    return exit_swap - exit_spot + exit_swap * swap_fee + exit_spot * spot_fee


@numba.jit(nopython=True)
def shift_array_possible_extension(array, shifted_array, amount):
    """
    Shift the input array by a specified amount and populate the shifted array.

    This function shifts the elements of the input array to the right by a specified amount, 
    filling in the shifted array with the corresponding values. If no suitable value is found 
    within the specified amount, the original value is copied.

    @param array The input array to be shifted.
    @param shifted_array The array to store the shifted values.
    @param amount The amount by which to shift the array.

    @return The shifted array with values adjusted according to the specified amount.
    """
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
    """
    Perform calculations on a NumPy array of matrix data to update entry and exit area spreads.

    This function iterates over the input matrix and updates entry and exit area spreads based on 
    differences between spread values and the corresponding entry/exit bands.

    @param df_mat A NumPy array of matrix data containing spread values, bands, and multipliers.

    @return The updated NumPy array with calculated entry and exit area spreads.
    """
    for idx in range(1, len(df_mat) - 1):
        if df_mat[idx, 0] >= df_mat[idx, 2]:
            df_mat[idx, 5] = df_mat[idx - 1, 5] + abs(df_mat[idx, 0] - df_mat[idx, 2]) * df_mat[idx, 4]

        if df_mat[idx, 1] <= df_mat[idx, 3]:
            df_mat[idx, 6] = df_mat[idx - 1, 6] + abs(df_mat[idx, 1] - df_mat[idx, 3]) * df_mat[idx, 4]
    return df_mat


@numba.jit(nopython=True)
def get_index_left(timestamps, current_index, latency):
    """
    Find the index of the first element in a sorted array that is less than the current element minus latency.

    @param timestamps A sorted array of timestamps.
    @param current_index The index of the current element.
    @param latency The latency value to compare against.

    @return The index of the first element that satisfies the condition.
    """
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def shift_array(array, shifted_array, amount):
    """
    Shift the elements of the input array by a specified amount and populate the shifted array.

    This function shifts the elements of the input array to the right by a specified amount, 
    filling in the shifted array with the corresponding values. If no suitable value is found 
    within the specified amount, the original value is copied.

    @param array The input array to be shifted.
    @param shifted_array The array to store the shifted values.
    @param amount The amount by which to shift the array.

    @return The shifted array with values adjusted according to the specified amount.
    """
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
    """
    Find the index of the first element in a sorted array that is less than the current element minus latency.

    @param timestamps A sorted array of timestamps.
    @param current_index The index of the current element.
    @param latency The latency value to compare against.

    @return The index of the first element that satisfies the condition.
    """
    for j in range(current_index, -1, -1):
        if timestamps[j] < timestamps[current_index] - latency:
            return j


@numba.jit(nopython=True)
def get_index_right(timestamps, current_index, latency):
    """
    Find the index of the first element in a sorted array that is greater than the current element plus latency.

    @param timestamps A sorted array of timestamps.
    @param current_index The index of the current element.
    @param latency The latency value to compare against.

    @return The index of the first element that satisfies the condition.
    """
    for j in range(current_index, len(timestamps)):
        if timestamps[j] > timestamps[current_index] + latency:
            return j
    return len(timestamps) - 1


@numba.jit(nopython=True)
def shift_array(array, shifted_array, amount):
    """
    Shift the elements of the input array by a specified amount and populate the shifted array.

    This function shifts the elements of the input array to the right by a specified amount, 
    filling in the shifted array with the corresponding values. If no suitable value is found 
    within the specified amount, the original value is copied.

    @param array The input array to be shifted.
    @param shifted_array The array to store the shifted values.
    @param amount The amount by which to shift the array.

    @return The shifted array with values adjusted according to the specified amount.
    """
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
