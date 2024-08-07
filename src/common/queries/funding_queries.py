import os
from dataclasses import dataclass
from datetime import time
from typing import Dict

import numba
import numpy as np
import pandas as pd

from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.constants.constants import one_hour, one_second, five_minutes


@dataclass
class FundingRatiosParams:
    """
    @brief A data class for storing funding ratio parameters.

    This class encapsulates the parameters for funding ratios used in trading strategies. It includes
    entry and exit ratios, as well as ratios to zero, and provides methods for comparison and negation.

    @param ratio_to_zero_entry The ratio used for entry funding to zero.
    @param ratio_to_zero_exit The ratio used for exit funding to zero.
    @param ratio_entry The ratio used for entry funding.
    @param ratio_exit The ratio used for exit funding.
    """

    ratio_to_zero_entry: int = None
    ratio_to_zero_exit: int = None
    ratio_entry: int = None
    ratio_exit: int = None

    def __bool__(self):
        """
        @brief Determines if the object is considered True based on its attributes.

        This method returns True if the `ratio_to_zero_entry` attribute is not None, indicating that
        the object is valid for use in funding calculations.

        @return True if `ratio_to_zero_entry` is not None, otherwise False.
        """
        if self.ratio_to_zero_entry is not None:
            return True
        return False

    def __eq__(self, other):
        """
        @brief Checks equality between two FundingRatiosParams objects.

        This method compares all attributes of two FundingRatiosParams objects to determine if they
        are equal.

        @param other The other FundingRatiosParams object to compare against.
        @return True if all attributes are equal, otherwise False.
        """
        return (self.ratio_to_zero_entry == other.ratio_to_zero_entry) and \
            (self.ratio_to_zero_exit == other.ratio_to_zero_exit) and \
            (self.ratio_entry == other.ratio_entry) and \
            (self.ratio_exit == other.ratio_exit)

    def __neg__(self):
        """
        @brief Negates the values of the funding ratios.

        This method returns a new FundingRatiosParams object with all ratio values negated.

        @return A new FundingRatiosParams object with negated ratio values.
        """
        return FundingRatiosParams(ratio_to_zero_entry=-self.ratio_to_zero_entry,
                                   ratio_to_zero_exit=-self.ratio_to_zero_exit,
                                   ratio_entry=-self.ratio_entry,
                                   ratio_exit=-self.ratio_exit)


class FundingBase:
    """
    @brief A base class for managing funding data.

    This class provides basic functionality for managing funding data, including updating and
    retrieving funding values based on a given timestamp.

    @param fundings A numpy array containing funding data, where each row represents a funding event
    with a timestamp and a value.
    """

    ms_column = 0
    value_column = 1
    idx = 0

    def __init__(self, fundings):
        """
        @brief Initializes the FundingBase class with funding data.

        This constructor sets up the funding data and initializes the current funding value to zero.

        @param fundings A numpy array containing funding data.
        """
        self.fundings = fundings
        self.current_funding = 0

    def update(self, timestamp):
        """
        @brief Updates the current funding value based on the given timestamp.

        This method searches the funding data for the entry corresponding to the provided timestamp
        and updates the current funding value accordingly.

        @param timestamp The timestamp to search for in the funding data.
        """
        self.idx = np.searchsorted(self.fundings[:, self.ms_column], timestamp, side='right')
        self.current_funding = self.fundings[self.idx, self.value_column]

    def get_next_funding(self):
        """
        @brief Retrieves the next funding value.

        This method returns the current funding value, which is updated based on the timestamp provided
        in the `update` method.

        @return The current funding value.
        """
        return self.current_funding

    def get_next_funding_entry(self):
        """
        @brief Retrieves the next funding value for entry.

        This method returns the current funding value, which is used for entry funding calculations.

        @return The current funding value for entry.
        """
        return self.get_next_funding()

    def get_next_funding_to_zero_entry(self):
        """
        @brief Retrieves the next funding value to zero for entry.

        This method returns the current funding value, which is used for entry funding to zero calculations.

        @return The current funding value to zero for entry.
        """
        return self.get_next_funding()

    def get_next_funding_exit(self):
        """
        @brief Retrieves the next funding value for exit.

        This method returns the current funding value, which is used for exit funding calculations.

        @return The current funding value for exit.
        """
        return self.get_next_funding()

    def get_next_funding_to_zero_exit(self):
        """
        @brief Retrieves the next funding value to zero for exit.

        This method returns the current funding value, which is used for exit funding to zero calculations.

        @return The current funding value to zero for exit.
        """
        return self.get_next_funding()

    def get_predicted_funding(self):
        """
        @brief Placeholder for predicted funding calculations.

        This method is intended to be overridden by subclasses that implement predicted funding calculations.

        @return None
        """
        return


class FundingBaseWithRatios(FundingBase):
    """
    @brief A class for managing funding data with ratios.

    This class extends FundingBase to include functionality for managing funding data with
    entry and exit ratios, allowing for more precise funding calculations.

    @param fundings A numpy array containing funding data.
    @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
    """

    ms_column = 0
    value_column = 1
    idx = 0

    def __init__(self, fundings, funding_ratios: FundingRatiosParams):
        """
        @brief Initializes the FundingBaseWithRatios class with funding data and ratios.

        This constructor sets up the funding data and funding ratios, initializing the current
        funding value to zero.

        @param fundings A numpy array containing funding data.
        @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
        """
        super(FundingBaseWithRatios, self).__init__(fundings)
        self.funding_ratios = funding_ratios

    def update(self, timestamp):
        """
        @brief Updates the current funding value based on the given timestamp and funding ratios.

        This method searches the funding data for the entry corresponding to the provided timestamp,
        updates the current funding value, and applies the appropriate funding ratios.

        @param timestamp The timestamp to search for in the funding data.
        """
        self.idx = np.searchsorted(self.fundings[:, self.ms_column], timestamp, side='right')
        if self.idx >= len(self.fundings):
            self.current_funding = 0
            return
        self.current_funding = self.fundings[self.idx, self.value_column]

    def get_next_funding(self):
        """
        @brief Retrieves the next funding value.

        This method returns the current funding value, which is updated based on the timestamp provided
        in the `update` method.

        @return The current funding value.
        """
        return self.current_funding

    def get_next_funding_entry(self):
        """
        @brief Retrieves the next funding value for entry, adjusted by the entry ratio.

        This method returns the current funding value multiplied by the entry ratio specified in the
        funding ratios.

        @return The current funding value for entry, adjusted by the entry ratio.
        """
        return self.get_next_funding() * self.funding_ratios.ratio_entry

    def get_next_funding_to_zero_entry(self):
        """
        @brief Retrieves the next funding value to zero for entry, adjusted by the zero entry ratio.

        This method returns the current funding value multiplied by the zero entry ratio specified in
        the funding ratios.

        @return The current funding value to zero for entry, adjusted by the zero entry ratio.
        """
        return self.get_next_funding() * self.funding_ratios.ratio_to_zero_entry

    def get_next_funding_exit(self):
        """
        @brief Retrieves the next funding value for exit, adjusted by the exit ratio.

        This method returns the current funding value multiplied by the exit ratio specified in the
        funding ratios.

        @return The current funding value for exit, adjusted by the exit ratio.
        """
        return self.get_next_funding() * self.funding_ratios.ratio_exit

    def get_next_funding_to_zero_exit(self):
        """
        @brief Retrieves the next funding value to zero for exit, adjusted by the zero exit ratio.

        This method returns the current funding value multiplied by the zero exit ratio specified in
        the funding ratios.

        @return The current funding value to zero for exit, adjusted by the zero exit ratio.
        """
        return self.get_next_funding() * self.funding_ratios.ratio_to_zero_exit

    def get_predicted_funding(self):
        """
        @brief Placeholder for predicted funding calculations.

        This method is intended to be overridden by subclasses that implement predicted funding calculations.

        @return None
        """
        return


class FundingBinanceDiscounted(FundingBase):
    """
    @brief A class for managing discounted funding data from Binance.

    This class extends FundingBase to apply a discount to funding values based on a specified
    weight and funding interval, as defined by Binance.

    @param fundings A numpy array containing funding data.
    """

    funding_interval = 8 * one_hour
    max_weight = 5760  # the n in the formula of the TWA. From Binance

    def __init__(self, fundings):
        """
        @brief Initializes the FundingBinanceDiscounted class with funding data.

        This constructor sets up the funding data and initializes the weight for discount calculations.

        @param fundings A numpy array containing funding data.
        """
        super(FundingBinanceDiscounted, self).__init__(fundings)
        self.weight = 1

    def update(self, timestamp):
        """
        @brief Updates the current funding value and weight based on the given timestamp.

        This method calculates the weight for discounting based on the timestamp and funding interval,
        then updates the current funding value.

        @param timestamp The timestamp to search for in the funding data.
        """
        super(FundingBinanceDiscounted, self).update(timestamp)
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight

    def get_next_funding(self):
        """
        @brief Retrieves the next discounted funding value.

        This method returns the current funding value multiplied by the calculated weight.

        @return The current discounted funding value.
        """
        return self.current_funding * self.weight


class FundingBinanceDiscountedWithRatios(FundingBaseWithRatios):
    """
    @brief A class for managing discounted funding data with ratios from Binance.

    This class extends FundingBaseWithRatios to apply a discount to funding values based on a specified
    weight and funding interval, as well as funding ratios for entry and exit.

    @param fundings A numpy array containing funding data.
    @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
    """

    funding_interval = 8 * one_hour
    max_weight = 5760  # the n in the formula of the TWA. From Binance

    def __init__(self, fundings, funding_ratios: FundingRatiosParams):
        """
        @brief Initializes the FundingBinanceDiscountedWithRatios class with funding data and ratios.

        This constructor sets up the funding data and funding ratios, initializing the weight for discount calculations.

        @param fundings A numpy array containing funding data.
        @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
        """
        super(FundingBinanceDiscountedWithRatios, self).__init__(fundings, funding_ratios)
        self.weight = 1

    def update(self, timestamp):
        """
        @brief Updates the current funding value, weight, and applies funding ratios based on the given timestamp.

        This method calculates the weight for discounting based on the timestamp and funding interval,
        updates the current funding value, and applies the appropriate funding ratios.

        @param timestamp The timestamp to search for in the funding data.
        """
        super(FundingBinanceDiscountedWithRatios, self).update(timestamp)
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight

    def get_next_funding(self):
        """
        @brief Retrieves the next discounted funding value.

        This method returns the current funding value multiplied by the calculated weight and adjusted by the funding
        ratios.

        @return The current discounted funding value.
        """
        return super(FundingBinanceDiscountedWithRatios, self).get_next_funding() * self.weight


class FundingBinanceDiscountedWithRatiosModel(FundingBaseWithRatios):
    """
    @brief A class for managing predicted discounted funding data with ratios from Binance.

    This class extends FundingBaseWithRatios to apply a discount to funding values based on a specified
    weight, funding interval, and funding ratios. It also incorporates a prediction model for estimating
    future funding values.

    @param fundings A numpy array containing funding data.
    @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
    @param prediction_emitter A prediction model used to estimate future funding values.
    """

    funding_interval = 8 * one_hour
    max_weight = 5760  # the n in the formula of the TWA. From Binance

    def __init__(self, fundings, funding_ratios: FundingRatiosParams, prediction_emitter):
        """
        @brief Initializes the FundingBinanceDiscountedWithRatiosModel class with funding data, ratios, and a prediction model.

        This constructor sets up the funding data, funding ratios, and prediction model, initializing the weight for
        discount calculations.

        @param fundings A numpy array containing funding data.
        @param funding_ratios An instance of FundingRatiosParams specifying the funding ratios.
        @param prediction_emitter A prediction model used to estimate future funding values.
        """
        super(FundingBinanceDiscountedWithRatiosModel, self).__init__(fundings, funding_ratios)
        self.weight = 1
        self.prediction_emitter = prediction_emitter

    def update(self, timestamp):
        """
        @brief Updates the current funding value, weight, and applies the prediction model based on the given timestamp.

        This method calculates the weight for discounting based on the timestamp and funding interval, updates the
        current funding value using the prediction model, and applies the appropriate funding ratios.

        @param timestamp The timestamp to search for in the funding data.
        """
        timestamp_previous_funding = timestamp // self.funding_interval * self.funding_interval
        self.weight = (timestamp - timestamp_previous_funding) // (5 * one_second) / self.max_weight
        self.current_funding = self.prediction_emitter.predict(timestamp)

    def get_next_funding(self):
        """
        @brief Retrieves the next predicted discounted funding value.

        This method returns the predicted funding value adjusted by the funding ratios.

        @return The predicted discounted funding value.
        """
        return super(FundingBinanceDiscountedWithRatiosModel, self).get_next_funding()  # * self.weight


class FundingSystemEmpty:
    """
    @brief An empty funding system class used as a placeholder or base for other systems.

    This class provides basic structure for a funding system with methods for updating and adjusting
    trading bands based on funding data. It serves as a template or no-op implementation.
    """

    name = ""
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBase):
        """
        @brief Initializes the FundingSystemEmpty with spot and swap funding data.

        This constructor sets up the spot and swap funding objects and initializes the timestamp for the last update.

        @param funding_spot The spot funding data object.
        @param funding_swap The swap funding data object.
        """
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        """
        @brief Updates the funding system based on the given timestamp.

        This method is intended to be overridden by subclasses with specific update logic. The base implementation
        does nothing.

        @param timestamp The current timestamp for updating the funding system.
        """
        pass

    def band_adjustments(self):
        """
        @brief Calculates band adjustments for trading.

        This method returns adjustments for entry and exit bands based on funding data. In this empty implementation,
        it returns zero for both adjustments.

        @return A tuple containing entry and exit band adjustments.
        """
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def entry_band_adjustment(self):
        """
        @brief Calculates the entry band adjustment.

        This method returns the adjustment for the entry band. In this empty implementation, it returns zero.

        @return The entry band adjustment, which is zero in this implementation.
        """
        return 0

    def exit_band_adjustment(self):
        """
        @brief Calculates the exit band adjustment.

        This method returns the adjustment for the exit band. In this empty implementation, it returns zero.

        @return The exit band adjustment, which is zero in this implementation.
        """
        return 0

    def entry_band_adjustment_to_zero(self):
        """
        @brief Calculates the entry band adjustment to zero.

        This method returns the adjustment to bring the entry band to zero. In this empty implementation, it returns zero.

        @return The entry band adjustment to zero, which is zero in this implementation.
        """
        return 0

    def exit_band_adjustment_to_zero(self):
        """
        @brief Calculates the exit band adjustment to zero.

        This method returns the adjustment to bring the exit band to zero. In this empty implementation, it returns zero.

        @return The exit band adjustment to zero, which is zero in this implementation.
        """
        return 0


class FundingSystemDeribitBitMEXWithRatios:
    """
    @brief A funding system class for Deribit and BitMEX with funding ratios.

    This class calculates trading band adjustments based on funding data with specific ratios for
    Deribit and BitMEX exchanges.
    """

    name = "deribit_bitmex_with_ratios"
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBaseWithRatios, funding_swap: FundingBaseWithRatios):
        """
        @brief Initializes the FundingSystemDeribitBitMEXWithRatios with spot and swap funding data.

        This constructor sets up the spot and swap funding objects and initializes the timestamp for the last update.

        @param funding_spot The spot funding data object with ratios.
        @param funding_swap The swap funding data object with ratios.
        """
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        """
        @brief Updates the funding system based on the given timestamp.

        This method updates both the spot and swap funding data and sets the last update timestamp.

        @param timestamp The current timestamp for updating the funding system.
        """
        self.funding_spot.update(timestamp)
        self.funding_swap.update(timestamp)
        self.timestamp_last_update = timestamp

    def band_adjustments(self):
        """
        @brief Calculates band adjustments for trading.

        This method returns adjustments for entry and exit bands based on the current funding data.

        @return A tuple containing entry and exit band adjustments.
        """
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def band_adjustments_to_zero(self):
        """
        @brief Calculates band adjustments to zero for trading.

        This method returns adjustments to bring both entry and exit bands to zero based on the current funding data.

        @return A tuple containing entry and exit band adjustments to zero.
        """
        return self.entry_band_adjustment_to_zero(), self.exit_band_adjustment_to_zero()

    def entry_band_adjustment(self):
        """
        @brief Calculates the entry band adjustment.

        This method calculates the adjustment for the entry band based on the funding entries for spot and swap.

        @return The calculated entry band adjustment.
        """
        return 10000 * (self.funding_spot.get_next_funding_entry() + self.funding_swap.get_next_funding_entry())

    def exit_band_adjustment(self):
        """
        @brief Calculates the exit band adjustment.

        This method calculates the adjustment for the exit band based on the funding exits for spot and swap.

        @return The calculated exit band adjustment.
        """
        return 10000 * (self.funding_spot.get_next_funding_exit() + self.funding_swap.get_next_funding_exit())

    def entry_band_adjustment_to_zero(self):
        """
        @brief Calculates the entry band adjustment to zero.

        This method calculates the adjustment to bring the entry band to zero based on the funding entries to zero for spot and swap.

        @return The calculated entry band adjustment to zero.
        """
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_entry() + self.funding_swap.get_next_funding_to_zero_entry())

    def exit_band_adjustment_to_zero(self):
        """
        @brief Calculates the exit band adjustment to zero.

        This method calculates the adjustment to bring the exit band to zero based on the funding exits to zero for spot and swap.

        @return The calculated exit band adjustment to zero.
        """
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_exit() + self.funding_swap.get_next_funding_to_zero_exit())


class FundingSystemOkxBinanceDiscounted:
    """
    @brief A funding system class for OKX and Binance with discounted funding.

    This class calculates trading band adjustments based on funding data with specific logic for OKX
    and Binance exchanges.
    """

    name = "okx_binance_discounted"
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBase):
        """
        @brief Initializes the FundingSystemOkxBinanceDiscounted with spot and swap funding data.

        This constructor sets up the spot and swap funding objects and initializes the timestamp for the last update.

        @param funding_spot The spot funding data object.
        @param funding_swap The swap funding data object.
        """
        self.funding_spot = funding_spot
        self.funding_swap = funding_swap
        self.timestamp_last_update = 0

    def update(self, timestamp):
        """
        @brief Updates the funding system based on the given timestamp.

        This method updates both the spot and swap funding data and sets the last update timestamp.

        @param timestamp The current timestamp for updating the funding system.
        """
        self.funding_spot.update(timestamp)
        self.funding_swap.update(timestamp)
        self.timestamp_last_update = timestamp

    def band_adjustments(self):
        """
        @brief Calculates band adjustments for trading.

        This method returns adjustments for entry and exit bands based on the current funding data.

        @return A tuple containing entry and exit band adjustments.
        """
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def entry_band_adjustment(self):
        """
        @brief Calculates the entry band adjustment.

        This method calculates the adjustment for the entry band based on the funding conditions for spot and swap.

        @return The calculated entry band adjustment.
        """
        if self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() >= 0:
            return abs(self.funding_spot.get_next_funding()) * 10000
        elif self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() < 0:
            return (abs(self.funding_spot.get_next_funding()) + abs(self.funding_swap.get_next_funding())) * 10000
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() >= 0:
            return 0
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() < 0:
            return abs(self.funding_swap.get_next_funding()) * 10000

    def exit_band_adjustment(self):
        """
        @brief Calculates the exit band adjustment.

        This method calculates the adjustment for the exit band based on the funding conditions for spot and swap.

        @return The calculated exit band adjustment.
        """
        if self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() >= 0:
            # Move band down by swap funding
            return abs(self.funding_swap.get_next_funding()) * 10000
        elif self.funding_spot.get_next_funding() >= 0 and self.funding_swap.get_next_funding() < 0:
            return 0
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() >= 0:
            return (abs(self.funding_spot.get_next_funding()) + abs(self.funding_swap.get_next_funding())) * 10000
        elif self.funding_spot.get_next_funding() <= 0 and self.funding_swap.get_next_funding() < 0:
            return abs(self.funding_spot.get_next_funding()) * 10000

    def entry_band_adjustment_to_zero(self):
        """
        @brief Calculates the entry band adjustment to zero.

        This method returns zero for the adjustment to bring the entry band to zero in this implementation.

        @return The entry band adjustment to zero, which is zero in this implementation.
        """
        return 0

    def exit_band_adjustment_to_zero(self):
        """
        @brief Calculates the exit band adjustment to zero.

        This method returns zero for the adjustment to bring the exit band to zero in this implementation.

        @return The exit band adjustment to zero, which is zero in this implementation.
        """
        return 0


class FundingSystemOkxBinanceDiscoutedWithRatios(FundingSystemOkxBinanceDiscounted):
    """
    @brief A funding system class for OKX and Binance with discounted funding and ratios.

    This class extends FundingSystemOkxBinanceDiscounted to include funding ratios for calculating
    trading band adjustments.
    """

    name = "okx_binance_discounted_with_ratios"
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBinanceDiscountedWithRatios):
        """
        @brief Initializes the FundingSystemOkxBinanceDiscoutedWithRatios with spot and swap funding data.

        This constructor sets up the spot and swap funding objects, including funding ratios, and initializes the
        timestamp for the last update.

        @param funding_spot The spot funding data object.
        @param funding_swap The swap funding data object with ratios.
        """
        super(FundingSystemOkxBinanceDiscoutedWithRatios, self).__init__(funding_spot, funding_swap)

    def band_adjustments(self):
        """
        @brief Calculates band adjustments for trading.

        This method returns adjustments for entry and exit bands based on the current funding data and ratios.

        @return A tuple containing entry and exit band adjustments.
        """
        return self.entry_band_adjustment(), self.exit_band_adjustment()

    def band_adjustments_to_zero(self):
        """
        @brief Calculates band adjustments to zero for trading.

        This method returns adjustments to bring both entry and exit bands to zero based on the current funding data and ratios.

        @return A tuple containing entry and exit band adjustments to zero.
        """
        return self.entry_band_adjustment_to_zero(), self.exit_band_adjustment_to_zero()

    def entry_band_adjustment(self):
        """
        @brief Calculates the entry band adjustment.

        This method calculates the adjustment for the entry band based on the funding entries for spot and swap with ratios.

        @return The calculated entry band adjustment.
        """
        return 10000 * (self.funding_spot.get_next_funding_entry() + self.funding_swap.get_next_funding_entry())

    def exit_band_adjustment(self):
        """
        @brief Calculates the exit band adjustment.

        This method calculates the adjustment for the exit band based on the funding exits for spot and swap with ratios.

        @return The calculated exit band adjustment.
        """
        return 10000 * (self.funding_spot.get_next_funding_exit() + self.funding_swap.get_next_funding_exit())

    def entry_band_adjustment_to_zero(self):
        """
        @brief Calculates the entry band adjustment to zero.

        This method calculates the adjustment to bring the entry band to zero based on the funding entries to zero for spot and swap with ratios.

        @return The calculated entry band adjustment to zero.
        """
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_entry() + self.funding_swap.get_next_funding_to_zero_entry())

    def exit_band_adjustment_to_zero(self):
        """
        @brief Calculates the exit band adjustment to zero.

        This method calculates the adjustment to bring the exit band to zero based on the funding exits to zero for spot and swap with ratios.

        @return The calculated exit band adjustment to zero.
        """
        return 10000 * (
                self.funding_spot.get_next_funding_to_zero_exit() + self.funding_swap.get_next_funding_to_zero_exit())


class FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff(FundingSystemOkxBinanceDiscoutedWithRatios):
    """
    @brief A funding system class for OKX and Binance with discounted funding and opposing ratios.

    This class extends FundingSystemOkxBinanceDiscoutedWithRatios to enforce opposing funding ratios for
    spot and swap.
    """

    name = "okx_binance_discounted_with_ratios_on_diff"
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBaseWithRatios, funding_swap: FundingBinanceDiscountedWithRatios):
        """
        @brief Initializes the FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff with spot and swap funding data.

        This constructor sets up the spot and swap funding objects, enforcing opposing funding ratios, and initializes
        the timestamp for the last update.

        @param funding_spot The spot funding data object with ratios.
        @param funding_swap The swap funding data object with opposing ratios.
        
        @throws Exception If the spot and swap funding ratios are not opposing values.
        """
        super(FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff, self).__init__(funding_spot, funding_swap)
        if funding_spot.funding_ratios != -funding_swap.funding_ratios:
            raise Exception("This funding system accepts only opposite values for the spot and swap ratios!")


class FundingSystemBinanceBinanceDiscoutedWithRatiosOnDiff(FundingSystemOkxBinanceDiscoutedWithRatios):
    """
    @brief A funding system class for Binance with discounted funding and opposing ratios.

    This class extends FundingSystemOkxBinanceDiscoutedWithRatios to provide functionality for Binance
    exchanges with enforced opposing funding ratios for spot and swap.
    """

    name = "binance_binance_discounted_with_ratios_on_diff"


class FundingSystemOkxBinanceDiscoutedWithRatiosModel(FundingSystemOkxBinanceDiscoutedWithRatios):
    """
    @brief A funding system class for OKX and Binance with discounted funding, ratios, and a predictive model.

    This class extends FundingSystemOkxBinanceDiscoutedWithRatios to include a predictive model for estimating
    future funding values.
    """

    name = "okx_binance_discounted_with_ratios_model"
    update_interval = five_minutes

    def __init__(self, funding_spot: FundingBase, funding_swap: FundingBinanceDiscountedWithRatiosModel):
        """
        @brief Initializes the FundingSystemOkxBinanceDiscoutedWithRatiosModel with spot and swap funding data.

        This constructor sets up the spot and swap funding objects, including a predictive model, and initializes
        the timestamp for the last update.

        @param funding_spot The spot funding data object.
        @param funding_swap The swap funding data object with a predictive model.
        """
        super(FundingSystemOkxBinanceDiscoutedWithRatiosModel, self).__init__(funding_spot, funding_swap)

    # What is needed? move each band up or down. There are 4 bands instead of 2 (entry/exit in position and
    # entry/exit not in position). There are 8 parameters, 2 per band, as each band can move up or down.
    # There are two additional cases


funding_systems = {"okx_binance_discounted": FundingSystemOkxBinanceDiscounted,
                   "": FundingSystemEmpty,
                   "okx_binance_discounted_with_ratios": FundingSystemOkxBinanceDiscoutedWithRatios,
                   "okx_binance_discounted_with_ratios_model": FundingSystemOkxBinanceDiscoutedWithRatiosModel,
                   "okx_binance_discounted_with_ratios_on_diff": FundingSystemOkxBinanceDiscoutedWithRatiosOnDiff,
                   "deribit_bitmex_with_ratios": FundingSystemDeribitBitMEXWithRatios,
                   "binance_binance_discounted_with_ratios_on_diff": FundingSystemBinanceBinanceDiscoutedWithRatiosOnDiff}

funding_classes = {"Binance": FundingBinanceDiscountedWithRatios,
                   "Okex": FundingBaseWithRatios,
                   "BitMEX": FundingBaseWithRatios,
                   "Deribit": FundingBaseWithRatios}


### CODE

def funding_values(t0, t1, exchange, symbol, environment):
    """
    @brief Retrieves funding values for a given time range, exchange, and symbol.

    This function queries the funding data from an InfluxDB database for a specified time range,
    exchange, and symbol. It supports both production and staging environments, handling special
    cases for the 'Deribit' exchange with additional processing for staging data.

    @param t0 The start time in milliseconds for the query.
    @param t1 The end time in milliseconds for the query.
    @param exchange The name of the exchange (e.g., 'Deribit', 'Binance').
    @param symbol The trading symbol (e.g., 'BTC-PERPETUAL').
    @param environment The environment to query ('production' or 'staging').

    @return A pandas DataFrame containing the queried funding data.

    @note The function uses different queries for 'Deribit' due to its unique data structure.
    """
    connection = InfluxConnection.getInstance()
    denormalized_factor = 8 * 60 * 60
    temp = []
    if exchange == 'Deribit':
        query = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                    FROM "real_time_funding"
                    WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t0}ms and time <= {t1}ms) GROUP BY time(1s)'''
    else:
        query = f''' SELECT "funding"
                        FROM "funding"
                        WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t0}ms and time <= {t1}ms) '''
    if environment == 'production':
        result = connection.prod_client_spotswap_dataframe.query(query, epoch='ns')
    elif environment == 'staging':
        if exchange == 'Deribit':
            t_start = t0
            t_end = t_start + 24 * 60 * 60 * 1000
            while t_end <= t1:
                query1 = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                                FROM "real_time_funding"
                                WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t_start}ms and time <= {t_end}ms) GROUP BY time(1s)'''
                result = connection.staging_client_spotswap_dataframe.query(query1, epoch='ns')
                temp.append(result['real_time_funding'])
                t_start = t_end
                t_end = t_start + 24 * 60 * 60 * 1000
                time.sleep(1)
            if 0 < t1 - t_start < 24 * 60 * 60 * 1000:
                query1 = f''' SELECT mean("funding") / {denormalized_factor} as "funding"
                                FROM "real_time_funding"
                                WHERE "exchange" = '{exchange}' AND "symbol" = '{symbol}' AND (time >= {t_start}ms and time <= {t1}ms) GROUP BY time(1s)'''
                result = connection.staging_client_spotswap_dataframe.query(query1, epoch='ns')
                if len(result) > 0:
                    temp.append(result['real_time_funding'])
            return pd.concat(temp)
        else:
            result = connection.staging_client_spotswap_dataframe.query(query, epoch='ns')
    else:
        result = None

    if exchange == 'Deribit':
        return result["real_time_funding"]
    else:
        return result["funding"]


def get_real_time_funding_local(t0: int = 0, t1: int = 0, market: str = 'Deribit', symbol: str = 'BTC-PERPETUAL'):
    """
    @brief Retrieves real-time funding data from local storage or InfluxDB for a given time range and market.

    This function attempts to load real-time funding data from local storage based on the specified time range,
    market, and symbol. If the local file is not found, it retrieves the data from an InfluxDB database.
    It processes daily funding files and handles the merging of data across the specified time range.

    @param t0 The start time in milliseconds for the query.
    @param t1 The end time in milliseconds for the query.
    @param market The name of the market (e.g., 'Deribit', 'Binance').
    @param symbol The trading symbol (e.g., 'BTC-PERPETUAL').

    @return A pandas DataFrame containing the real-time funding data.

    @note The function handles cases where the time range spans multiple days and combines the data accordingly.
    """
    try:
        day_in_millis = 1000 * 60 * 60 * 24
        dfs = []
        if t1 - t0 >= day_in_millis:
            t_start = t0
            t_end = t0 + day_in_millis

            while t_end <= t1:
                if t1 - day_in_millis <= t_end <= t1:
                    t_end = t1

                base_dir = f"/home/equinoxai/data"
                if not os.path.isdir(base_dir):
                    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../",
                                            "simulations_management", "data")
                local_dir_swap = f"{base_dir}/real_time_funding/{market}/{symbol}/{market}_{symbol}_{pd.to_datetime(t_start, unit='ms', utc=True).date()}.parquet.br"
                if os.path.exists(local_dir_swap):
                    # print(f"Loading real time funding from local file {local_dir_swap}")
                    try:
                        df = pd.read_parquet(local_dir_swap, engine="pyarrow")
                    except:
                        df = pd.read_parquet(local_dir_swap)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
                        df = df.set_index("timestamp")
                    elif 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], unit='ns', utc=True)
                        df = df.set_index("time")
                    else:
                        df.index = pd.to_datetime(df.index, unit='ns', utc=True)
                    df['funding'] = df['funding'].astype(np.float64)
                    dfs.append(df)
                    # print(df.head())
                    df = None
                    # time.sleep(1)
                else:
                    print(f"Loading real time funding from influx. Couldn't find {local_dir_swap}")
                    # print(f"t_start: {pd.to_datetime(t_start, unit='ms', utc=True).date()}, t_end: {pd.to_datetime(t_end, unit='ms', utc=True).date()}")
                    dfs.append(
                        funding_values(t0=t_start, t1=t_end, exchange=market, symbol=symbol, environment='staging'))
                    time.sleep(1)
                t_start = t_start + day_in_millis + 1000
                t_end = t_start + day_in_millis
            # print('end while loop')
            if 0 < t1 - t_start < day_in_millis:
                # print('entering last if')
                dfs.append(funding_values(t0=t_start, t1=t1, exchange=market, symbol=symbol, environment='staging'))
        else:
            df = funding_values(t0=t0, t1=t1, exchange=market, symbol=symbol, environment='staging')
            dfs.append(df)
            time.sleep(1)
        # print(f"dfs: {pd.concat(dfs)}")
        return pd.concat(dfs)
    except KeyError:
        print(f"keyError: {KeyError} in get_real_time_funding_local")
        return pd.DataFrame(columns=['funding'])


def funding_implementation(t_start: int = 0,
                           t_end: int = 0,
                           swap_exchange: str = None,
                           swap_symbol: str = None,
                           spot_exchange: str = None,
                           spot_symbol: str = None,
                           position_df: pd.DataFrame = None,
                           environment: str = None):
    """
    @brief Calculates total funding for swap and spot exchanges over a specified time range.

    This function computes the total funding for swap and spot exchanges by retrieving funding data
    and applying it to a given position DataFrame. It handles both continuous and periodical funding
    methods, depending on the exchange type.

    @param t_start The start time in milliseconds for the calculation.
    @param t_end The end time in milliseconds for the calculation.
    @param swap_exchange The name of the swap exchange (e.g., 'Deribit').
    @param swap_symbol The trading symbol for the swap exchange.
    @param spot_exchange The name of the spot exchange.
    @param spot_symbol The trading symbol for the spot exchange.
    @param position_df A pandas DataFrame containing position data with timestamps and traded volumes.
    @param environment The environment to query ('production' or 'staging').

    @return A tuple containing:
        - total_funding_spot: The total funding value for the spot exchange.
        - total_funding_swap: The total funding value for the swap exchange.
        - total_funding: The combined total funding for both spot and swap exchanges.
        - spot_funding: A DataFrame containing the spot funding data.
        - swap_funding: A DataFrame containing the swap funding data.

    @note The function handles both Deribit and other exchanges with specific logic for funding retrieval and calculations.
    """
    if len(position_df) == 0:
        return 0, 0, 0, 0, 0

    try:
        if swap_exchange == 'Deribit':
            swap_funding = get_real_time_funding_local(t0=t_start, t1=t_end, market=swap_exchange, symbol=swap_symbol)
        else:
            swap_funding = funding_values(t0=t_start, t1=t_end, exchange=swap_exchange, symbol=swap_symbol,
                                          environment=environment)
        swap_funding['timems'] = swap_funding.index.view(np.int64) // 10 ** 6
        swap_funding.reset_index(drop=True, inplace=True)
        if swap_exchange == 'Deribit':
            swap_index = swap_funding['timems'].searchsorted(position_df['timems'].to_list(), side='left')
        else:
            swap_index = 0
    except:
        swap_funding = pd.DataFrame()
        swap_index = None

    try:
        if spot_exchange == 'Deribit':
            spot_funding = get_real_time_funding_local(t0=t_start, t1=t_end, market=spot_exchange, symbol=spot_symbol)
        else:
            spot_funding = funding_values(t0=t_start, t1=t_end, exchange=spot_exchange, symbol=spot_symbol,
                                          environment=environment)
        spot_funding['timems'] = spot_funding.index.view(np.int64) // 10 ** 6
        spot_funding.reset_index(drop=True, inplace=True)
        if spot_exchange == 'Deribit':
            spot_index = spot_funding['timems'].searchsorted(position_df['timems'].to_list(), side='left')
        else:
            spot_index = 0
    except:
        spot_funding = pd.DataFrame()
        spot_index = None

    if len(swap_funding.index) != 0:
        swap_funding['value'] = 0.0
        if swap_exchange == 'Deribit':
            # swap_funding = continuous_funding_fun(t1=t1, funding_df=swap_funding, index_list=swap_index,
            #                                       position_df=position_df, spot=False)
            swap_funding.reset_index(drop=True, inplace=True)
            swap_funding_array = swap_funding.fillna(0).to_numpy()
            position_df_array = position_df[['timems', 'traded_volume']].fillna(0).to_numpy()
            position_df_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            position_df_columns_map['timems'] = 0
            position_df_columns_map['traded_volume'] = 1
            swap_funding_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            for ix in range(len(swap_funding.columns)):
                swap_funding_columns_map[swap_funding.columns[ix]] = ix
            swap_funding_array = continuous_funding_fun_numba(funding_array=swap_funding_array, index_list=swap_index,
                                                              position_array=position_df_array, spot=False,
                                                              position_columns_map=position_df_columns_map,
                                                              funding_columns_map=swap_funding_columns_map)
            swap_funding = pd.DataFrame(swap_funding_array, columns=swap_funding.columns)

        else:
            swap_funding = periodical_funding_fun(t0=t_start, t1=t_end, swap_exchange=swap_exchange,
                                                  swap_symbol=swap_symbol,
                                                  position_df=position_df, environment=environment, spot=False)
        total_funding_swap = swap_funding['value'].sum()
    else:
        total_funding_swap = 0

    if len(spot_funding.index) != 0:
        spot_funding['value'] = 0.0
        if spot_exchange == 'Deribit':
            # spot_funding = continuous_funding_fun(t1=t1, funding_df=spot_funding, index_list=spot_index,
            #                                       position_df=position_df)
            spot_funding_array = spot_funding.fillna(0).to_numpy()
            position_df_array = position_df[['timems', 'traded_volume']].fillna(0).to_numpy()
            position_df_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            position_df_columns_map['timems'] = 0
            position_df_columns_map['traded_volume'] = 1
            spot_funding_columns_map = Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.int64)
            for ix in range(len(spot_funding.columns)):
                spot_funding_columns_map[spot_funding.columns[ix]] = ix
            spot_funding_array = continuous_funding_fun_numba(funding_array=spot_funding_array, index_list=spot_index,
                                                              position_array=position_df_array, spot=True,
                                                              position_columns_map=position_df_columns_map,
                                                              funding_columns_map=spot_funding_columns_map)
            spot_funding = pd.DataFrame(spot_funding_array, columns=spot_funding.columns)


        else:
            spot_funding = periodical_funding_fun(t0=t_start, t1=t_end, swap_exchange=spot_exchange,
                                                  swap_symbol=spot_symbol,
                                                  position_df=position_df, environment=environment, spot=True)

        total_funding_spot = spot_funding['value'].sum()
    else:
        total_funding_spot = 0

    return total_funding_spot, total_funding_swap, total_funding_swap + total_funding_spot, spot_funding, swap_funding


def periodical_funding_fun(t0: int = 0, t1: int = 0, swap_exchange: str = None, swap_symbol: str = None,
                           position_df: pd.DataFrame = None, environment: str = None, spot=True):
    """
    @brief Calculates funding values over discrete periods for a given swap exchange and symbol.

    This function retrieves funding data for a specified swap exchange and symbol over discrete periods,
    merging it with position data to compute funding values based on traded volumes and funding rates.

    @param t0 The start time in milliseconds for the calculation.
    @param t1 The end time in milliseconds for the calculation.
    @param swap_exchange The name of the swap exchange (e.g., 'Binance').
    @param swap_symbol The trading symbol for the swap exchange.
    @param position_df A pandas DataFrame containing position data with timestamps and traded volumes.
    @param environment The environment to query ('production' or 'staging').
    @param spot A boolean indicating whether the calculation is for spot trading (default is True).

    @return A pandas DataFrame containing the computed funding values with timestamps and funding rates.

    @note The function handles both spot and swap trading with specific logic for funding value calculations.
    """
    funding_df = funding_values(t0=t0, t1=t1, exchange=swap_exchange, symbol=swap_symbol, environment=environment)
    if len(funding_df) == 0 or funding_df.empty:
        print('No funding data available')
        return pd.DataFrame(columns=['funding', 'timems', 'value'])
    funding_df['timems'] = funding_df.index.view(np.int64) // 10 ** 6
    funding_df.fillna(0, inplace=True)
    df = pd.merge_ordered(position_df, funding_df, on='timems')
    df['traded_volume'].ffill(inplace=True)
    df['value'] = np.nan
    for ix in df[~df['funding'].isna()].index:
        if spot:
            if df.loc[ix, 'traded_volume'] > 0 and df.loc[ix, 'funding'] > 0:
                df.loc[ix, 'value'] = -abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] > 0 >= df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] <= 0 < df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            else:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
        else:
            if df.loc[ix, 'traded_volume'] > 0 and df.loc[ix, 'funding'] > 0:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] > 0 >= df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            elif df.loc[ix, 'traded_volume'] <= 0 < df.loc[ix, 'funding']:
                df.loc[ix, 'value'] = - abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
            else:
                df.loc[ix, 'value'] = abs(df.loc[ix, 'funding'] * df.loc[ix, 'traded_volume'])
    temp = df.loc[~df['value'].isna(), 'value']
    funding_df['value'] = 0.0
    if len(temp) <= len(funding_df):
        val = len(funding_df) - len(temp)
        funding_df.iloc[val:, -1] = temp.values
        return funding_df
    else:
        funding_df['value'] = temp.iloc[:len(funding_df)].values
        return funding_df


@numba.jit(nopython=True)
def continuous_funding_fun_numba(funding_array: np.array = None, index_list: list = None,
                                 position_array: np.array = None, spot: bool = True,
                                 position_columns_map=None, funding_columns_map=None):
    """
    @brief Computes continuous funding values for positions using Numba for optimization.

    This function calculates continuous funding values for trading positions using funding and position
    data arrays. It employs Numba for just-in-time compilation to optimize performance and speed up
    calculations.

    @param funding_array A numpy array containing funding data.
    @param index_list A list of indices for matching funding data to positions.
    @param position_array A numpy array containing position data.
    @param spot A boolean indicating whether the calculation is for spot trading (default is True).
    @param position_columns_map A mapping of position column names to indices.
    @param funding_columns_map A mapping of funding column names to indices.

    @return A numpy array with updated funding values based on the provided data.

    @note The function assumes that input arrays and mappings are correctly structured and valid.
    """
    timems = position_columns_map['timems']
    traded_volume = position_columns_map['traded_volume']
    funding = funding_columns_map['funding']
    value = funding_columns_map['value']
    abs_funding_array = np.abs(funding_array[:, :])
    abs_position_array = np.abs(position_array[:, :])
    for idx3, idx4 in zip(index_list, range(len(position_array))):
        if idx4 >= len(position_array) - 1:
            break
        if idx3 >= len(funding_array) - 1:
            break

        pos_dur = int((position_array[idx4 + 1, timems] - position_array[idx4, timems]) / 1000)

        idx_local = idx3
        counter = pos_dur
        while counter >= 0 and idx_local < len(funding_array) - 1:

            funding_deribit = abs_funding_array[idx_local, funding] * abs_position_array[idx4, traded_volume]

            if spot:
                if position_array[idx4, traded_volume] > 0 and funding_array[idx_local, funding] > 0:
                    funding_value = - funding_deribit
                elif position_array[idx4, traded_volume] > 0 >= funding_array[idx_local, funding]:
                    funding_value = funding_deribit
                elif position_array[idx4, traded_volume] <= 0 < funding_array[idx_local, funding]:
                    funding_value = funding_deribit
                else:
                    funding_value = - funding_deribit
            else:
                if position_array[idx4, traded_volume] > 0 and funding_array[idx_local, funding] > 0:
                    funding_value = funding_deribit
                elif position_array[idx4, traded_volume] > 0 >= funding_array[idx_local, funding]:
                    funding_value = - funding_deribit
                elif position_array[idx4, traded_volume] <= 0 < funding_array[idx_local, funding]:
                    funding_value = - funding_deribit
                else:
                    funding_value = funding_deribit

            idx_local += 1
            counter -= 1

            funding_array[idx_local, value] = funding_value
    return funding_array
