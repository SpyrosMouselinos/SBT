import pandas as pd
from datetime import datetime, timedelta

def convert_dates_to_timestamps(t_start, t_end):
    """
    Helper function to convert date range to timestamps in milliseconds.

    Parameters:
    - t_start (datetime): Start date.
    - t_end (datetime): End date.

    Returns:
    - Tuple of (temp_start, temp_end) in milliseconds.
    """
    temp_start = int(t_start.timestamp() * 1000)
    temp_end = int(t_end.timestamp() * 1000)
    return temp_start, temp_end


def drop_hours_from_datetime_object(dt: datetime):
    """
    Helper function to drop the hours, and keep only the date from a datetime object.
    """
    return dt.strftime('%Y-%m-%d')


class DateRangeIterator:
    def __init__(self, start: datetime, end: datetime):
        """
        Initialize the DateRangeIterator with start and end dates.

        Parameters:
        - start (datetime): The initial start date and time.
        - end (datetime): The initial end date and time.
        """
        self.start = start
        self.end = end
        self.current_start = start
        self.current_end = start

    def get_next_range(self):
        """
        Get the next date range for data fetching.

        Returns:
        - Tuple of (current_start, current_end) for the next range.
        """
        self.current_start = self.current_end
        self.current_end = self.current_start + timedelta(days=1)

        # If the next end date goes beyond the final end date, adjust it
        if self.current_end > self.end:
            self.current_end = self.end

        return self.current_start, self.current_end

    def has_more_ranges(self):
        """
        Check if there are more date ranges to process.

        Returns:
        - Boolean indicating if more ranges are available.
        """
        return self.current_start < self.end



if __name__ == '__main__':
    # Usage example:
    start = pd.to_datetime(1705492800000, unit='ms', utc=True)
    end = pd.to_datetime(1705665600000, unit='ms', utc=True)
    iterator = DateRangeIterator(start, end)

    while iterator.has_more_ranges():
        current_start, current_end = iterator.get_next_range()
        print(f"Start: {current_start}, End: {current_end}")
