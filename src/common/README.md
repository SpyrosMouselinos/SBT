## 📂 `src/common` Folder Documentation

### Overview
The `src/common` folder contains essential modules and utilities used across the project. These modules facilitate connections to databases, handle API calls, manage utility functions, and more. Below is an exhaustive breakdown of each significant file within the `src/common` directory.

### Contents

- `api_calls.py`
- `connections/`
  - `DatabaseConnections.py`
  - `ExchangeConnections.py`
- `queries/`
  - `funding_queries.py`
  - `queries.py`
  - `__init__.py`
- `io/`
  - `TraderParsers.py`
- `utils/`
  - `utils.py`
  - `wandb_login.py`

### Modules

#### `api_calls.py`
This module provides functionality to connect with the Equinox API and create bands and spreads:

```python
class DatalinkCreateBands:
    """
    Class to connect with equinox API and create bands and spreads.
    """
```
- **Methods and attributes** include parameters for start and end times, exchange names, symbols, window sizes, and more. Its primary purpose is to interact with the API to gather and manipulate trading data.

#### `connections/DatabaseConnections.py`
This module manages database connections for different environments (staging, production, local):

```python
class InfluxConnection:
    """
    Singleton class to handle connections to InfluxDB.
    """
    @staticmethod
    def getInstance():
        """Returns the singleton instance."""
    # Various properties for different clients (staging, production, etc.)
```
- **Key Classes**:
  - `InfluxConnection`: Singleton managing InfluxDB connections.
  - `ClickhouseConnection`: Handles connections to Clickhouse database, used for querying funding rates.

#### `connections/ExchangeConnections.py`
Handles connections to different exchanges using websockets:

```python
class GenericExchange(EventEmitter):
    """
    Base class for managing connections to exchanges via websockets.
    """
```
- **Key Classes**:
  - `GenericExchange`: Manages websocket connections and handles events.
  - `TakerTrade`: Represents a trade, including price, size, side, timestamp, exchange, and symbol.

#### `queries/funding_queries.py`
Contains configurations and functionalities for handling funding queries:

```python
class FundingRatiosParams:
    """
    Data class for funding ratio parameters.
    """
```
- **Key Classes**:
  - `FundingBase`: Base class for operations involving funding data.
  - `FundingBaseWithRatios`: Extends `FundingBase` to include ratio handling.

#### `queries/queries.py`
Manages querying of data from various sources and processes it:

```python
class Prices(DataQueryBase):
    """Handles querying price data."""
```
- **Key Classes**:
  - `PriceResult`: Represents the result of a price query.
  - `DataQueryBase` and `Prices`: Base and extended class for querying data.

#### `queries/__init__.py`
Dynamically imports query functions:

```python
import inspect
import importlib

# Filter functions that start with 'get'
get_functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction) if name.startswith('get')]
```
- **Purpose**: Automatically imports query functions, facilitating dynamic access.

#### `io/TraderParsers.py`
Parser for command-line arguments used throughout the project:

```python
class GenericParser:
    """
    Generic argument parser with common and specific arguments.
    """
```
- **Key Methods**:
  - `add_common_arguments()`: Adds common CLI arguments.
  - `add_specific_arguments()`: Adds specific arguments for the application.

#### `utils/utils.py`
Contains various utility functions:

```python
@numba.jit()
def spread_entry_func_numba(entry_swap, entry_spot, swap_fee, spot_fee):
    """
    Numba-optimized function to calculate the entry spread.
    """
```
- **Utility Functions**:
  - Numba-optimized functions for calculation and data manipulation (e.g., `shift_array`, `df_numba`).

#### `utils/wandb_login.py`
Automates the process of logging into Weights & Biases (WandB):

```python
import os, subprocess
key = os.getenv("WANDB_API_KEY")
command = f"wandb login --host https://wandb.staging.equinoxai.com {key}"
process = subprocess.Popen(command, shell=True)
result = process.communicate()
```
- **Purpose**: Facilitates easy authentication with WandB using environment variables.

### Summary

The `src/common` directory is the backbone of the project, containing various utilities and connection handlers essential for data interaction and manipulation within trading environments. Each module and sub-module has specialized purposes, from database connections to utility functions, ensuring the smooth operation of higher-level functionalities.