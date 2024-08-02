# 📄 Documentation for `simulations_management` Folder

This document provides an in-depth look at the functionalities within the `simulations_management` folder. This folder contains multiple Python scripts that manage various aspects of simulations. The primary purpose of this module is to facilitate automatic simulations, manage training and confirmations, handle sweep operations, and merge results.

## Table of Contents

- [File Listing](#file-listing)
- [File Descriptions](#file-descriptions)
  - [`SimulationsMasterFile.py`](#SimulationsMasterFilepy)
  - [`download_sweep_results.py`](#download_sweep_resultspy)
- [Classes](#classes)
  - [`AutomatedSimulation`](#automatedsimulation)
    - [`AutomatedSimulationSinglePeriod`](#automatedsimulationsingleperiod)
  - [`DataFetcher`](#datafetcher)
- [Functions](#functions)
  - [Sweep Functions](#sweep-functions)
- [Modules](#modules)
  - [Dependencies](#dependencies)

## File Listing

```plaintext
- SimulationsMasterFile.py
- download_sweep_results.py
```

## File Descriptions

### `SimulationsMasterFile.py`

This script handles the entire workflow of an automated simulation, which includes starting training, fetching results, and initiating confirmations.

**Key Classes and Functions:**
- **AutomatedSimulation:** Manages the lifecycle of simulations.
- **AutomatedSimulationSinglePeriod:** Extends `AutomatedSimulation` to handle single-period simulations.
- **DataFetcher:** Fetches and uploads data between different stages of simulation.

### `download_sweep_results.py`

This script is responsible for rerunning simulation sweeps and downloading the corresponding results.

**Key Functions:**
- **sweep_rerun_simulations:** Re-runs simulations based on a specific sweep ID and downloads the results from W&B.

## Classes

### `AutomatedSimulation`

```python
class AutomatedSimulation:
    def __init__(self, symbol, sweep_id, t_start, t_end, name="Estimated PNL with Realized Quanto_profit"):
        # Initialization code...

    def connect_to_wandb(self):
        # Connect to W&B...

    def start(self):
        # Start the simulation...

    def init_confirmations(self):
        # Initialize confirmations...

    def start_training(self, agents):
        # Start training...

    def get_training_results_and_start_controllers(self):
        # Get training results and start controllers...

    def wait_for_controllers_and_start_confirmations(self):
        # Wait for controllers and start confirmations...

    def merge_results(self):
        # Merge results...
```

### `AutomatedSimulationSinglePeriod`

```python
class AutomatedSimulationSinglePeriod(AutomatedSimulation):
    @staticmethod
    def setup_and_run(symbol, file, time_from, time_to):
        # Setup and run single-period simulation...

    @staticmethod
    def get_parameters_from_csv(filename):
        # Get parameters from CSV...

    @staticmethod
    def init_sweep_and_start_containers(name, symbol, t_start, t_end, params_df=None):
        # Initialize sweep and start containers...
```

### `DataFetcher`

```python
class DataFetcher:
    def __init__(self):
        # Initialization code...

    def fetch_and_upload(self, base_path, exchange, symbol, start, end):
        # Fetch and upload data...
```

## Functions

### Sweep Functions

#### `sweep_rerun_simulations`

```python
def sweep_rerun_simulations(sweep_id: str = 'jd1a03uf', select_num_simulations: int = 30, custom_filter: str = 'no', project_name: str = 'taker_maker_simulations_2023_2'):
    # Rerun sweep simulations...
```

## Modules

### Dependencies

The `simulations_management` folder relies on a variety of libraries and other modules to function correctly:

**Python Standard Library**
- `os`
- `time`
- `datetime`

**Third-Party Libraries**
- `wandb`: Used for managing and retrieving sweep data.
- `pandas`: For data manipulation.
- `subprocess`: For running shell commands.

**Environment Management**
- `dotenv`: For loading environment variables.

**Internal Modules**
- `DatabaseConnections`: Manages connections to databases like InfluxDB and PostgreSQL.
- `utils`: Contains utility functions such as `create_array_from_generator`.

## Summary

The `simulations_management` module is a critical part of the project, managing the lifecycle of simulations from start to finish. It automates the training and confirmation processes, communicates with external services like W&B, and merges results to provide a comprehensive overview of the simulation performance.

For more detailed information, refer to the specific function and class documentation within the scripts.