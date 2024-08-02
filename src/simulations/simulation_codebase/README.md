# Documentation for the Simulation Codebase

## Overview

The **simulation_codebase** folder is a key part of a broader simulation project built to model trading activities, particularly in cryptocurrency markets. This simulation system is designed in **Python** and leverages various Python libraries such as `numpy`, `pandas`, and `numba`. It also includes integration with `transitions` for state management and `pytictoc` for timing tasks.

**Primary Functionalities:**
- Perform real-time and historical trading simulations.
- Calculate profits and losses (PnL) for specific trading strategies.
- Adjust trading band parameters dynamically based on market conditions.

## Folder Structure
The folder comprises several subfolders and files, each crafted to handle specific tasks within the simulation process. Below, we will delve into the key components and their roles.

### Key Subfolders and Files

#### quanto_systems

1. **QuantoProfitSystem.py**
    - **Purpose:** Define the base class for quanto profit systems and the basic methods for updating prices and trading simulation logic.
    - **Important Classes:**
        - `QuantoSystemEmpty`: Serves as a base class holding common attributes and methods such as `update_trade`, `entry_band_adjustment`, and `exit_band_adjustment`.
        - `QuantoProfitSystem`: Inherits from `QuantoSystemEmpty` and adds specific logic for handling Bitcoin (BTC) and Ethereum (ETH) price dataframes, as well as parameters for trading bands and rolling windows.

2. **QuantoProfitBuildQuanto.py**
    - **Purpose:** Extends the `QuantoSystemEmpty` with additional logic for computing and adjusting trading bands based on dynamic signals.
    - **Important Methods:**
        - `update`: Processes and updates index positions for BTC and ETH prices based on the timestamp.
        - `entry_band_adjustment` & `exit_band_adjustment`: Adjust trading entry and exit bands dynamically based on computed quanto profits.

3. **QuantoBoth.py**
    - **Purpose:** Implements trading logic for scenarios involving both entry and exit ratios affected by different market states.
    - **Important Attributes:**
        - `stop_trading_enabled`: Boolean flag to initiate trading stop mechanisms.
        - `trading_condition`: Define conditions under which trading can continue or be halted.

4. **QuantoBothExtended.py**
    - **Purpose:** Extends `QuantoBoth.py` to add more business logic covering extended cases.
    - **Integrated Components:** Includes calls to utility functions for computing quanto PnL.

5. **QuantoExponential.py**
    - **Purpose:** Similar to `QuantoProfitSystem.py` but uses an exponential model to calculate and adjust trading strategies.
    - **Important Methods:**
        - `update`: Sets price indices and price points for BTC and ETH.
        - `entry_band_adjustment` & `exit_band_adjustment`: Use exponential formula for more nuanced trading logic.

#### core_code

1. **base_new.py**
    - **Purpose:** Acts as the core of the trading system, establishing trading rules and handling the state machine for trading activities.
    - **Key Features:**
        - Integration of state machine using `transitions`.
        - Execution of the main trading logic utilizing various quanto profit systems.

2. **simulation_taker_trades.py**
    - **Purpose:** Handles the simulation of trading activities, particularly taker trades.

#### pnl_computation_functions

1. **pnl_computation.py**
    - **Purpose:** Contains methods for calculating and rolling PnL for both entry and exit executions.
    - **Key Processes:**
        - `compute_rolling_pnl`: Rolls up profits and losses by considering the cumulative volume and executed spreads.
        - Uses `numba` just-in-time compilation to accelerate computation.

#### process_results

1. **convert_simulation_csv_for_production.py**
    - **Purpose:** Converts simulation results from CSV files to a format suitable for production environments.
    - **Methods:**
        - `convert_simulation_csv_to_production_csv`: Cleans and processes raw simulation data to extract and format required parameters.

#### simulation_tests

1. **funding_computation_test.py**
    - **Purpose:** Test the robustness of funding computations involved in trading simulations.
    - **Tasks:**
        - Downloads simulation results and tests different funding scenarios using integrated Backblaze client.

## How the Files Connect

1. **Data Handling and Initialization:**
    - **base_new.py** initializes trading systems, reading data, and linking various quanto profit systems as per the specified strategy.

2. **Simulation Execution:**
    - **simulation_taker_trades.py** is used to handle specific types of trades and simulate their performance against historical data.

3. **Profit and Loss Calculation:**
    - **pnl_computation.py** imports execution data and computes rolling PnL for various trading strategies.

4. **Results Processing:**
    - **convert_simulation_csv_for_production.py** is responsible for formatting and refining the simulation data, preparing it for productive analytics or further use.

5. **Testing:**
    - **funding_computation_test.py** ensures that the computations and simulations are reliable by running specific test cases.

## Data Flow
1. **Initialization**:
    - Load price and execution data through dataframes.
    - Initialize system state and parameters.

2. **Execution**:
    - Simulate trades using `QuantoProfitBuildQuanto`, `QuantoBoth`, and other system classes.
    - Continuously update prices and trading conditions.

3. **Calculation**:
    - Compute profits and losses using defined computational functions.
    - Adjust trading parameters dynamically based on computing signals.

4. **Output**:
    - Convert and store results for further analysis or production use.

## Conclusion

The `simulation_codebase` directory forms the backbone of a sophisticated trading simulation platform. Leveraging multiple Python libraries and organized into a series of specialized modules and classes, it handles the full cycle from data ingestion to trade simulation and results processing. These functionalities enable robust testing and adjustment of crypto trading strategies in various market conditions.