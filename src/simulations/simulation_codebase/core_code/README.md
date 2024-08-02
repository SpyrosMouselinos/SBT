# Documentation for `src/simulations/simulation_codebase/core_code`

This folder contains core code for various trading and simulation functionalities. The code base is largely constructed in Python and leverages several libraries including pandas, numpy, and transitions. The main objective of these modules is to interface with trading systems and handle simulations for Quanto and Taker Maker strategies. Below is an in-depth description of the key components found within this directory.

## **Files and Classes**

### 1. `base_new.py`

This is a fundamental module in the core code, which mainly deals with the simulation models used in trading. 

#### **Key Components:**

1. **Imports:**
   - Libraries like `numba`, `pandas`, `numpy`, `transitions`, `logging`, and `pytictoc` are employed.
   - Specific functions from other project modules are also imported for various utilities.

2. **Class `TraderExpectedExecutions`:**
   - **Variables:** Describes key variables like `source`, `final_spread`, `df`, `timestamp`, `side`, `position`, etc.
   - **Methods:**
     - `exit_band_adjustment()`: Adjusts the exit band based on various parameters.
     - `entry_band_adjustment()`: Adjusts the entry band based on parameters like `entry_band`, `exit_band`.
   - **Finite State Machine Transitions:**
     - This class employs FSM transitions using the `transitions` library to switch between different states like `clear`, `trying_to_post`, `posted`, `try_to_cancel`, and `executing`.

### 2. `quanto_systems`

Several quanto profit and loss systems are defined here. 

- **`QuantoBoth`**: Handles combined algorithms for both profit and loss in Quanto.
- **`QuantoBothExtended`**: Extends functionalities offered by `QuantoBoth`.
- **`QuantoExponential`**: Exponential variations for the Quanto system.
- **`QuantoProfit`**: Basic Quanto profit system.
- **`QuantoProfitBuildQuanto`**: Build functionalities for Quanto profit systems.
- **`QuantoSystemEmpty`**: An empty template for Quanto systems.
- **`QuantoLossSystem`**: Handles loss parameters in Quanto trading.

### 3. `simulation_taker_trades.py`

This module focuses on simulating taker trades.

#### **Key Components:**

- **Functions:**
  - **Various FSM event handlers:** Systems to handle FSM states and transitions similar to the `TraderExpectedExecutions` class.

### 4. `simulation_maker_taker_function.py`

A crucial module that handles the maker-taker function simulations.

#### **Key Components:**

1. **Methods for Fetching Data:**
   - Uses `get_data_for_trader`, `funding_implementation()`, `compute_rolling_pnl()`, among other helper functions.
  
2. **Main Simulation Function:**
   - `simulation_trader(params)`: Core logic for simulation based on the given parameters which include `band`, `lookback`, `recomputation_time`, `target_percentage_exit`, etc.

### 5. `taker_maker/simulation_taker_maker_at_depth_function.py`

This file focuses on handling depth-level computations for Taker-Maker simulations.

#### **Key Components:**

- **Function: `simulation_trader_taker_maker_at_depth(params)`:**
  - Handles parameter fetching and initiates depth-level analysis for trades.
  - Employs `ConstantDepthPosting`, `get_entry_exit_bands`, etc.【4:0†source】【4:2†source】【4:3†source】.

### **General Notes**

- **Finite State Machines (FSM):** Many classes leverage FSM to manage and transition system states for various trading operations.
- **Core Parameters:** Most trading simulations depend on parameters like `latency`, `spread`, `fee`, `instrument`, `environment`, `strategy`, etc.
- **Wandb Integration:** The modules often use `wandb` for logging and tracking experiments and their parameters, crucial for model evaluation.
- **Error Handling:** Logging and error management are embedded, with warnings filtered to provide smooth simulation running.

### **Usage Example**

#### **Example to Run a Simple Simulation:**

```python
from src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function import simulation_trader

params = {
    'band': 'bogdan_bands',
    'lookback': None,
    'recomputation_time': None,
    'target_percentage_exit': None,
    'target_percentage_entry': None,
    'entry_opportunity_source': None,
    'exit_opportunity_source': None,
    't_start': '2023-01-01',
    't_end': '2023-01-02',
    'family': 'deribit_eth',
    'environment': 'production',
    'strategy': 'strategy1',
    # Add other necessary parameters...
}

simulation_trader(params=params)
```

### **Conclusion**

The `core_code` directory in the simulation codebase is vital for executing trading strategies and experiments. The comprehensive use of FSM ensures that simulations can smoothly transit between different states, making the system highly flexible and robust.