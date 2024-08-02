# Detailed Documentation for `\src\simulations\taker_maker` Folder

This documentation provides a comprehensive overview of the components located within the `\src\simulations\taker_maker` folder. It includes class definitions, functions, and operational details relevant to the Taker-Maker simulation system.

### Overview

The Taker-Maker simulation system is designed to emulate market conditions and provide insights into trading strategies. Key functionalities of this system involve the evaluation of taker and maker trades, execution quality computation, and latency settings. The folder contains several primary Python files that implement these functionalities.

### Files and Their Functionalities

| Filename                                                         | Description                                                                                           |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `TakerMakerDeeperLevelSpread.py`                                 | Contains the main class `TakerMakerDeeperLevelSpread` which extends the `TakerMakerFunctions`.        |
| `simulation_taker_maker_at_depth_function.py`                    | Implements the `simulation_trader_taker_maker_at_depth` function, used for running the depth strategy.|
| `TakerMakerFunctions.py`                                         | Defines core functionalities like `LimitOrder`, `TakerExecution`, and various helper functions.       |
| `taker_maker_worker.py`                                          | Implements the worker class `TakerMakerWorker` for handling task queues and executing functions.      |
| Various `generate_reports`, `process_sweep_results`, and other helper scripts. | Auxiliary scripts for generating reports and processing results.                                      |

### Detailed Class and Function Documentation

#### 1. `TakerMakerDeeperLevelSpread` Class in `TakerMakerDeeperLevelSpread.py`

**Inheritance**: Extends from `TakerMakerFunctions`

```python
class TakerMakerDeeperLevelSpread(TakerMakerFunctions):
    # Define the states of the trading:
    states = [...]
```

#### Initialization Parameters

- **df**: DataFrame containing trading data.
- **spot_fee, swap_fee**: Fees for spot and swap transactions.
- **area_spread_threshold**: Threshold for spread area.
- **latency_spot, latency_swap**: Latency settings for spot and swap operations.
- Several other parameters such as `max_position`, `max_trade_volume`, `environment`, `exchange_swap`, and others defining trading environment settings.

#### Key Methods

- **`__init__`**: Initializes the simulation with various parameters.
- **States and Transitions**:
  - **states**: Defines various states like `clear`, `trying_to_post`, `posted`, `executing`, etc.
  - **transitions**: Adds transitions between states specifying conditions and after-effects.

#### 2. `simulation_trader_taker_maker_at_depth` Function in `simulation_taker_maker_at_depth_function.py`

```python
def simulation_trader_taker_maker_at_depth(params):
    ...
```

**Parameters**:
- **params**: Dictionary containing keys like `band`, `lookback`, `recomputation_time`, `strategy`, `exchange_spot`, `exchange_swap`, etc.

**Functionality**:
- This function uses various parameters to simulate a trader's operations at a certain depth, determining entries and exits based on the provided bands and latencies.

#### 3. Core Functionalities in `TakerMakerFunctions.py`

- **LimitOrder Class**:
  ```python
  class LimitOrder:
      def __init__(self, timestamp_posted, price, is_executed, side, ...):
          ...
  ```

- **TakerExecution Class**:
  ```python
  class TakerExecution:
      def __init__(self, timestamp_posted, targeted_price, executed_price, side, ...):
          ...
  ```

- **TakerMakerFunctions Class**:
  - **set_order_depth(new_depth)**: Sets new order depth.
  - **reset_depth(event)**: Resets the depth predictor.
  - Additional methods like **`is_order_too_deep`** for evaluating the order conditions.

#### 4. Worker Implementation in `taker_maker_worker.py`

**Class**: `TakerMakerWorker`

```python
class TakerMakerWorker:
    def __init__(self, queue="simulation_rpc_queue"):
        ...
```

**Methods**:
- **on_message(self, ch, method, props, body)**: Receives messages from the queue and initiates processing.
- **work(self, connection, channel, delivery_tag, body)**: Delegates the work to simulation functions.
- **ack_message(self, channel, delivery_tag)**: Acknowledges message processing.

#### 5. Report Generation and Processing Scripts

These scripts handle the generation of reports based on simulation results and manage the process of sweeping and filtering the results.

### Usage Example

Below is an example of initializing the `TakerMakerDeeperLevelSpread` class and running a simulation:

```python
# Initialize the class with required parameters
taker_maker = TakerMakerDeeperLevelSpread(df=data, spot_fee=0.001, swap_fee=0.002, ...)

# Example to run simulation for specific depth
params = {
    'band': 'percentage_band',
    'lookback': 100,
    'recomputation_time': 60,
    't_start': datetime.datetime(2023, 1, 1),
    't_end': datetime.datetime(2023, 1, 2),
    # Additional parameters
}

simulation_trader_taker_maker_at_depth(params)
```

This setup ensures that the trading environment is simulated with predefined parameters, giving insights into trading strategies' outcomes.