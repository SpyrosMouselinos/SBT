# Documentation for `src/scripts` Folder

## Overview
The `src/scripts` folder contains various Python scripts aimed at tasks related to data backfilling, processing results of swept simulations particularly involving Weights & Biases (WandB), backups and alerts, and trading opportunities. The goal of these scripts is to facilitate the automation, data retrieval, and result processing of various trading strategies and simulations.

## Folder Structure

- `data_backfilling`
  - `backfill_taker_trades.py`
  - `backfill_opportunities.py`
- `backups_and_alerts`
  - `fetch_unhedged.py`
  - `posting_alerts.py`
- `wandb_sweeps`
  - `process_sweep_results`
    - `download_wandb_results.py`
    - `combine_sweep_results.py`
    - `mapping_simulations_real_results.py`
    - `controller_specific_combinations.py`
  - `maker_taker`
    - `taker_maker_quanto_contracts.py`
    - `taker_maker_inverse_contracts.py`

## Detailed Description of Scripts

### 1. **Data Backfilling**

#### 1.1 `backfill_taker_trades.py`
This script handles the backfilling of taker trades data. It connects to an InfluxDB client to write points into the database, ensuring that trades are correctly noted and reconciled.

#### 1.2 `backfill_opportunities.py`
The `backfill_opportunities.py` script focuses on backfilling opportunity points data. It connects to multiple databases for fetching existing trade data and reprocesses opportunities for various trading strategies.

```python
import warnings
import time
import math
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import numpy as np
from tqdm import tqdm
from src.common.queries.queries import Funding, Prices, Takers
from src.common.queries.queries import get_entry_opportunity_points, get_exit_opportunity_points
from src.common.connections.ExchangeConnections import Deribit, BitMEX
from src.common.connections.DatabaseConnections import InfluxConnection
from src.common.utils.utils import aggregated_volume
```

### 2. **Backups and Alerts**

#### 2.1 `fetch_unhedged.py`
The script `fetch_unhedged.py` retrieves and updates unhedged amounts for a specific strategy by querying data from PostgreSQL and BitMEX. It attempts to fill in missing price data in the database entries for unhedged events.

```python
import json
import psycopg2, os
import requests
import datetime
import urllib.parse
import time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_unhedged():
    connection = psycopg2.connect(
        user=os.getenv("POSTGRES_USERNAME"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="pgbouncer",
        port=6432
    )
    cursor = connection.cursor()
    cursor.execute("SELECT id, timestamp FROM strategy_mark_to_market_unhedged_amount_events WHERE price is NULL ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    cursor.close()
    # [Continues with processing each row]
```

#### 2.2 `posting_alerts.py`
This script checks for active strategies and verifies their posting and trading status via logs. In case a strategy has not been posting or trading for more than a day, it sends out alerts.

```python
import requests
import json
from time import sleep
from src.common.connections.DatabaseConnections import InfluxMeasurements

def posting_alerts(influx_measurements):
    strategies = influx_measurements.get_active_strategies()
    # [Continues with logic to post alerts]
```


### 3. **Weights & Biases Sweeps**

#### 3.1 `process_sweep_results` Subdirectory

##### 3.1.1 `download_wandb_results.py`
This script is designed to download and compile results from a specified sweep in Weights & Biases. It connects to the WandB API, retrieves the results and processes them into a comprehensive DataFrame.

##### 3.1.2 `combine_sweep_results.py`
Combines results from multiple sweeps into a single summarized DataFrame. Designed to process sweeps for specific trading pairs like ETHUSD or XBTUSD.

```python
from datetime import datetime
from src.common.utils.utils import parse_args
import argparse
from src.scripts.wandb_sweeps.process_sweep_results.download_wandb_results import AutomateParameterSelectionEthusd, AutomateParameterSelectionXbtusd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--strategy_family', default='ETHUSD', type=str)
```

##### 3.1.3 `mapping_simulations_real_results.py`
Maps real trading parameters to simulation parameters to ensure that configurations used in backtests align closely with real-world trading settings.

```python
import pandas as pd
import numpy as np
from src.scripts.wandb_sweeps.process_sweep_results.download_wandb_results import AutomateParameterSelection

def mapping_real_params_to_simulation_params(df):
    rename_variables = {
        "window_size.1": "window_size2",
        # [Other mappings]
    }
    df.drop(columns=['rolling_time_window_size'], inplace=True)
    df.rename(columns=rename_variables, inplace=True)
    df["quanto_threshold"] = - df["quanto_threshold"]
    return df
```

##### 3.1.4 `controller_specific_combinations.py`
Generates and handles combinations of trading parameters specific to certain controllers, which can be used in simulations.

```python
import argparse
import os
from dotenv import find_dotenv, load_dotenv
import wandb
from human_id import generate_id

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sweep_id', default=None, type=str)
```

#### 3.2 `maker_taker` Subdirectory

##### 3.2.1 `taker_maker_quanto_contracts.py`
Manages trading logic and parameters for quanto contracts in taker-maker strategies.

##### 3.2.2 `taker_maker_inverse_contracts.py`
Focuses on inverse contracts within taker-maker strategies, handling simulation logic and execution setups.

## Usage

To execute any of these scripts, navigate to the root directory of the project and run the script using Python. For example, to backfill taker trades:

```bash
python src/scripts/data_backfilling/backfill_taker_trades.py
```

Ensure that the necessary environment variables are set, particularly for database connections and API keys, either through a `.env` file or by exporting them in your shell environment.

This documentation summarizes the key aspects of the scripts found in the `src/scripts` folder, providing a high-level overview and details for specific files. For complete functionality, refer to the script files and ensure all dependencies and environment configurations are properly set up.