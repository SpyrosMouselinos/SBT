# Detailed Documentation for the `src/streamlit` Folder

## Overview

The `src/streamlit` folder contains several scripts that leverage the **Streamlit** framework to create interactive data applications and trading simulators. These applications are designed to analyze trading strategies, generate reports, and manage simulations. The Streamlit framework allows these applications to be interactive and user-friendly, providing essential interfaces for trader simulation and report generation.

### Key Components:

1. **streamlit_maker_taker_report_genarator.py**
2. **streamlit_maker_maker_report_genarator.py**
3. **streamlit_message_to_send_maker_maker.py**
4. **streamlit_page_taker_maker_at_depth.py**
5. **streamlit_maker_taker_report_genarator_mutlicoin.py**
6. **streamlit_page_maker_maker.py**

---

## 1. streamlit_maker_taker_report_genarator.py

### Overview:
This script is designed to generate various reports for the Maker-Taker simulation, providing insights into trading strategies over specified time periods.

### Features:
- Date range selection for the report generation.
- Selection of the type of bands to use for simulations.
- Integration with Backblaze for data storage and retrieval.
- Visualization of trading data using Plotly graphs.

### Sample Function: `report_generator_maker_taker`

```python
def report_generator_maker_taker():
    funding_system_list = ['Quanto_loss', 'Quanto_profit', 'Quanto_profit_BOX', 'Quanto_profit_exp', 
                           'Quanto_both', 'Quanto_both_extended']
    st.title('Maker Taker Simulations Report Generator')
    date_range = st.date_input("Enter a period where the report is generated", 
                               [datetime.date.today() - datetime.timedelta(days=7), 
                               datetime.date.today() + datetime.timedelta(days=1)])
    t_start_search = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month, 
                                           day=date_range[0].day).timestamp() * 1000)
    t_end_search = int(datetime.datetime(year=date_range[1].year, month=date_range[1].month, 
                                         day=date_range[1].day).timestamp() * 1000)
```

## 2. streamlit_maker_maker_report_genarator.py

### Overview:
This script focuses on generating reports for Maker-Maker simulations, which analyze trading strategies that involve both sides of a trade being made by a maker.

### Features:
- Customizable date ranges for report generation.
- Options to disable opportunity points creation.
- Integration with plotting libraries such as HiPlot and Plotly.
- Data authorization and retrieval from Backblaze.

### Sample Function: `report_generator_maker_maker`

```python
def report_generator_maker_maker():
    disable_opp = st.sidebar.selectbox('Disable Opportunity Points creation', ('yes', 'no'))
    store_opp = st.sidebar.selectbox('Store the Opportunity points', ('no', 'yes'))
    date_range = st.date_input("Enter a period where the report is generated", 
                               [datetime.date.today() - datetime.timedelta(days=7), datetime.date.today()])
    t_start_search = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month, 
                                           day=date_range[0].day).timestamp() * 1000)
    t_end_search = int(datetime.datetime(year=date_range[1].year, month=date_range[1].month, 
                                         day=date_range[1].day).timestamp() * 1000)
```

## 3. streamlit_message_to_send_maker_maker.py

### Overview:
This script provides an interface to send messages for Maker-Maker simulations, facilitating the discovery of new trading opportunities.

### Features:
- Sidebar control elements for quick actions like clearing caches.
- Ability to select and customize bands and trading strategies.
- Use of RPCClient for managing simulation messages.

### Sample Function: `streamlit_trader_message`

```python
def streamlit_trader_message():
    if st.button("Clear All"):
        st.experimental_singleton.clear()
    st.sidebar.write("Trader Simulator in order to discover new markets and trading opportunities")
    st.title('Trading Simulator')
    band = st.sidebar.selectbox('Select the type of bands you want to use', 
                                ('bogdan_bands', 'percentage_bogdan_bands', 'quanto_profit', 'percentage_band'))
    if band == 'percentage_band':
        lookback = st.sidebar.text_input('lookback')
        recomputation_time = st.sidebar.text_input('recomputation_time')
```

## 4. streamlit_page_taker_maker_at_depth.py

### Overview:
This script provides a more in-depth analysis of Taker-Maker trading by looking into deeper levels of trading data.

### Features:
- Different customizable parameters to simulate trading at various depths.
- Visualization features for comprehensively analyzing trading metrics.
- Integration with common trading and data analysis libraries.

### Sample Function: `run_simulation`

```python
def run_simulation():
    t = TicToc()
    if st.button("Clear All"):
        st.experimental_singleton.clear()
    st.sidebar.write("Taker maker trading with at-depth posting")
    st.title('Trading Simulator')
    band = st.sidebar.selectbox('Select the type of bands you want to use', 
                                ('bogdan_bands', 'percentage_bogdan_bands', 'quanto_profit', 'percentage_band'))
    date_range = st.date_input("Input a range of time report", 
                               [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
    t_start = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month, 
                                    day=date_range[0].day).timestamp() * 1000)
```

## 5. streamlit_maker_taker_report_genarator_mutlicoin.py

### Overview:
This script extends the Maker-Taker report generation by incorporating multiple coins, allowing simultaneous analysis across different cryptocurrencies.

### Features:
- Multicoin simulation and reporting.
- Date range selection for simulation periods.
- Detailed breakdown of simulation parameters and results.

### Sample Function: `report_generator_maker_taker_multicoin`

```python
def report_generator_maker_taker_multicoin():
    st.title('Maker Taker Simulations Report Generator Multicoin')
    date_range = stp.date_input("Enter a period where the report is generated",
                                [datetime.date.today() - datetime.timedelta(days=7), 
                                datetime.date.today() + datetime.timedelta(days=1)])
    t_start_search = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month,
                                           day=date_range[0].day).timestamp() * 1000)
    t_end_search = int(datetime.datetime(year=date_range[1].year, month=date_range[1].month,
                                         day=date_range[1].day).timestamp() * 1000)
```

## 6. streamlit_page_maker_maker.py

### Overview:
This script provides an interactive page specifically for Maker-Maker trading simulations, focusing on strategy discovery and trading volume analysis.

### Features:
- Enables strategic discovery through detailed simulations.
- Interactive visual displays to analyze trading volumes.
- Configurable inputs for trading parameters.

### Sample Function: `streamlit_maker_maker_trader`

```python
def streamlit_maker_maker_trader():
    t = TicToc()
    if st.button("Clear All"):
        st.experimental_singleton.clear()
    st.sidebar.write("Trader Simulator in order to discover new markets and trading opportunities")
    st.title('Trading Simulator')
    band = st.sidebar.selectbox('Select the type of bands you want to use', 
                                ('bogdan_bands', 'percentage_bogdan_bands', 'quanto_profit', 'percentage_band'))
    date_range = st.date_input("Input a range of time report", 
                               [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
    t_start = int(datetime.datetime(year=date_range[0].year, month=date_range[0].month, 
                                    day=date_range[0].day).timestamp() * 1000)
```

---

## Common Imports and Dependencies

Here are some common imports and libraries used across the scripts in `src/streamlit`:

- **Libraries**: 
  - Streamlit
  - NumPy
  - Pandas
  - Plotly
  - Datetime
  - dotenv

### Requirements (`requirements.txt`)

```plaintext
pandas~=2.2.1
numpy~=1.24.0
numba~=0.59.0
scikit-learn~=0.24.2
streamlit~=1.36.0
click==8.0.4
requests~=2.31.0
python-dotenv~=0.17.1
configparser~=5.0.2
flask~=2.0.1
influxdb~=5.3.1
scipy~=1.8
urllib3~=1.25.11
simplejson~=3.17.2
mlflow~=2.10.1
pyqtgraph~=0.11.0
vispy~=0.7.0
wandb~=0.12.21
```

### Example of Common Imports

```python
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
```

---

This documentation should provide a comprehensive understanding of the scripts located within the `src/streamlit` folder, including their purposes, functionalities, and sample code snippets. Each script serves as a key component in facilitating interactive trading simulations and reports, leveraging the power of Streamlit to create visually engaging and user-friendly applications.