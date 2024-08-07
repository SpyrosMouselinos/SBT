IS_ANY_SWEEP_RUNNING = ''' 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep "sweep_" | awk "NR==1{print $1}"' '''
IS_ANY_CONTROLLER_RUNNING = ''' 'ps -aux | grep "controller_specific_combinations.py" | wc -l' '''

image = ["local/automated_simulation_ethusd_multi:v0.1", "local/automated_simulation_ethusd:v0.1"]

COLS = [
    "window_size",
    "current_r",
    "high_r",
    "hours_to_stop",
    "quanto_threshold",
    "ratio_entry_band_mov_ind",
    "rolling_time_window_size",
    "band_funding_system"
]

MAXIMUM_RESULTS = 13000

CONTROLLER_CONF = {
    "controller": {
        "type": "local"
    },
    "method": "grid",
    "parameters": {
        "t_end": {
            "distribution": "constant",

        },
        "t_start": {
            "distribution": "constant",

        }
    },
    "program": "src/scripts/wandb_sweeps/maker_taker_ethusd_short_conditional_r_revert.py"
}

CONTROLLER_CONFS_TIMESTAMPS = [
    {"t_start": 1672794000000, "t_end": 1675040400000},
    {"t_start": 1672794000000, "t_end": 1675040400000},
    {"t_start": 1672794000000, "t_end": 1675040400000},
    {"t_start": 1672794000000, "t_end": 1675040400000}
]

CONTROLLER_A_CONF = {
    "controller": {
        "type": "local"
    },
    "method": "grid",
    "parameters": {
        "t_end": {
            "distribution": "constant",
            "value": 1667944800000
        },
        "t_start": {
            "distribution": "constant",
            "value": 1665262800000
        }
    }
}

CONTROLLER_B_CONF = {
    "controller": {
        "type": "local"
    },
    "method": "grid",
    "parameters": {
        "t_end": {
            "distribution": "constant",
            "value": 1662930000000
        },
        "t_start": {
            "distribution": "constant",
            "value": 1660510800000
        }
    }
}

CONTROLLER_C_CONF = {
    "controller": {
        "type": "local"
    },
    "method": "grid",
    "parameters": {
        "t_end": {
            "distribution": "constant",
            "value": 1672794000000
        },
        "t_start": {
            "distribution": "constant",
            "value": 1668474000000
        }
    }
}

CONTROLLER_D_CONF = {
    "controller": {
        "type": "local"
    },
    "method": "grid",
    "parameters": {
        "t_end": {
            "distribution": "constant",
            "value": 1675040400000
        },
        "t_start": {
            "distribution": "constant",
            "value": 1672794000000
        }
    }
}

SYMBOL_PROGRAM = {
    "ETHUSD": "src/scripts/wandb_sweeps/maker_taker_ethusd_short_conditional_r_revert.py",
    "XBTUSD": "src/scripts/wandb_sweeps/taker_maker_inverse_contracts.py"
}

CONTROLLER_CONFS = [CONTROLLER_A_CONF, CONTROLLER_B_CONF, CONTROLLER_C_CONF, CONTROLLER_D_CONF]

HOSTS = [
    "simulations-dedicated-17", "simulations-dedicated-18"
]
