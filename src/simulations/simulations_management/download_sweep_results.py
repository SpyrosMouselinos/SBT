import pandas as pd
import wandb
import os
from src.common.constants.constants import WANDB_ENTITY, TAKER_MAKER_PROJECT_2023
from dotenv import find_dotenv, load_dotenv
from pytictoc import TicToc
from tqdm import tqdm

load_dotenv(find_dotenv())


def sweep_rerun_simulations(sweep_id: str = 'jd1a03uf', select_num_simulations: int = 30, custom_filter: str = 'no',
                            project_name: str = 'taker_maker_simulations_2023_2'):
    # st.write('hello')
    # st.write(os.getenv("WANDB_HOST"))
    # st.write(os.getenv("WANDB_API_KEY"))
    t = TicToc()
    t.tic()
    wandb.login(host=os.getenv("WANDB_HOST"), key="local-c079a1f81a639c9546d4e0a7790074d341572ef7")

    try:
        api = wandb.Api({'entity': WANDB_ENTITY, 'project': f'{project_name}', 'sweep': f'{sweep_id}'}, timeout=300)
        runs = api.runs(f"{WANDB_ENTITY}/{project_name}",
                        filters={"sweep": sweep_id},
                        order="-summary_metrics.Estimated PNL with Funding"
                        )
        if len(runs) == 0:
            api = wandb.Api({'entity': WANDB_ENTITY, 'project': f'{project_name}', 'sweep': f'{sweep_id}'},
                            timeout=300)
            runs = api.runs(
                f"{WANDB_ENTITY}/{project_name}",
                filters={"sweep": sweep_id},
                order="-summary_metrics.Estimated PNL with Quanto_profit"
            )
    except:
        api = wandb.Api({'entity': WANDB_ENTITY, 'project': TAKER_MAKER_PROJECT_2023, 'sweep': f'{sweep_id}'},
                        timeout=300)
        try:
            runs = api.runs(
                f"{WANDB_ENTITY}/{TAKER_MAKER_PROJECT_2023}",
                filters={"sweep": sweep_id},
                order="-summary_metrics.Estimated PNL with Quanto_profit"
            )
        except:
            runs = api.runs(
                f"{WANDB_ENTITY}/{TAKER_MAKER_PROJECT_2023}",
                filters={"sweep": sweep_id},
                order="-summary_metrics.Estimated PNL with Funding"
            )
    params = []
    print(project_name)
    print(sweep_id)
    print(len(runs))
    for idx, run in tqdm(enumerate(runs)):
        if idx == 0:
            config_dict = run.config
            params.append([idx, config_dict])
            run.summary._json_dict.update(config_dict)
            sweep_results_df = pd.DataFrame(run.summary._json_dict, index=[0])
            sweep_results_df = sweep_results_df.reindex(sorted(sweep_results_df.columns), axis=1)

        if len(dict(run.summary)) > 1 and idx != 0:
            config_dict = run.config
            params.append([idx, config_dict])
            run.summary._json_dict.update(config_dict)
            df = pd.DataFrame(run.summary._json_dict, index=[0])
            df = df.reindex(sorted(df.columns), axis=1)
            sweep_results_df = pd.concat([sweep_results_df, df], ignore_index=True)
    try:
        sweep_results_df.sort_values(by='Estimated PNL with Quanto_profit', inplace=True, ignore_index=False,
                                     ascending=False)
    except:
        sweep_results_df.sort_values(by='Estimated PNL with Funding', inplace=True, ignore_index=False,
                                     ascending=False)

    sweep_results_df['param_index'] = sweep_results_df.index
    sweep_results_df.reset_index(inplace=True, drop=True)
    params_list = []
    if custom_filter == 'filter1':
        try:
            indexes_list = sweep_results_df.loc[(sweep_results_df['entry_delta_spread'] <= 2.15) &
                                                (sweep_results_df['Funding in Total'] >= -3300) &
                                                (sweep_results_df['ratio_entry_band_mov'] <= -0.67) &
                                                (sweep_results_df['ratio_exit_band_mov'] >= 0.63) &
                                                (sweep_results_df['Coin Volume in entry side'] >= 380) &
                                                (sweep_results_df['Estimated PNL with Quanto_profit'] >= 2000),
            "param_index"].to_list()
        except:
            indexes_list = []
    elif custom_filter == 'filter2':
        try:
            indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= -1820) &
                                                (sweep_results_df['Entry 26 to 28 Volume Oct in USD'] >= 98000) &
                                                (sweep_results_df['Entry 14 to 17 Volume Oct in USD'] >= 50000) &
                                                (sweep_results_df['Entry 08 Nov Volume in USD'] >= 146000),
            "param_index"].to_list()
        except:
            indexes_list = []
    elif custom_filter == 'filter3':
        try:
            indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= -1584) &
                                                (sweep_results_df['Entry 26 to 28 Volume Oct in USD'] >= 74400) &
                                                (sweep_results_df['Entry 14 to 17 Volume Oct in USD'] >= 98933) &
                                                (sweep_results_df['Entry 08 Nov Volume in USD'] >= 95200) &
                                                (sweep_results_df['Estimated PNL with Quanto_profit'] >= 4000),
            "param_index"].to_list()
        except:
            indexes_list = []
    elif custom_filter == 'filter4':
        try:
            indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= -2079) &
                                                (sweep_results_df['Entry 26 to 28 Volume Oct in USD'] >= 240000) &
                                                (sweep_results_df['Entry 14 to 17 Volume Oct in USD'] >= 145333) &
                                                (sweep_results_df['Entry 08 Nov Volume in USD'] >= 190399) &
                                                (sweep_results_df['F8'] >= 3000),
            "param_index"].to_list()
        except:
            indexes_list = []
    elif custom_filter == 'filter5':
        indexes_list = sweep_results_df.loc[sweep_results_df['F8'] >= 2970, "param_index"].to_list()

    elif custom_filter == 'filter6':
        indexes_list = sweep_results_df.loc[sweep_results_df['Estimated PNL with Quanto_profit'] >= 1700,
        "param_index"].to_list()

    elif custom_filter == 'filter7':
        indexes_list = sweep_results_df.loc[sweep_results_df['Estimated PNL with Quanto_profit'] >= 350,
        "param_index"].to_list()

    elif custom_filter == 'filter8':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= -2000) &
                                            (sweep_results_df['Sharpe Ratio'] >= 5) &
                                            (sweep_results_df['ROR Annualized'] >= 30), "param_index"].to_list()
    elif custom_filter == 'filter9':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= -200) &
                                            (sweep_results_df['funding_options'] == 'option1') &
                                            (sweep_results_df['Sharpe Ratio'] >= 6) &
                                            (sweep_results_df[
                                                 'Estimated PNL with Funding'] >= 2000), "param_index"].to_list()
    elif custom_filter == 'filter10':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Funding in Total'] >= 1000) &
                                            (sweep_results_df['Sharpe Ratio'] >= 4) &
                                            (sweep_results_df['Estimated PNL with Realized Quanto_profit'] >= 6200),
        "param_index"].to_list()

    elif custom_filter == 'filter11':
        indexes_list = sweep_results_df.loc[sweep_results_df['Estimated PNL with Quanto_profit'] >= 4000,
        "param_index"].to_list()

    elif custom_filter == 'global_filter':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Average_Distance'].astype(float) <= 1.5) &
                                            (sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["Std_daily_ROR"].astype(float) <= 0.01) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 30) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 4),
        "param_index"].to_list()
    elif custom_filter == 'global_filter_low':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Average_Distance'].astype(float) <= 1.5) &
                                            (sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["Std_daily_ROR"].astype(float) <= 0.01) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 20) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 4),
        "param_index"].to_list()
        try:
            if (not pd.isna(params[0][1]['t_start_period'])) or (not pd.isna(params[0][1]['t_end_period'])):
                indexes_list = sweep_results_df.loc[(sweep_results_df['Average_Distance'].astype(float) <= 1.5) &
                                                    (sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                                    (sweep_results_df["Std_daily_ROR"].astype(float) <= 0.01) &
                                                    (sweep_results_df["ROR Annualized"].astype(float) >= 30) &
                                                    (sweep_results_df["ROR_overall"].astype(float) >= 15) &
                                                    (sweep_results_df["Sharpe Ratio"].astype(float) >= 4) &
                                                    (sweep_results_df["Sharpe_total"].astype(float) >= 4),
                "param_index"].to_list()
        except:
            print('no multi-period')
    elif custom_filter == 'global_filter_ultra_low':
        indexes_list = sweep_results_df.loc[(sweep_results_df['Average_Distance'].astype(float) <= 1.5) &
                                            (sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["Std_daily_ROR"].astype(float) <= 0.01) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 8) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 4),
        "param_index"].to_list()
    elif custom_filter == 'global_filter_low_new':
        indexes_list = sweep_results_df.loc[(sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 15) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 4),
        "param_index"].to_list()
    elif custom_filter == "eth_multiperiod":
        indexes_list = sweep_results_df.loc[(sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p1"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p2"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p3"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p4"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p5"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p6"].astype(float) <= 1.7) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p1"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p2"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p3"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p4"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p5"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p6"].astype(float) >= 3) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p1"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p2"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p3"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p4"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p5"].astype(float) >= 25) &
                                            (sweep_results_df["ROR Annualized_p6"].astype(float) >= 25),
        "param_index"].to_list()
    elif custom_filter == "eth_multiperiod_low":
        indexes_list = sweep_results_df.loc[(sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p1"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p2"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p3"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p4"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p5"].astype(float) <= 1.7) &
                                            (sweep_results_df["max_drawdown_d_p6"].astype(float) <= 1.7) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p1"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p2"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p3"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p4"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p5"].astype(float) >= 3) &
                                            (sweep_results_df["Sharpe Ratio_p6"].astype(float) >= 3) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p1"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p2"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p3"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p4"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p5"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p6"].astype(float) >= 25),
        "param_index"].to_list()
    elif custom_filter == "eth_multiperiod_low_new":
        indexes_list = sweep_results_df.loc[(sweep_results_df["max_drawdown_d"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p1"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p2"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p3"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p4"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p5"].astype(float) <= 1.8) &
                                            (sweep_results_df["max_drawdown_d_p6"].astype(float) <= 1.8) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p1"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p2"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p3"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p4"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p5"].astype(float) >= 2) &
                                            (sweep_results_df["Sharpe Ratio_p6"].astype(float) >= 4) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p1"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p2"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p3"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p4"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p5"].astype(float) >= 0) &
                                            (sweep_results_df["ROR Annualized_p6"].astype(float) >= 25),
        "param_index"].to_list()
    else:
        indexes_list = sweep_results_df.loc[:select_num_simulations, "param_index"].to_list()

    if len(indexes_list) == 0 and custom_filter in ['global_filter_low_new', 'global_filter_low',
                                                    'global_filter_ultra_low']:
        indexes_list = sweep_results_df.loc[(sweep_results_df["max_drawdown_d"].astype(float) <= 1.7) &
                                            (sweep_results_df["ROR Annualized"].astype(float) >= 8) &
                                            (sweep_results_df["Sharpe Ratio"].astype(float) >= 2),
        "param_index"].to_list()

    for idx in indexes_list:
        params_list.append(params[idx][1])

    params_list_final = []
    valid_keys = list(params_list[0].keys())
    try:
        valid_keys.remove('t_end')
        valid_keys.remove('t_start')
    except:
        try:
            valid_keys.remove('t_end_period')
            valid_keys.remove('t_start_period')
            valid_keys.remove('train_multiple_periods')
        except:
            pass
    for idx in range(len(indexes_list)):
        data_dict = {valid_keys[ix]: {} for ix in range(len(valid_keys))}
        for i in range(len(valid_keys)):
            upd_dict = {'value': params_list[idx][valid_keys[i]]}
            data_dict[valid_keys[i]].update(upd_dict)
        params_list_final.append(data_dict)
    t.toc()
    if len(params_list_final) >= 3000:
        return params_list_final[:3000]
    return params_list_final


if __name__ == '__main__':
    sweep_rerun_simulations(sweep_id='shcva425', custom_filter='global_filter_low_new',
                            project_name='xbtusd_bitmex_high_funding_window_29_02_2024')
