from datetime import datetime
from src.common.utils.utils import parse_args
import argparse
from src.scripts.wandb_sweeps.process_sweep_results.download_wandb_results import AutomateParameterSelectionEthusd, \
    AutomateParameterSelectionXbtusd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def list_of_strings(arg):
    return arg.split(',')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--strategy_family', default='ETHUSD', type=str)
parser.add_argument('--project_name', default='taker_maker_simulations_2023', type=str)
parser.add_argument('--sweep_id_training', default=None, type=str)
parser.add_argument('--sweep_id_confirmations', default=None, type=list_of_strings)
params = vars(parse_args(parser))

if params['strategy_family'] == 'ETHUSD':
    data_processing = AutomateParameterSelectionEthusd(project_name=params['project_name'])
elif params['strategy_family'] == 'XBTUSD':
    data_processing = AutomateParameterSelectionXbtusd(project_name=params['project_name'])

df = data_processing.combine_results_to_single_df(sweep_id_confirm=params['sweep_id_confirmations'],
                                                  sweep_id_training=params['sweep_id_training'])
df1 = data_processing.combined_results_df(df)
df1.to_csv(f'/results/Combined_Results_{params["strategy_family"]}_{datetime.date.today()}.csv', index=False)
